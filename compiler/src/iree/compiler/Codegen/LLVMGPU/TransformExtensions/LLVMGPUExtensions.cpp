// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "LLVMGPUExtensions.h"

#include "iree-dialects/Dialect/LinalgTransform/SimplePatternRewriter.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/Region.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE;

iree_compiler::IREE::transform_dialect::LLVMGPUExtensions::LLVMGPUExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensionsOps.cpp.inc"
      >();
}

void mlir::iree_compiler::registerTransformDialectLLVMGPUExtension(
    DialectRegistry &registry) {
  registry.addExtensions<transform_dialect::LLVMGPUExtensions>();
}

//===---------------------------------------------------------------------===//
// IREE-specific LLVMGPU transformations.
//===---------------------------------------------------------------------===//

// TODO: if the number of threads was wired like the workgroup_count, we could
// reuse most of the code and not require a static number of threads.
// TODO: synchronizations for imperfectly nested stuff.
DiagnosedSilenceableFailure
transform_dialect::MapNestedForeachThreadToGpuThreads::applyToOne(
    func::FuncOp target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  if (!isa<HAL::ExecutableOp, HAL::ExecutableVariantOp>(state.getTopLevel())) {
    state.getTopLevel()->emitOpError(
        "requires HAL::ExecutableOp or HAL::ExecutableVariantOp toplevel to "
        "attach the workgroup size information to a nested ExecutableExportOp");
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  }

  IREE::HAL::ExecutableExportOp exportOp;
  state.getTopLevel()->walk([&](IREE::HAL::ExecutableExportOp op) {
    if (op.getSymName() == target.getName()) exportOp = op;
  });
  if (!exportOp) {
    state.getTopLevel()->emitOpError("no IREE::HAL::ExecutableExportOp found");
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  }

  SmallVector<int64_t> workgroupSize =
      extractFromI64ArrayAttr(getWorkgroupSize());
  // TODO: no magic constant but IREE uses this extensively.
  workgroupSize.resize(/*size=*/3, /*value=*/1);

  auto transformOp = cast<transform::TransformOpInterface>(getOperation());
  SimplePatternRewriter rewriter(target);
  DiagnosedSilenceableFailure diag =
      mlir::transform::gpu::mapNestedForeachToThreadsImpl(
          rewriter, target, workgroupSize, true, transformOp);
  if (diag.succeeded()) {
    auto newAttr = rewriter.getIndexArrayAttr(workgroupSize);
    // TODO: should really be: exportOp.setWorkgroupSizeAttr(newAttr);
    exportOp->setAttr(exportOp.getWorkgroupSizeAttrName(), newAttr);
  }
  results.assign({target});
  return diag;
}

//===---------------------------------------------------------------------===//
// VectorToWarpExecuteOnLane0Op.
//===---------------------------------------------------------------------===//

/// Helper method to replace all uses of the laneId operand by the constant
/// 0 inside the region. This is a necessary prerequisite to perform any kind of
/// hoisting of IR that is inside the region.
/// Return success if any replacement occurred, failure otherwise.
// TODO: this is currently brittle, what we really need here is a scope-aware
// SCCP.
static LogicalResult replaceAllUsesOfLaneWithin(
    RewriterBase &b, vector::WarpExecuteOnLane0Op executeOp) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(executeOp);
  Value zero = b.create<arith::ConstantIndexOp>(executeOp.getLoc(), 0);
  b.setInsertionPointToStart(&executeOp.getWarpRegion().front());
  Value laneId = executeOp.getLaneid();
  bool applied = false;
  for (Operation *user : llvm::make_early_inc_range(laneId.getUsers())) {
    if (!executeOp->isProperAncestor(user)) continue;
    b.startRootUpdate(user);
    user->replaceUsesOfWith(laneId, zero);
    b.finalizeRootUpdate(user);
    applied = true;
  }
  return success(applied);
}

/// Return the gpu::ThreadIdOp for which the predicate if equivalent to
/// `if (threadIdx.x == 0)`.
// TODO: Figure out the proper canonicalization and drop the complexity here.
// TODO: More sophisticated detection for matching
//   (threadIdx.x == 0 && other stuff not involving threadIdx.x)
static FailureOr<gpu::ThreadIdOp> isThreadIdxxZeroPredicate(scf::IfOp ifOp) {
  if (!ifOp || ifOp.getNumResults() > 0 ||
      ifOp.getThenRegion().getBlocks().size() != 1 ||
      !ifOp.getElseRegion().empty())
    return failure();
  auto pred = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!pred) return failure();
  auto EQ = arith::CmpIPredicate::eq;
  auto SLT = arith::CmpIPredicate::slt;
  auto SLE = arith::CmpIPredicate::sle;
  auto ULT = arith::CmpIPredicate::ult;
  auto ULE = arith::CmpIPredicate::ule;
  if (auto threadIdOp = pred.getLhs().getDefiningOp<gpu::ThreadIdOp>()) {
    if (threadIdOp.dimension() != gpu::Dimension::x) return failure();
    if (pred.getPredicate() == EQ && isConstantIntValue(pred.getRhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SLE && isConstantIntValue(pred.getRhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == ULE && isConstantIntValue(pred.getRhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SLT && isConstantIntValue(pred.getRhs(), 1))
      return threadIdOp;
    if (pred.getPredicate() == ULT && isConstantIntValue(pred.getRhs(), 1))
      return threadIdOp;
  }
  auto SGT = arith::CmpIPredicate::sgt;
  auto SGE = arith::CmpIPredicate::sge;
  auto UGT = arith::CmpIPredicate::ugt;
  auto UGE = arith::CmpIPredicate::uge;
  if (auto threadIdOp = pred.getRhs().getDefiningOp<gpu::ThreadIdOp>()) {
    if (threadIdOp.dimension() != gpu::Dimension::x) return failure();
    if (pred.getPredicate() == EQ && isConstantIntValue(pred.getLhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SGE && isConstantIntValue(pred.getLhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == UGE && isConstantIntValue(pred.getLhs(), 0))
      return threadIdOp;
    if (pred.getPredicate() == SGT && isConstantIntValue(pred.getLhs(), 1))
      return threadIdOp;
    if (pred.getPredicate() == UGT && isConstantIntValue(pred.getLhs(), 1))
      return threadIdOp;
  }
  return failure();
}

struct VectorDistributionResult {
  vector::WarpExecuteOnLane0Op warpOp;
};

static FailureOr<VectorDistributionResult> rewriteScfIfAsWarpExecuteOnLane0(
    PatternRewriter &rewriter, Location loc, scf::IfOp ifOp,
    int64_t workgroupSizeX, int64_t warpSize) {
  // Bail if cond is not `if (threadIdx.x == 0)`.
  FailureOr<gpu::ThreadIdOp> maybeThreadIdxxOp =
      isThreadIdxxZeroPredicate(ifOp);
  if (failed(maybeThreadIdxxOp)) return failure();

  // All the code below will be executed on a single warp given a fixed
  // (threadIdxy, threadIdxz).
  // Note, we reuse `maybeThreadIdxxOp` here because we later want to replace
  // this op instance by 0 without relying on CSE or canonicalizations.
  Value threadIdxx = *maybeThreadIdxxOp;

  assert(workgroupSizeX % warpSize == 0);
  if (workgroupSizeX != warpSize) {
    // Add a guard for `threadIdxx < warp size` around the WarpExecuteOnLane0Op.
    Value predicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, threadIdxx,
        rewriter.create<arith::ConstantIndexOp>(loc, warpSize));
    // Note: return-less IfOp is built with a terminator, no need to add one.
    auto newIfOp =
        rewriter.create<scf::IfOp>(loc, predicate, /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&newIfOp.getThenRegion().front());
  }
  auto warpOp = rewriter.create<vector::WarpExecuteOnLane0Op>(
      loc, TypeRange(), threadIdxx, warpSize);

  // Move the code from the previous ifOp to the WarpExecuteOnLane0Op.
  Block &sourceBlock = ifOp.getThenRegion().front();
  Block &targetBlock = warpOp.getWarpRegion().front();
  Block::iterator insertionPoint = targetBlock.begin();
  targetBlock.getOperations().splice(insertionPoint,
                                     sourceBlock.getOperations(),
                                     sourceBlock.without_terminator().begin(),
                                     sourceBlock.without_terminator().end());
  rewriter.setInsertionPointToEnd(&targetBlock);
  rewriter.create<vector::YieldOp>(loc);

  // Erase old op.
  rewriter.eraseOp(ifOp);

  // This simple rewrite propagates zero in lieu of laneId within the
  // warp_execute_on_lane_0 op.
  // Atm, this **must** occur before any hoisting of code.
  // TODO: Replace this by a more robust scoped SCCP that will make it more
  // robust re. hoisting.
  (void)replaceAllUsesOfLaneWithin(rewriter, warpOp);

  // Hoist the scalar code outside of the warp region.
  // Note: moving code does not require a listener.
  vector::moveScalarUniformCode(warpOp);

  return VectorDistributionResult{warpOp};
}

// TODO: Refactor in a generic util that can be reused.
static HAL::ExecutableExportOp getExecutableExportOpForFunc(
    HAL::ExecutableVariantOp halExecutableVariantOp, func::FuncOp funcOp) {
  if (!halExecutableVariantOp || !funcOp) return {};
  HAL::ExecutableExportOp exportOp;
  halExecutableVariantOp->walk([&](HAL::ExecutableExportOp op) {
    if (op.getSymName() != funcOp.getName()) return WalkResult::advance();
    exportOp = op;
    return WalkResult::interrupt();
  });
  return exportOp;
}

DiagnosedSilenceableFailure
transform_dialect::VectorToWarpExecuteOnLane0Op::applyToOne(
    scf::IfOp target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  if (!isa<HAL::ExecutableOp, HAL::ExecutableVariantOp>(state.getTopLevel())) {
    state.getTopLevel()->emitOpError(
        "requires HAL::ExecutableOp or HAL::ExecutableVariantOp toplevel so "
        "that IR is properly isolated. This is required so we can safely "
        "inspect the HAL::ExecutableExportOp under multi-threaded pass "
        "assumptions.");
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  }

  auto halExecutableVariantOp =
      target->getParentOfType<HAL::ExecutableVariantOp>();
  auto funcOp = target->getParentOfType<func::FuncOp>();
  HAL::ExecutableExportOp exportOp =
      getExecutableExportOpForFunc(halExecutableVariantOp, funcOp);
  if (!halExecutableVariantOp || !funcOp || !exportOp) {
    // Return a silenceable failure and set the expected 1 result to nullptr.
    results.assign(1, nullptr);
    return emitDefaultSilenceableFailure(target)
           << "export op is missing --- the transform is not applied";
  }

  Optional<ArrayAttr> maybeAttr = exportOp.getWorkgroupSize();
  // TODO: Pervasive 3 constant in IREE.
  if (!maybeAttr || maybeAttr->size() != 3) {
    // Return a silenceable failure and set the expected 1 result to nullptr.
    results.assign(1, nullptr);
    return emitDefaultSilenceableFailure(target)
           << "export op must have workgroup_size attribute set with 3 entries "
              "--- the transform is not applied";
  }

  int64_t workgroupSizeX = (*maybeAttr)[0].cast<IntegerAttr>().getInt();
  int64_t warpSize = getWarpSize();
  if (workgroupSizeX % warpSize != 0) {
    // Return a silenceable failure and set the expected 1 result to nullptr.
    results.assign(1, nullptr);
    return emitDefaultSilenceableFailure(target)
           << "vector distribution requires workgroup size for x to be a "
           << "multiple of the warp size: " << workgroupSizeX << " vs "
           << warpSize << " --- the transform is not applied";
  }

  SimplePatternRewriter rewriter(target);
  FailureOr<VectorDistributionResult> vectorDistributionResult =
      rewriteScfIfAsWarpExecuteOnLane0(rewriter, target->getLoc(), target,
                                       workgroupSizeX, warpSize);
  if (failed(vectorDistributionResult)) {
    // Return a silenceable failure and set the expected 1 result to nullptr.
    results.assign(1, nullptr);
    return emitDefaultSilenceableFailure(target)
           << "scf::ifOp needs to be predicated on threadIdx.x == 0 --- the "
              "transform is not applied";
  }
  results.assign({vectorDistributionResult->warpOp});
  return DiagnosedSilenceableFailure(success());
}

//===---------------------------------------------------------------------===//
// VectorWarpDistributionOp.
//===---------------------------------------------------------------------===//

/// Emit shared local memory allocation in case it is needed when lowering the
/// warp operations.
static Value allocateGlobalSharedMemory(Location loc, OpBuilder &builder,
                                        vector::WarpExecuteOnLane0Op warpOp,
                                        Type type) {
  MemRefType memrefType;
  if (auto vectorType = type.dyn_cast<VectorType>()) {
    memrefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType(), {},
                        gpu::GPUDialect::getWorkgroupAddressSpace());
  } else {
    memrefType = MemRefType::get({1}, type, {},
                                 gpu::GPUDialect::getWorkgroupAddressSpace());
  }
  return builder.create<memref::AllocOp>(loc, memrefType);
}

/// Emit warp reduction code sequence for a given input.
static Value warpReduction(Location loc, OpBuilder &builder, Value input,
                           vector::CombiningKind kind, uint32_t size) {
  Value laneVal = input;
  // Parallel reduction using butterfly shuffles.
  for (uint64_t i = 1; i < size; i <<= 1) {
    Value shuffled = builder
                         .create<gpu::ShuffleOp>(loc, laneVal, i,
                                                 /*width=*/size,
                                                 /*mode=*/gpu::ShuffleMode::XOR)
                         .getShuffleResult();
    laneVal = makeArithReduction(builder, loc, kind, laneVal, shuffled);
  }
  return laneVal;
}

/// Return a value yielded by `warpOp` which statifies the filter lamdba
/// condition and is not dead.
static OpOperand *getWarpResult(vector::WarpExecuteOnLane0Op warpOp,
                                function_ref<bool(Operation *)> fn) {
  auto yield = cast<vector::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  for (OpOperand &yieldOperand : yield->getOpOperands()) {
    Value yieldValues = yieldOperand.get();
    Operation *definedOp = yieldValues.getDefiningOp();
    if (definedOp && fn(definedOp)) {
      if (!warpOp.getResult(yieldOperand.getOperandNumber()).use_empty())
        return &yieldOperand;
    }
  }
  return {};
}

namespace {

/// Pattern to convert InsertElement to broadcast, this is a workaround until
/// MultiDimReduction distribution is supported.
class InsertElementToBroadcast final
    : public OpRewritePattern<vector::InsertElementOp> {
 public:
  using OpRewritePattern<vector::InsertElementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertElementOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (insertOp.getDestVectorType().getNumElements() != 1) return failure();
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        insertOp, insertOp.getDestVectorType(), insertOp.getSource());
    return success();
  }
};

/// Sink out load op feeding into a warp op yield.
/// ```
/// %0 = vector.warp_execute_on_lane_0(%arg0) -> (f32) {
///   ...
//    %2 = memref.load %src[%c0] : memref<1024xf32>
///   vector.yield %2 : f32
/// }
/// ```
/// To
/// ```
/// %dead = vector.warp_execute_on_lane_0(%arg0) -> (f32) {
///   ...
//    %2 = memref.load %src[%c0] : memref<1024xf32>
///   vector.yield %2 : f32
/// }
/// gpu.synchronize
/// %0 = memref.load %src[%c0] : memref<1024xf32>
struct WarpOpLoad : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(
        warpOp, [](Operation *op) { return isa<memref::LoadOp>(op); });
    if (!operand) return failure();
    auto load = operand->get().getDefiningOp<memref::LoadOp>();
    unsigned operandIndex = operand->getOperandNumber();
    Value distributedVal = warpOp.getResult(operandIndex);

    SmallVector<Value, 4> indices(load.getIndices().begin(),
                                  load.getIndices().end());
    if (!indices.empty()) return failure();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(warpOp);
    // TODO: generalize this.
    // options.warpSyncronizationFn currently must take a WarpExecuteOnLane0Op
    // which we don't have here.
    rewriter.create<gpu::BarrierOp>(load.getLoc());
    Value newRead = rewriter.create<memref::LoadOp>(
        load.getLoc(), distributedVal.getType(), load.getMemref(), indices);

    // The result type of WarpExecuteOnLane0Op may or may not match the yielded
    // type depending on whether the op has "broadcast" behavior (see the doc
    // of WarpExecuteOnLane0Op).
    for (OpOperand &use : distributedVal.getUses()) {
      rewriter.startRootUpdate(use.getOwner());
      Value replacement = newRead;
      if (use.get().getType() != newRead.getType()) {
        replacement = rewriter.create<vector::BroadcastOp>(
            load.getLoc(), use.get().getType(), newRead);
      }
      use.getOwner()->setOperand(use.getOperandNumber(), replacement);
      rewriter.finalizeRootUpdate(use.getOwner());
    }
    return success();
  }
};
}  // namespace

static void populateMultiReductionLoweringPatterns(Operation *target,
                                                   RewritePatternSet &patterns,
                                                   PatternBenefit benefit) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());

  vector::populateVectorMultiReductionLoweringPatterns(
      patterns, vector::VectorMultiReductionLowering::InnerReduction, benefit);
  patterns.add<InsertElementToBroadcast>(target->getContext(), benefit);
}

static AffineMap simpleDistributionFunction(vector::TransferWriteOp writeOp) {
  // Create a map (d0, d1) -> (d1) to distribute along the inner
  // dimension. Once we support n-d distribution we can add more
  // complex cases.
  int64_t vecRank = writeOp.getVectorType().getRank();
  OpBuilder builder(writeOp.getContext());
  auto map = AffineMap::get(vecRank, 0, builder.getAffineDimExpr(vecRank - 1));
  return map;
}

static void populateVectorTransferWriteDistribution(Operation *target,
                                                    RewritePatternSet &patterns,
                                                    PatternBenefit benefit) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());
  vector::populateDistributeTransferWriteOpPatterns(
      patterns, simpleDistributionFunction, benefit);
}

static void populatePropagateVectorDistribution(Operation *target,
                                                RewritePatternSet &patterns,
                                                PatternBenefit benefit) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());
  vector::populatePropagateWarpVectorDistributionPatterns(patterns, benefit);
  vector::populateDistributeReduction(patterns, warpReduction, benefit);
  patterns.add<WarpOpLoad>(target->getContext(), benefit);
}

static void warpSyncronizationFn(Location loc, OpBuilder &builder,
                                 vector::WarpExecuteOnLane0Op warpOp) {
  builder.create<gpu::BarrierOp>(loc);
};

static void populateWarpExecuteOnLane0ToScf(
    Operation *target, RewritePatternSet &patterns,
    const vector::WarpExecuteOnLane0LoweringOptions &options,
    PatternBenefit benefit) {
  assert(target->hasTrait<OpTrait::IsIsolatedFromAbove>());
  vector::populateWarpExecuteOnLane0OpToScfForPattern(patterns, options,
                                                      benefit);
}

DiagnosedSilenceableFailure
transform_dialect::VectorWarpDistributionOp::applyToOne(
    Operation *target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    target->emitOpError(
        "applies only to isolated-from-above targets because it needs to apply "
        "patterns greedily");
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  }

  // TODO: Hook up into the ApplyPatternOp in CommonExtensions.cpp to
  // automatically get listening capabilities.

  MLIRContext *ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  // MultiReduction lowering is necessary until we have explicit support for
  // distributing that op.
  populateMultiReductionLoweringPatterns(target, patterns, /*benefit=*/3);
  populateVectorTransferWriteDistribution(target, patterns, /*benefit=*/2);
  populatePropagateVectorDistribution(target, patterns, /*benefit=*/1);
  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns)))) {
    target->emitOpError("warp distribution patterns failed to apply");
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  }

  RewritePatternSet endPatterns(ctx);
  vector::WarpExecuteOnLane0LoweringOptions options;
  options.warpAllocationFn = allocateGlobalSharedMemory;
  options.warpSyncronizationFn = warpSyncronizationFn;
  populateWarpExecuteOnLane0ToScf(target, endPatterns, options, /*benefit=*/0);
  if (failed(applyPatternsAndFoldGreedily(target, std::move(endPatterns)))) {
    target->emitOpError(
        "warp execute on lane 0 to scf patterns failed to apply");
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  }

  return DiagnosedSilenceableFailure(success());
}

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensionsOps.cpp.inc"