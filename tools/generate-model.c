#include <iree/base/status.h>
#include <iree/base/string_builder.h>
#include <iree/base/string_view.h>
#include <iree/compiler/API2/ToolEntryPoints.h>
#include <iree/hal/buffer_view.h>
#include <iree/hal/buffer_view_util.h>
#include <iree/modules/hal/types.h>
#include <iree/runtime/call.h>
#include <iree/vm/list.h>
#include <iree/vm/module.h>
#include <stdio.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/compiler/API2/MLIRInterop.h"
#include "iree/runtime/api.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/Support.h"

#define FAIL(msg)      \
  {                    \
    printf("error: "); \
    printf(msg);       \
    printf("\n");      \
    exit(1);           \
  }

#define MLIR_STR(s) mlirStringRefCreate(s, strlen(s))

static void bytecode_builder_callback(MlirStringRef str, void *userdata) {
  iree_string_builder_t *builder = (iree_string_builder_t *)userdata;
  iree_string_builder_append_string(
      builder, iree_make_string_view(str.data, str.length));
}

void callback(MlirStringRef text, void *userdata) {
  FILE *f = (FILE *)userdata;
  fwrite(text.data, 1, text.length, f);
}

static void run_sum_reduce_1d(iree_const_byte_span_t bytecode) {
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);

  iree_runtime_instance_t *instance = NULL;
  IREE_CHECK_OK(iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance));

  iree_hal_device_t *device = NULL;
  IREE_CHECK_OK(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view("local-task"), &device));

  iree_string_view_t device_id = iree_hal_device_id(device);
  printf("device id = ");
  fwrite(device_id.data, 1, device_id.size, stdout);
  printf("\n");

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t *session = NULL;

  IREE_CHECK_OK(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session));
  iree_hal_device_release(device);

  IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_memory(
      session, bytecode, iree_allocator_null()));

  iree_runtime_call_t call;

  IREE_CHECK_OK(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.sum_reduce_1d"), &call));


  iree_vm_function_signature_t signature =
      iree_vm_function_signature(&call.function);
  iree_host_size_t out_argument_count, out_result_count;
  iree_vm_function_call_count_arguments_and_results(
      &signature, &out_argument_count, &out_result_count);
  printf("%ld arguments -> %ld results\n", out_argument_count,
         out_result_count);
  printf("callconv = '%s'\n", signature.calling_convention.data);

  iree_hal_buffer_view_t *arg0 = NULL;
  static const iree_hal_dim_t unary_shape[1] = {10};

  const float data[10] = {
      1., 0., 42., 0., 1., 0., 1., 0., 1., 0.,
  };

  iree_hal_buffer_view_allocate_buffer(
      iree_runtime_session_device_allocator(session), 1, unary_shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .access = IREE_HAL_MEMORY_ACCESS_READ,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(data, sizeof(data)), &arg0);

  iree_host_size_t offset;
  iree_hal_buffer_t *buf = iree_hal_buffer_view_buffer(arg0);

  const iree_hal_dim_t two = 2;
  IREE_CHECK_OK(iree_hal_buffer_view_compute_offset(arg0, 1, &two, &offset));

  float val;
  IREE_CHECK_OK(iree_hal_buffer_map_read(buf, offset, &val, sizeof(val)));
  printf("read from buf[%ld] = %f\n", two, val);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);

  IREE_CHECK_OK(iree_hal_buffer_view_append_to_builder(arg0, 10, &builder));
  printf("strbuilder = ");
  fwrite(builder.buffer, 1, builder.size, stdout);
  printf("\ncapacity = %ld\n", builder.capacity);

  IREE_CHECK_OK(iree_hal_buffer_view_fprint(
      stdout, arg0, 10, iree_runtime_session_host_allocator(session)));
  printf("\n");

  // iree_runtime_call_inputs_push_back_buffer_view(&call, arg0);

  IREE_CHECK_OK(iree_vm_list_resize(call.inputs, 1));
  iree_vm_ref_t arg0ref = {0};
  IREE_CHECK_OK(
      iree_vm_ref_wrap_assign(arg0, iree_hal_buffer_view_type_id(), &arg0ref));
  IREE_CHECK_OK(iree_vm_list_set_ref_retain(call.inputs, 0, &arg0ref));

  iree_hal_buffer_view_release(arg0);

  IREE_CHECK_OK(iree_runtime_call_invoke(&call, 0));

  iree_hal_buffer_view_t *ret0 = NULL;
  IREE_CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &ret0));
  IREE_CHECK_OK(iree_hal_buffer_view_fprint(
      stdout, ret0, 1, iree_runtime_session_host_allocator(session)));
  printf("\n");

  iree_hal_buffer_view_release(ret0);
  iree_runtime_call_deinitialize(&call);

  iree_runtime_session_release(session);
}

static void run_bytecode(iree_const_byte_span_t bytecode) {
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);

  iree_runtime_instance_t *instance = NULL;
  IREE_CHECK_OK(iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance));

  iree_hal_device_t *device = NULL;
  IREE_CHECK_OK(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view("local-task"), &device));

  iree_string_view_t device_id = iree_hal_device_id(device);
  printf("device id = ");
  fwrite(device_id.data, 1, device_id.size, stdout);
  printf("\n");

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t *session = NULL;

  IREE_CHECK_OK(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session));
  iree_hal_device_release(device);

  IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_memory(
      session, bytecode, iree_allocator_null()));

  iree_runtime_call_t call;
  IREE_CHECK_OK(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.predict"), &call));

  iree_vm_function_signature_t signature =
      iree_vm_function_signature(&call.function);
  printf("callconv = '%s'\n", signature.calling_convention.data);

  iree_host_size_t argument_count, result_count;
  IREE_CHECK_OK(iree_vm_function_call_count_arguments_and_results(
      &signature, &argument_count, &result_count));
  printf("args(%ld) results(%ld)\n", argument_count, result_count);

  int a = 4;
  int b = 4;

  iree_vm_list_resize(call.inputs, 2);

  iree_vm_value_t val = iree_vm_value_make_i32(a);
  iree_vm_list_set_value(call.inputs, 0, &val);

  val = iree_vm_value_make_i32(b);
  iree_vm_list_set_value(call.inputs, 1, &val);

  /**
  iree_hal_buffer_view_t *arg0 = NULL, *arg1 = NULL;
  static const iree_hal_dim_t unary_shape[1] = {4};

  iree_hal_buffer_view_allocate_buffer(
      iree_runtime_session_device_allocator(session), 1, unary_shape,
      IREE_HAL_ELEMENT_TYPE_INT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .access = IREE_HAL_MEMORY_ACCESS_READ,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(&a, sizeof(a)), &arg0);

  iree_hal_buffer_view_allocate_buffer(
      iree_runtime_session_device_allocator(session), 1, unary_shape,
      IREE_HAL_ELEMENT_TYPE_INT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .access = IREE_HAL_MEMORY_ACCESS_READ,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(&b, sizeof(b)), &arg1);

  IREE_CHECK_OK(iree_hal_buffer_view_fprint(
      stdout, arg0, 1, iree_runtime_session_host_allocator(session)));
  IREE_CHECK_OK(iree_hal_buffer_view_fprint(
      stdout, arg1, 1, iree_runtime_session_host_allocator(session)));

  iree_runtime_call_inputs_push_back_buffer_view(&call, arg0);
  iree_hal_buffer_view_release(arg0);

  iree_runtime_call_inputs_push_back_buffer_view(&call, arg1);
  iree_hal_buffer_view_release(arg1);
  */

  IREE_CHECK_OK(iree_runtime_call_invoke(&call, 0));

  size_t retsize = iree_vm_list_size(call.outputs);
  if (retsize != 1) FAIL("invalid output size");

  iree_vm_value_t ret0;
  iree_vm_list_get_value_as(call.outputs, 0, IREE_VM_VALUE_TYPE_I32, &ret0);
  printf("@predict(%d, %d) = %d\n", a, b, ret0.i32);

  // iree_hal_buffer_view_t *ret0 = NULL;
  // IREE_CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call,
  // &ret0)); IREE_CHECK_OK(iree_hal_buffer_view_fprint(
  //     stdout, ret0, 1, iree_runtime_session_host_allocator(session)));

  // iree_hal_buffer_view_release(ret0);
  // iree_runtime_call_deinitialize(&call);

  iree_runtime_session_release(session);
}

void create_mhlo_function(MlirContext ctx, MlirModule mod);

iree_const_byte_span_t compile_module(MlirContext ctx, MlirModule mod) {
  IreeCompilerOptions options = ireeCompilerOptionsCreate();
  const char *compiler_flags[] = {"--iree-hal-target-backends=llvm-cpu",
                                  "--iree-input-type=mhlo"};
  MlirLogicalResult status =
      ireeCompilerOptionsSetFlags(options, 2, compiler_flags, NULL, NULL);

  if (mlirLogicalResultIsFailure(status)) {
    ireeCompilerOptionsDestroy(options);
    mlirModuleDestroy(mod);
    mlirContextDestroy(ctx);
    FAIL("failed to create compiler options");
  }

  MlirPassManager pass = mlirPassManagerCreate(ctx);
  MlirOpPassManager op_pass = mlirPassManagerGetAsOpPassManager(pass);

  ireeCompilerBuildIREEVMPassPipeline(options, op_pass);

  status = mlirPassManagerRun(pass, mod);
  if (mlirLogicalResultIsFailure(status)) {
    ireeCompilerOptionsDestroy(options);
    mlirModuleDestroy(mod);
    mlirContextDestroy(ctx);
    FAIL("failed to run pass manager");
  }

  iree_string_builder_t out_builder;
  iree_allocator_t allocator = iree_allocator_system();
  iree_string_builder_initialize(allocator, &out_builder);

  status = ireeCompilerTranslateModuletoVMBytecode(
      options, mlirModuleGetOperation(mod), bytecode_builder_callback,
      &out_builder);

  if (mlirLogicalResultIsFailure(status)) {
    mlirPassManagerDestroy(pass);
    ireeCompilerOptionsDestroy(options);
    mlirModuleDestroy(mod);
    mlirContextDestroy(ctx);
    FAIL("failed to run pass manager");
  }

  const char *content = iree_string_builder_buffer(&out_builder);
  iree_host_size_t length = iree_string_builder_size(&out_builder);

  iree_const_byte_span_t bytecode = iree_make_const_byte_span(content, length);
  return bytecode;
}

int main(int argc, const char **argv) {
  ireeCompilerRegisterTargetBackends();
  ireeCompilerRegisterAllPasses();

  MlirContext ctx = mlirContextCreate();
  if (mlirContextIsNull(ctx)) FAIL("context is null");

  {
    int ndialects = mlirContextGetNumLoadedDialects(ctx);
    printf("loaded %d dialects\n", ndialects);
  }

  ireeCompilerRegisterAllDialects(ctx);

  // MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirContextLoadAllAvailableDialects(ctx);

  MlirDialect func = mlirContextGetOrLoadDialect(ctx, MLIR_STR("func"));
  if (mlirDialectIsNull(func)) FAIL("failed to load func");

  // MlirDialect linalg = mlirContextGetOrLoadDialect(
  //     ctx, MLIR_STR("linalg"));
  // if (mlirDialectIsNull(linalg)) FAIL("failed to load linalg");

  // MlirDialect arith =
  //     mlirContextGetOrLoadDialect(ctx,
  //     MLIR_STR("arith"));
  // if (mlirDialectIsNull(arith)) FAIL("failed to load arith");

  MlirDialect mhlo = mlirContextGetOrLoadDialect(ctx, MLIR_STR("mhlo"));
  if (mlirDialectIsNull(mhlo)) FAIL("failed to load mhlo");

  int ndialects = mlirContextGetNumLoadedDialects(ctx);
  printf("loaded %d dialects\n", ndialects);

#define NOPS 5
  char *ops[NOPS] = {
      "func.func", "linalg.generic", "builtin.module", "mhlo.dot", "arith.addi",
  };

  for (int i = 0; i < NOPS; ++i) {
    char *str = ops[i];
    bool registered = mlirContextIsRegisteredOperation(ctx, MLIR_STR(str));

    printf("op: %s ", str);
    if (registered) {
      printf("registered\n");
    } else {
      printf("not registered\n");
    }
  }

  MlirModule mod = mlirModuleCreateEmpty(mlirLocationUnknownGet(ctx));
  if (mlirModuleIsNull(mod)) FAIL("failed to create module from op");

  MlirBlock block = mlirModuleGetBody(mod);
  if (mlirBlockIsNull(block)) FAIL("failed to get module block");

  printf("MlirOperationState = %ld bytes\n", sizeof(MlirOperationState));

  MlirOperationState state =
      mlirOperationStateGet(MLIR_STR("func.func"), mlirLocationUnknownGet(ctx));
  MlirRegion region = mlirRegionCreate();

  MlirType i32 = mlirIntegerTypeGet(ctx, 32);
  MlirType argtypes[] = {i32, i32};

  MlirLocation locs[] = {mlirLocationUnknownGet(ctx),
                         mlirLocationUnknownGet(ctx)};
  MlirBlock funcblock = mlirBlockCreate(
      2, argtypes, locs);  // NOTE: locs has to be the same lang argtypes
  mlirRegionAppendOwnedBlock(region, funcblock);
  mlirOperationStateAddOwnedRegions(&state, 1, &region);

  MlirType functype = mlirFunctionTypeGet(ctx, 2, argtypes, 1, &i32);
  MlirAttribute functypeattr = mlirTypeAttrGet(functype);

  MlirNamedAttribute attributes[] = {
      mlirNamedAttributeGet(mlirIdentifierGet(ctx, MLIR_STR("sym_name")),
                            mlirStringAttrGet(ctx, MLIR_STR("predict"))),
      mlirNamedAttributeGet(mlirIdentifierGet(ctx, MLIR_STR("function_type")),
                            functypeattr)};
  mlirOperationStateAddAttributes(&state, 2, attributes);

  MlirValue addoperands[] = {mlirBlockGetArgument(funcblock, 0),
                             mlirBlockGetArgument(funcblock, 1)};

  MlirOperationState addstate = mlirOperationStateGet(
      MLIR_STR("arith.muli"), mlirLocationUnknownGet(ctx));
  mlirOperationStateAddResults(&addstate, 1, &i32);
  mlirOperationStateAddOperands(&addstate, 2, addoperands);
  MlirOperation addop = mlirOperationCreate(&addstate);
  mlirBlockAppendOwnedOperation(funcblock, addop);

  /*
  MlirOperationState conststate = mlirOperationStateGet(
      MLIR_STR("arith.constant"), mlirLocationUnknownGet(ctx));
  mlirOperationStateAddResults(&conststate, 1, &i32);
  *attributes =
      mlirNamedAttributeGet(mlirIdentifierGet(ctx, MLIR_STR("value")),
                            mlirAttributeParseGet(ctx, MLIR_STR("42 : i32")));
  mlirOperationStateAddAttributes(&conststate, 1, attributes);
  */

  MlirValue operands[1] = {
      mlirOperationGetResult(addop, 0),
  };

  MlirOperationState opstate = mlirOperationStateGet(
      MLIR_STR("func.return"), mlirLocationUnknownGet(ctx));
  mlirOperationStateAddOperands(&opstate, 1, operands);
  MlirOperation retop = mlirOperationCreate(&opstate);

  mlirBlockAppendOwnedOperation(funcblock, retop);

  MlirOperation op = mlirOperationCreate(&state);
  mlirBlockAppendOwnedOperation(block, op);

  if (argc >= 2) {
    FILE *f = fopen(argv[1], "w");
    MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
    mlirOpPrintingFlagsPrintGenericOpForm(flags);
    mlirOperationPrintWithFlags(mlirModuleGetOperation(mod), flags, callback,
                                f);
    fclose(f);
  }

  create_mhlo_function(ctx, mod);

  mlirOperationDump(mlirModuleGetOperation(mod));

  iree_const_byte_span_t bytecode = compile_module(ctx, mod);
  run_bytecode(bytecode);
  run_sum_reduce_1d(bytecode);

  mlirModuleDestroy(mod);
  mlirContextDestroy(ctx);
}

void create_mhlo_function(MlirContext ctx, MlirModule mod) {
  MlirOperationState funcstate =
      mlirOperationStateGet(MLIR_STR("func.func"), mlirLocationUnknownGet(ctx));
  MlirRegion region = mlirRegionCreate();

  const int64_t ten[] = {10};
  MlirType argtype = mlirRankedTensorTypeGetChecked(mlirLocationUnknownGet(ctx),
                                                    1, ten, mlirF32TypeGet(ctx),
                                                    mlirAttributeGetNull());
  MlirType rettype = mlirRankedTensorTypeGetChecked(
      mlirLocationUnknownGet(ctx), 0, NULL, mlirF32TypeGet(ctx),
      mlirAttributeGetNull());

  MlirLocation loc = mlirLocationUnknownGet(ctx);
  MlirBlock body = mlirBlockCreate(1, &argtype, &loc);
  mlirRegionAppendOwnedBlock(region, body);
  mlirOperationStateAddOwnedRegions(&funcstate, 1, &region);

  MlirType functype = mlirFunctionTypeGet(ctx, 1, &argtype, 1, &rettype);

  MlirNamedAttribute attributes[] = {
      mlirNamedAttributeGet(mlirIdentifierGet(ctx, MLIR_STR("sym_name")),
                            mlirStringAttrGet(ctx, MLIR_STR("sum_reduce_1d"))),
      mlirNamedAttributeGet(mlirIdentifierGet(ctx, MLIR_STR("function_type")),
                            mlirTypeAttrGet(functype))};
  mlirOperationStateAddAttributes(&funcstate, 2, attributes);

  MlirAttribute encoding = mlirAttributeGetNull();
  MlirType f32_unary = mlirRankedTensorTypeGetChecked(
      mlirLocationUnknownGet(ctx), 0, NULL, mlirF32TypeGet(ctx), encoding);

  MlirOperationState cst_state = mlirOperationStateGet(
      MLIR_STR("mhlo.constant"), mlirLocationUnknownGet(ctx));
  mlirOperationStateAddResults(&cst_state, 1, &f32_unary);

  const float zero = 0.;
  MlirNamedAttribute cst_attribute =
      mlirNamedAttributeGet(mlirIdentifierGet(ctx, MLIR_STR("value")),
                            mlirDenseElementsAttrFloatGet(f32_unary, 1, &zero));
  mlirOperationStateAddAttributes(&cst_state, 1, &cst_attribute);

  MlirOperation cst = mlirOperationCreate(&cst_state);
  mlirBlockAppendOwnedOperation(body, cst);
  MlirValue cst_val = mlirOperationGetResult(cst, 0);

  MlirOperationState sum_state = mlirOperationStateGet(
      MLIR_STR("mhlo.reduce"), mlirLocationUnknownGet(ctx));
  mlirOperationStateAddResults(&sum_state, 1, &rettype);

  int64_t dim[] = {0};
  int64_t rank[] = {1};
  MlirType ranked =
      mlirRankedTensorTypeGetChecked(mlirLocationUnknownGet(ctx), 1, rank,
                                     mlirIntegerTypeGet(ctx, 64), encoding);
  MlirNamedAttribute dimensions =
      mlirNamedAttributeGet(mlirIdentifierGet(ctx, MLIR_STR("dimensions")),
                            mlirDenseElementsAttrInt64Get(ranked, 1, dim));
  mlirOperationStateAddAttributes(&sum_state, 1, &dimensions);

  MlirValue sumargs[] = {mlirBlockGetArgument(body, 0), cst_val};
  mlirOperationStateAddOperands(&sum_state, 2, sumargs);

  MlirRegion reduceregion = mlirRegionCreate();
  MlirType reduceargs[] = {f32_unary, f32_unary};
  MlirLocation locs[] = {mlirLocationUnknownGet(ctx),
                         mlirLocationUnknownGet(ctx)};
  MlirBlock reduceregion_block = mlirBlockCreate(2, reduceargs, locs);
  mlirOperationStateAddOwnedRegions(&sum_state, 1, &reduceregion);
  mlirRegionAppendOwnedBlock(reduceregion, reduceregion_block);

  MlirOperationState inner_sum_state =
      mlirOperationStateGet(MLIR_STR("mhlo.add"), mlirLocationUnknownGet(ctx));
  MlirValue inner_sum_add_args[] = {
      mlirBlockGetArgument(reduceregion_block, 0),
      mlirBlockGetArgument(reduceregion_block, 1)};
  mlirOperationStateAddOperands(&inner_sum_state, 2, inner_sum_add_args);
  mlirOperationStateAddResults(&inner_sum_state, 1, &f32_unary);

  MlirOperation inner_sum = mlirOperationCreate(&inner_sum_state);
  mlirBlockAppendOwnedOperation(reduceregion_block, inner_sum);

  MlirOperationState inner_return_state = mlirOperationStateGet(
      MLIR_STR("mhlo.return"), mlirLocationUnknownGet(ctx));
  MlirValue inner_sum_result = mlirOperationGetResult(inner_sum, 0);
  mlirOperationStateAddOperands(&inner_return_state, 1, &inner_sum_result);

  MlirOperation inner_return = mlirOperationCreate(&inner_return_state);
  mlirBlockAppendOwnedOperation(reduceregion_block, inner_return);

  MlirOperation sum = mlirOperationCreate(&sum_state);
  mlirBlockAppendOwnedOperation(body, sum);

  MlirOperationState retstate = mlirOperationStateGet(
      MLIR_STR("func.return"), mlirLocationUnknownGet(ctx));
  MlirValue sum0 = mlirOperationGetResult(sum, 0);
  mlirOperationStateAddOperands(&retstate, 1, &sum0);

  MlirOperation retop = mlirOperationCreate(&retstate);
  mlirBlockAppendOwnedOperation(body, retop);

  MlirOperation func = mlirOperationCreate(&funcstate);

  MlirBlock modblock = mlirModuleGetBody(mod);
  mlirBlockAppendOwnedOperation(modblock, func);
}
