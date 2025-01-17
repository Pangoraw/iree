// RUN: iree-opt --split-input-file -iree-flow-collapse-dims %s | FileCheck %s

func.func @multi_reduce_dim(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  %cst = arith.constant -0.000000e+00 : f32
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2x32x10x4096xf32>
  %1 = tensor.empty() : tensor<2x32xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2x32xf32>) -> tensor<2x32xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%0 : tensor<2x32x10x4096xf32>) outs(%2 : tensor<2x32xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %6 = arith.addf %arg1, %arg2 : f32
    linalg.yield %6 : f32
  } -> tensor<2x32xf32>
  %4 = tensor.expand_shape %3 [[0], [1, 2, 3]] : tensor<2x32xf32> into tensor<2x32x1x1xf32>
  %5 = hal.tensor.export %4 : tensor<2x32x1x1xf32> -> !hal.buffer_view
  return %5 : !hal.buffer_view
}

// Check that we collapse dimensions.
// CHECK: @multi_reduce_dim
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "reduction"]
