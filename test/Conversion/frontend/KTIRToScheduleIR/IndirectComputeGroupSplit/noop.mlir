// RUN: dataflow-scheduler-opt --indirect-compute-group-split %s | FileCheck %s

// No-op case: the input has no ktdp.construct_indirect_access_tile.
// The pass should be a no-op and emit the IR unchanged.

// CHECK:      module {
// CHECK-NEXT:   module {
// CHECK:          func.func @add()
// CHECK:          func.func private @local_schedule_0()
// CHECK:        }
// CHECK:        module @local_schedule_0 {
// CHECK:          func.func @local_schedule_0()
// CHECK:          ktdp.load
// CHECK:          linalg.add
// CHECK:          ktdp.store
// CHECK-NOT:    ktdp_lowering.construct_indirect_access_tile
// CHECK-NOT:    local_schedule_idx_to_addr

#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  module {
    func.func @add() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0  = arith.constant 0    : index
      %c1  = arith.constant 1024 : index
      %c2  = arith.constant 12288 : index
      %c3  = arith.constant 18432 : index

      %A = ktdp.construct_memory_view %c1, sizes: [96, 64], strides: [64, 1]
          {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>}
          : memref<96x64xf16>
      %B = ktdp.construct_memory_view %c2, sizes: [96, 64], strides: [64, 1]
          {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>}
          : memref<96x64xf16>
      %C = ktdp.construct_memory_view %c3, sizes: [96, 64], strides: [64, 1]
          {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>}
          : memref<96x64xf16>

      %at_A = ktdp.construct_access_tile %A[%c0, %c0]
          {access_tile_set = #set, access_tile_order = #map}
          : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>
      %at_B = ktdp.construct_access_tile %B[%c0, %c0]
          {access_tile_set = #set, access_tile_order = #map}
          : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>
      %at_C = ktdp.construct_access_tile %C[%c0, %c0]
          {access_tile_set = #set, access_tile_order = #map}
          : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>

      %a = ktdp.load %at_A : <3x64xindex> -> tensor<3x64xf16>
      %b = ktdp.load %at_B : <3x64xindex> -> tensor<3x64xf16>
      %e = tensor.empty() : tensor<3x64xf16>
      %c = linalg.add ins(%a, %b : tensor<3x64xf16>, tensor<3x64xf16>)
                      outs(%e : tensor<3x64xf16>) -> tensor<3x64xf16>
      ktdp.store %c, %at_C : tensor<3x64xf16>, <3x64xindex>
      return
    }
  }
}
