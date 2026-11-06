// RUN: dataflow-scheduler-opt -ktir-to-dfir "%s" | FileCheck "%s"

// A re-run's input: the kernel already in extracted form, with a declaration
// module calling a program in a loop. The value the call passes varies per
// iteration, and the declaration module declares it, as a caller has to.
//
// compute-group-extraction reuses the declaration module rather than extracting
// it again. The program's argument takes the declared id, -100, and the
// addresses the program computes from it become derived symbols over -100.
// The kernel's own argument is declared below every id in use.

// CHECK-LABEL: func.func @arith_math_test(
// CHECK-SAME:      %[[BASE:.*]]: index)
// CHECK:         symbol.create_id %[[BASE]] {symbol_id = -101 : si64}
// CHECK:         scf.for
// CHECK:           %[[A:.*]] = arith.addi %[[BASE]]
// CHECK:           symbol.create_id %[[A]] {symbol_id = -100 : si64}
// CHECK:           call @local_schedule_0(%[[A]])

// CHECK-LABEL: module @local_schedule_0 attributes {dataflow_scheduler.scheduled}
// CHECK:         func.func @local_schedule_0(%[[ARG:.*]]: index)
// CHECK:           symbol.create_id %[[ARG]] {symbol_id = -100 : si64}
// CHECK:           %[[D0:.*]] = arith.divsi %[[ARG]]
// CHECK:           symbol.create_id %[[D0]] {symbol_id = -102 : si64}
// CHECK:           %[[OFF:.*]] = arith.addi %[[ARG]]
// CHECK:           %[[D1:.*]] = arith.divsi %[[OFF]]
// CHECK:           symbol.create_id %[[D1]] {symbol_id = -103 : si64}
// CHECK:         func.func @local_schedule_0_body()
// CHECK-DAG:       symbol.create_symbol {SymbolId = -102 : i64}
// CHECK-DAG:       symbol.create_symbol {SymbolId = -103 : i64}

#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  module {
    func.func @arith_math_test(%base: index) attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %off = arith.muli %arg0, %c4096 : index
        %a = arith.addi %base, %off : index
        symbol.create_id %a {symbol_id = -100 : si64} : index
        func.call @local_schedule_0(%a) : (index) -> ()
      }
      return
    }
    func.func private @local_schedule_0(index)
  }
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0(%arg0: index) attributes {grid = [2]} {
      %c0_r = arith.constant 0 : index
      %0 = ktdp.construct_memory_view %arg0, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
      %1 = ktdp.get_compute_tile_id : index
      %c3 = arith.constant 3 : index
      %2 = arith.muli %1, %c3 : index
      %3 = arith.addi %2, %c0_r : index
      %c0 = arith.constant 0 : index
      %4 = ktdp.construct_access_tile %0[%3, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
      %c12288 = arith.constant 12288 : index
      %5 = ktdp.construct_memory_view %c12288, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
      %6 = ktdp.construct_access_tile %5[%3, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
      %7 = ktdp.load %4 : <1x64xindex> -> tensor<1x64xf16>
      %8 = ktdp.load %6 : <1x64xindex> -> tensor<1x64xf16>
      %9 = tensor.empty() : tensor<1x64xf16>
      %c18432 = arith.constant 18432 : index
      %10 = ktdp.construct_memory_view %c18432, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
      %11 = linalg.add ins(%7, %8 : tensor<1x64xf16>, tensor<1x64xf16>) outs(%9 : tensor<1x64xf16>) -> tensor<1x64xf16>
      %12 = ktdp.construct_access_tile %10[%3, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
      ktdp.store %11, %12 : tensor<1x64xf16>, <1x64xindex>
      return
    }
  }
}

