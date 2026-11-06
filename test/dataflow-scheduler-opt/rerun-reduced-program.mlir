// RUN: dataflow-scheduler-opt -ktir-to-dfir %s | FileCheck %s

// A re-run over a program a caller has reduced: what is left of @reduced after
// an earlier run is its entry function, saying which symbols it stands behind,
// and none of the DataflowIR it was compiled from. It keeps the marker the
// earlier run left.
//
// The pipeline leaves @reduced as it is. Its ids, down to -40, are seen by the
// next-free-id scan, so the new kernel's symbols are numbered below them; the
// kernel's own input keeps -1.

// CHECK-LABEL: module @reduced attributes {dataflow_scheduler.scheduled} {
// CHECK-NEXT:    func.func @reduced(%[[R:.*]]: index) {
// CHECK-NEXT:      %{{.*}} = arith.constant 64 : index
// CHECK-NEXT:      symbol.create_id %[[R]] {symbol_id = -1 : si64}
// CHECK-NEXT:      %[[O:.*]] = arith.addi %[[R]]
// CHECK-NEXT:      symbol.create_id %[[O]] {symbol_id = -40 : si64}
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-LABEL: module @new attributes {dataflow_scheduler.scheduled}
// CHECK:         func.func @new(%[[N:.*]]: index)
// CHECK:           symbol.create_id %[[N]] {symbol_id = -1 : si64}
// CHECK:           symbol.create_id %{{.*}} {symbol_id = -41 : si64}
// CHECK-NOT:     symbol_id = -{{([2-9]|[1-3][0-9]|40)}} :

module {
  module {
    func.func @kernel(%arg0: index) attributes {grid = [2]} {
      symbol.create_id %arg0 {symbol_id = -1 : si64} : index
      call @reduced(%arg0) : (index) -> ()
      call @new(%arg0) : (index) -> ()
      return
    }
    func.func private @reduced(index)
    func.func private @new(index)
  }
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  module @reduced attributes {dataflow_scheduler.scheduled} {
    func.func @reduced(%arg0: index) {
      %c64 = arith.constant 64 : index
      symbol.create_id %arg0 {symbol_id = -1 : si64} : index
      %0 = arith.addi %arg0, %c64 : index
      symbol.create_id %0 {symbol_id = -40 : si64} : index
      return
    }
  }
  module @new {
    func.func @new(%arg0: index) attributes {grid = [2]} {
      %c1024 = arith.constant 1024 : index
      %0 = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
      %1 = ktdp.get_compute_tile_id : index
      %c3 = arith.constant 3 : index
      %2 = arith.muli %1, %c3 : index
      %3 = arith.addi %2, %arg0 : index
      %c0 = arith.constant 0 : index
      %4 = ktdp.construct_access_tile %0[%3, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>, access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
      %c12288 = arith.constant 12288 : index
      %5 = ktdp.construct_memory_view %c12288, sizes: [96, 64], strides: [64, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
      %6 = ktdp.construct_access_tile %5[%3, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>, access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
      %7 = ktdp.load %4 : <1x64xindex> -> tensor<1x64xf16>
      %8 = ktdp.load %6 : <1x64xindex> -> tensor<1x64xf16>
      %9 = tensor.empty() : tensor<1x64xf16>
      %c18432 = arith.constant 18432 : index
      %10 = ktdp.construct_memory_view %c18432, sizes: [96, 64], strides: [64, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
      %11 = linalg.add ins(%7, %8 : tensor<1x64xf16>, tensor<1x64xf16>) outs(%9 : tensor<1x64xf16>) -> tensor<1x64xf16>
      %12 = ktdp.construct_access_tile %10[%3, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>, access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
      ktdp.store %11, %12 : tensor<1x64xf16>, <1x64xindex>
      return
    }
  }
}
