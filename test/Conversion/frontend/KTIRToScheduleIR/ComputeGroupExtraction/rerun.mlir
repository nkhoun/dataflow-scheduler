// RUN: dataflow-scheduler-opt --compute-group-extraction -split-input-file -verify-diagnostics %s | FileCheck %s

// A declaration module already present: the pipeline is being run again. The
// pass reuses it and extracts nothing. A scheduled program, in the shape
// wrap-program-dfir leaves (holding a function nothing calls), and an
// _idx_to_addr module are both called by the declaration module, so both are
// already extracted. So is the new kernel, whose load-to-store group the pass
// would otherwise extract a second time.

// CHECK-LABEL: module {
// CHECK:         module {
// CHECK:           func.func @kernel()
// CHECK:             call @prog_idx_to_addr()
// CHECK:             call @prog()
// CHECK:             call @new()
// CHECK-NOT:     module @local_schedule
// CHECK:         module @prog_idx_to_addr {
// CHECK:         module @prog attributes {dataflow_scheduler.scheduled} {
// CHECK:           func.func private @prog_body()
// CHECK:         module @new {
// CHECK:           ktdp.load
// CHECK:           ktdp.store
// CHECK-NOT:     module @local_schedule

module {
  module {
    func.func @kernel() {
      call @prog_idx_to_addr() : () -> ()
      call @prog() : () -> ()
      call @new() : () -> ()
      return
    }
    func.func private @prog_idx_to_addr()
    func.func private @prog()
    func.func private @new()
  }
  module @prog_idx_to_addr {
    func.func @prog_idx_to_addr() {
      return
    }
  }
  module @prog attributes {dataflow_scheduler.scheduled} {
    func.func private @prog_body()
    func.func @prog() {
      call @prog_body() : () -> ()
      return
    }
    module {
      func.func @prog_body() {
        return
      }
    }
  }
  module @new {
    func.func @new() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c2048 = arith.constant 2048 : index
      %a = ktdp.construct_memory_view %c1024, sizes: [1, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
      } : memref<1x64xf16>
      %b = ktdp.construct_memory_view %c2048, sizes: [1, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
      } : memref<1x64xf16>
      %ta = ktdp.construct_access_tile %a[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
      %tb = ktdp.construct_access_tile %b[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
      %v = ktdp.load %ta : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
      ktdp.store %v, %tb : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
      return
    }
  }
}

// -----

// A program module the declaration module does not call has no call site.

module {
  module {
    func.func @kernel() {
      return
    }
  }
  // expected-error @+1 {{program module @orphan is not called by the declaration module}}
  module @orphan {
    func.func @orphan() {
      return
    }
  }
}

// -----

// With a declaration module present, a function outside any module has no
// defined place for its call.

module {
  module {
    func.func @kernel() {
      return
    }
  }
  // expected-error @+1 {{function outside any module alongside a declaration module}}
  func.func @stray() {
    return
  }
}
