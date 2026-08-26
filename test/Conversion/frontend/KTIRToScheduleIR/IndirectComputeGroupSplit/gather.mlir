// RUN: dataflow-scheduler-opt --compute-group-extraction --indirect-compute-group-split %s | FileCheck %s

// Gather example: indirect load (ktdp.construct_indirect_access_tile with
// ind() on the first subscript dimension).  After IndirectComputeGroupSplit
// the original extracted module is split into two sibling modules:
//   1. local_schedule_idx_to_addr – resolves indices to flat addresses
//   2. local_schedule_* (renamed original) – consumes the addr_buf + computes
//
// The orchestrator is updated to call (1), then pass its result to (2) along
// with zero-initialised tile IV placeholders (Steps 2+3 will replace with loops).

// ── Orchestrator ──────────────────────────────────────────────────────────

// CHECK:       func.func @triton_unk_fused_index_0() {{.*}} {
// CHECK-NEXT:    %[[ADDR_BUF:.*]] = call @local_schedule_idx_to_addr_{{.*}}() :
// CHECK-SAME:        () -> memref<2x32xi32, #ktdp.memory_space<global>>
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:    call @local_schedule_{{[0-9_]+}}(%[[ADDR_BUF]], {{.*}}) :
// CHECK-SAME:        (memref<2x32xi32, #ktdp.memory_space<global>>, index, index, index, index) -> ()
// CHECK-NEXT:    return
// CHECK:       func.func private @local_schedule_{{[0-9_]+}}(memref<2x32xi32, #ktdp.memory_space<global>>, index, index, index, index)
// CHECK:       func.func private @local_schedule_idx_to_addr_{{.*}}() -> memref<2x32xi32, #ktdp.memory_space<global>>

// ── gather module (printed before idx_to_addr in the output) ──────────────

// CHECK: module @local_schedule_{{[0-9_]+}}
// CHECK:   func.func @local_schedule_{{[0-9_]+}}(
// CHECK-SAME:      %{{.*}}: memref<2x32xi32, #ktdp.memory_space<global>>,
// CHECK-SAME:      %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index)
// addr_buf → IAB: alloc + load + store
// CHECK:       memref.alloc() : memref<2x32xi32,
// CHECK-SAME:      "IAB"
// CHECK:       ktdp.load
// CHECK:       ktdp.store
// Lowered indirect access tile (boundary: consuming resolved addresses).
// CHECK:       ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:      base_ptr =
// Direct load, compute, output store.
// CHECK:       ktdp.load
// CHECK:       linalg.generic
// CHECK:       ktdp.store
// CHECK:       return

// ── idx_to_addr module ────────────────────────────────────────────────────

// CHECK: module @local_schedule_idx_to_addr_{{.*}}
// CHECK:   func.func @local_schedule_idx_to_addr_{{.*}}()
// CHECK-SAME:    -> memref<2x32xi32, #ktdp.memory_space<global>>
// CHECK:       memref.alloc() : memref<2x32xi32, #ktdp.memory_space<global>>
// CHECK:       ktdp.construct_access_tile
// CHECK-SAME:      memref<2x32xsi32>
// CHECK:       ktdp.load
// CHECK-SAME:      tensor<2x32xi32>
// CHECK:       memref.extract_strided_metadata
// CHECK:       memref.extract_aligned_pointer_as_index
// CHECK:       arith.index_castui
// CHECK:       arith.muli
// CHECK:       arith.addi
// CHECK:       ktdp.store
// CHECK:       return

// ─────────────────────────────────────────────────────────────────────────
// Input KTIR (pre-ComputeGroupExtraction form)
// ─────────────────────────────────────────────────────────────────────────

#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  func.func @triton_unk_fused_index_0() attributes {grid = [1]} {
    %c100   = arith.constant 100   : index
    %c1000  = arith.constant 1000  : index
    %c10000 = arith.constant 10000 : index
    %c0     = arith.constant 0     : index

    %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xsi32>

    %desc_1 = ktdp.construct_memory_view %c1000, sizes: [64, 2, 64], strides: [64, 4096, 1]
        {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>}
        : memref<64x2x64xf16>

    %desc_2 = ktdp.construct_memory_view %c10000, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
        {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>}
        : memref<2x32x2x64xf16>

    %tmp1 = ktdp.construct_indirect_access_tile
        intermediate_variables(%arg5, %arg6, %arg7, %arg8)
        %desc_1[ind(%desc_0[%c0 + %arg5, %c0 + %arg6]), (%c0 + %arg7), (%arg8)]
        {variables_space_order = #map, variables_space_set = #set2}
        : memref<64x2x64xf16>, memref<2x32xsi32> -> !ktdp.access_tile<2x32x2x64xindex>

    %tmp1_0 = ktdp.load %tmp1 : <2x32x2x64xindex> -> tensor<2x32x2x64xf16>

    %12 = tensor.empty() : tensor<2x32x2x64xf16>
    %13 = linalg.generic {
            indexing_maps = [#map, #map],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
          ins(%tmp1_0: tensor<2x32x2x64xf16>)
          outs(%12: tensor<2x32x2x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %cf10 = arith.constant 10.000000e+00 : f16
      %14 = arith.addf %in, %cf10 : f16
      linalg.yield %14 : f16
    } -> tensor<2x32x2x64xf16>

    %0 = ktdp.construct_access_tile %desc_2[%c0, %c0, %c0, %c0]
        {access_tile_order = #map, access_tile_set = #set2}
        : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
    ktdp.store %13, %0 : tensor<2x32x2x64xf16>, <2x32x2x64xindex>
    return
  }
}
