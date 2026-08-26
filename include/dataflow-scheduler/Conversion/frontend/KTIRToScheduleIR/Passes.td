//===-- Passes.td ------------------------------------------*- tablegen -*-===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares all the scheduler passes.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTIRTOSCHEDULEIR_PASSES_TD
#define DATAFLOW_SCHEDULER_CONVERSION_KTIRTOSCHEDULEIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def ComputeGroupExtractionPass : Pass<"compute-group-extraction", "mlir::ModuleOp"> {
  let summary = "Extract main and compute group modules from a module";
  let description = [{
    This pass extracts the main and compute groups from the initial module.
  }];
  let constructor = "::scheduler::createComputeGroupExtractionPass()";
}

def ConstructThreeStagePipelinePass : Pass<"construct-three-stage-pipeline", "mlir::ModuleOp"> {
  let summary = "Construct a three-stage pipeline";
  let description = [{
    This pass constructs a three-stage pipeline.
  }];
  let constructor = "::scheduler::createConstructThreeStagePipelinePass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::ktdf::KTDFDialect",
    "mlir::scf::SCFDialect"
  ];
}

def KTIRLegalityCheckPass : Pass<"ktir-legality-check", "mlir::ModuleOp"> {
  let summary = "Reject KTIR constructs V1 cannot lower yet";
  let description = [{
    Fails fast with a diagnostic when the input contains V1-unsupported
    constructs: scf.for with loop-carried args, scf.while, compute ops outside
    {add, mul, sub}, or the ktdp distributed-memory-view / indirect-access-tile
    ops. Mutates nothing; legal input passes through.
  }];
  let constructor = "::scheduler::createKTIRLegalityCheckPass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

def IndirectComputeGroupSplitPass
    : Pass<"indirect-compute-group-split", "mlir::ModuleOp"> {
  let summary = "Split extracted compute groups containing indirect access tiles";
  let description = [{
    For each extracted child module whose function body contains a
    `ktdp.construct_indirect_access_tile` op, this pass splits that module
    into two sibling child modules:

    1. An **index-to-address** child module (`local_schedule_idx_to_addr`)
       that loads the index tensor from the indirect memref, multiplies each
       entry by the appropriate stride of the base memref to resolve a flat
       byte-offset address, and stores the resulting address tensor into a
       freshly allocated `memref<...xindex, global>` addr_buf, which is
       returned to the orchestrator.

    2. A **gather/scatter** child module (the renamed original) that receives
       the addr_buf as a function argument, copies it into the IAB (indirect
       address buffer) memory view, and replaces
       `ktdp.construct_indirect_access_tile` with
       `ktdp_lowering.construct_indirect_access_tile` whose indirection is
       fully resolved through the IAB.  All subscripts into the base memref
       are direct in the lowered op.

    The orchestrator (top-level inner-module function) is rewritten so that
    it first calls the index-to-address function to obtain the addr_buf, then
    passes it to the gather/scatter function.

    The pass is a no-op when no `ktdp.construct_indirect_access_tile` is
    present in the module.
  }];
  let constructor = "::scheduler::createIndirectComputeGroupSplitPass()";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
    "mlir::ktdp::KtdpDialect",
    "mlir::ktdp_lowering::KTDPLoweringDialect"
  ];
}

#endif // DATAFLOW_SCHEDULER_CONVERSION_KTIRTOSCHEDULEIR_PASSES_TD
