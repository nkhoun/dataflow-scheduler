//===-- ProgramModules.cpp --------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Utils/ProgramModules.h"

#include <llvm/ADT/StringSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>

#include "ktir/Dialect/KTDP/KTDP.h"

namespace scheduler {

llvm::StringRef stringifyReentryMode(ReentryMode mode) {
  switch (mode) {
    case ReentryMode::Marked:
      return "marked";
  }
  llvm_unreachable("unknown re-entry mode");
}

std::optional<ReentryMode> symbolizeReentryMode(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<ReentryMode>>(name)
      .Case("marked", ReentryMode::Marked)
      .Default(std::nullopt);
}

bool isScheduled(mlir::ModuleOp module) {
  return module->hasAttr(kScheduledAttrName);
}

void markScheduled(mlir::ModuleOp module) {
  module->setAttr(kScheduledAttrName, mlir::UnitAttr::get(module.getContext()));
}

bool isOrchestratorModule(mlir::ModuleOp child) {
  return !child.getSymName().has_value();
}

mlir::ModuleOp getTopLevelChildModule(mlir::Operation* op) {
  for (auto module = op->getParentOfType<mlir::ModuleOp>(); module;
       module = module->getParentOfType<mlir::ModuleOp>()) {
    auto parent = module->getParentOfType<mlir::ModuleOp>();
    if (parent && !parent->getParentOp()) return module;
  }
  return nullptr;
}

bool hasTransferOps(mlir::ModuleOp module) {
  const mlir::WalkResult walk = module.walk([](mlir::Operation* op) {
    if (mlir::isa<mlir::ktdp::LoadOp, mlir::ktdp::StoreOp,
                  mlir::ktdp::ConstructAccessTilesOp,
                  mlir::ktdp::ConstructIndirectAccessTilesOp>(op))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });
  return walk.wasInterrupted();
}

mlir::FailureOr<ModuleKind> classifyModule(mlir::ModuleOp child,
                                           ReentryMode mode) {
  if (isOrchestratorModule(child)) return ModuleKind::Declaration;

  switch (mode) {
    case ReentryMode::Marked: {
      const bool marked = isScheduled(child);
      const bool transfers = hasTransferOps(child);
      if (marked && transfers) {
        return child.emitError()
               << "program module is marked '" << kScheduledAttrName
               << "' but still holds a ktdp transfer op; a module still to be "
                  "scheduled must not be marked";
      }
      if (!marked && !transfers) {
        return child.emitError()
               << "program module holds no ktdp transfer op but is not marked '"
               << kScheduledAttrName
               << "'; a module the scheduler is not to schedule must be marked";
      }
      return marked ? ModuleKind::Scheduled : ModuleKind::New;
    }
  }
  llvm_unreachable("unknown re-entry mode");
}

}  // namespace scheduler
