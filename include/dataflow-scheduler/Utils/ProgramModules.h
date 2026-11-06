//===-- ProgramModules.h ----------------------------------------*- c++ -*-===//
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
// This file declares what the scheduler decides about the child modules of the
// top-level module: which one is the declaration module, which program modules
// a run has already scheduled, and which are new.
//
// After compute-group-extraction the top-level module holds one unnamed child
// module, the declaration module, which calls the programs, and a named child
// module per program. A run on a module that already holds scheduled programs
// schedules only the new ones. How a pass tells the two apart is the re-entry
// mode, and every pass that has to tell asks classifyModule, so that the passes
// cannot come to disagree.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_UTILS_PROGRAMMODULES_H_
#define DATAFLOW_SCHEDULER_UTILS_PROGRAMMODULES_H_

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

#include <optional>

namespace scheduler {

/// How a run tells a program module it has already scheduled from a new one.
enum class ReentryMode {
  /// By the marker the last pass of every run leaves on each program module
  /// it scheduled.
  Marked,
};

/// The name of the re-entry mode option and of each of its values, as the
/// passes that take the option spell them.
llvm::StringRef stringifyReentryMode(ReentryMode mode);
std::optional<ReentryMode> symbolizeReentryMode(llvm::StringRef name);

/// The unit attribute that says a program module went through the whole
/// pipeline. Dialect-prefixed because builtin.module rejects a discardable
/// attribute without one.
inline constexpr llvm::StringLiteral kScheduledAttrName =
    "dataflow_scheduler.scheduled";

/// Whether \p module carries the marker.
bool isScheduled(mlir::ModuleOp module);

/// Marks \p module as scheduled. Also how a caller marks a program module it
/// added without the scheduler, so that a run in marked mode leaves it alone.
void markScheduled(mlir::ModuleOp module);

/// Whether \p child is the declaration module: the one child of the top-level
/// module that carries no name.
bool isOrchestratorModule(mlir::ModuleOp child);

/// The child module of the top-level module that \p op is in, or null when it
/// is in none. A scheduled program nests a module of its own inside its
/// program module, so the nearest enclosing module is not always the one.
mlir::ModuleOp getTopLevelChildModule(mlir::Operation* op);

/// Whether \p module holds an op that only KTIR still to be scheduled holds:
/// ktdp.load, ktdp.store, ktdp.construct_access_tile or
/// ktdp.construct_indirect_access_tile. ktir-bufferize removes them, so this
/// only means something before it.
bool hasTransferOps(mlir::ModuleOp module);

/// What a child module of the top-level module is to a run.
enum class ModuleKind {
  /// The declaration module.
  Declaration,
  /// A program module the run schedules.
  New,
  /// A program module an earlier run scheduled, or a caller marked.
  Scheduled,
};

/// Classifies \p child, a child module of the top-level module, under \p mode.
///
/// In marked mode, a marked program module is Scheduled and an unmarked one is
/// New. The two are checked against the module's ops: a marked module holding
/// a transfer op, or an unmarked one holding none, is reported on \p child and
/// is a failure. The check is only valid before ktir-bufferize, so only
/// ktir-legality-check makes it; a later pass reads isScheduled instead.
mlir::FailureOr<ModuleKind> classifyModule(mlir::ModuleOp child,
                                           ReentryMode mode);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_UTILS_PROGRAMMODULES_H_
