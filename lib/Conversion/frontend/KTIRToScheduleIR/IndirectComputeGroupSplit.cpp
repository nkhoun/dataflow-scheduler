//===----------------------------------------------------------------------===//
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
/// IndirectComputeGroupSplit — Step 1 of the gather/scatter lowering pipeline.
///
/// For each child module whose function body contains a
/// `ktdp.construct_indirect_access_tile` op, this pass splits it into two
/// sibling child modules:
///
///   1. **index-to-address** module (`local_schedule_idx_to_addr`):
///      Loads the index tensor from the indirect memref, multiplies each
///      element by the stride of the indirect dimension of the base memref to
///      produce a flat byte-offset address per entry, and stores the resulting
///      address tensor into a freshly allocated global addr_buf.  The addr_buf
///      is returned to the orchestrator.
///
///   2. **gather/scatter** module (the renamed original):
///      Receives the addr_buf as a function argument.  Allocates an IAB
///      (indirect address buffer) of the same shape, loads the addr_buf into
///      it, then replaces `ktdp.construct_indirect_access_tile` with
///      `ktdp_lowering.construct_indirect_access_tile` using the IAB as the
///      base-pointer source.  All subscripts into the base memref are direct
///      in the lowered op.
///
/// **Logical boundary**: everything needed to *resolve* the indirect subscript
/// (loading the index memref + stride-multiply address computation) belongs in
/// the idx_to_addr group.  Everything that *consumes* those addresses (filling
/// the IAB, the data transfer, compute, and output store) belongs in the
/// gather/scatter group.  The addr_buf memref is the sole interface between
/// the two groups.
///
/// The pass is a no-op when no `ktdp.construct_indirect_access_tile` is present.
///
//===----------------------------------------------------------------------===//

// clang-format off
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"
// clang-format on

#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#define PASS_NAME "indirect-compute-group-split"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_INDIRECTCOMPUTEGROUPSPLITPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Return the first `ktdp.construct_indirect_access_tile` found in @func.
static mlir::ktdp::ConstructIndirectAccessTilesOp
findIndirectOp(mlir::func::FuncOp func) {
  mlir::ktdp::ConstructIndirectAccessTilesOp found = nullptr;
  func.walk([&](mlir::ktdp::ConstructIndirectAccessTilesOp op) {
    found = op;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

/// Recursively clone @val and all its (transitive) defining ops into the
/// current @builder insertion point using @mapper.  Block arguments that are
/// already in @mapper are already resolved.  Any block argument not in
/// @mapper is recorded in @needed_block_args so the caller can create a
/// function parameter for it (duplicates are suppressed via @args_seen).
static void materializeValue(mlir::Value val, mlir::IRMapping& mapper,
                             mlir::OpBuilder& builder,
                             llvm::SmallVectorImpl<mlir::Value>& needed_block_args,
                             llvm::DenseSet<mlir::Value>& args_seen) {
  if (mapper.contains(val)) return;

  if (mlir::isa<mlir::BlockArgument>(val)) {
    if (args_seen.insert(val).second)
      needed_block_args.push_back(val);
    // Don't try to clone a block argument - it will be a func arg.
    return;
  }

  mlir::Operation* op = val.getDefiningOp();
  for (mlir::Value operand : op->getOperands())
    materializeValue(operand, mapper, builder, needed_block_args, args_seen);

  if (!mapper.contains(val))
    builder.clone(*op, mapper);
}

/// Return a `#ktdp.memory_space<global>` attribute.
static mlir::Attribute globalMemorySpaceAttr(mlir::MLIRContext* ctx) {
  return mlir::ktdp::MemorySpaceAttr::get(ctx,
                                          mlir::ktdp::MemorySpaceKind::global,
                                          /*ct_id=*/-1);
}

/// Return an identity AffineMap of the given rank.
static mlir::AffineMap identityMap(mlir::MLIRContext* ctx, unsigned rank) {
  return mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
}

/// Project the first @keep_dims dimensions out of @full_set.  Constraints
/// that reference higher dimensions are dropped (they belong to the direct
/// subscript space, not the IAB space).
static mlir::IntegerSet projectIntegerSet(mlir::IntegerSet full_set,
                                         unsigned keep_dims) {
  mlir::MLIRContext* ctx = full_set.getContext();
  if (full_set.getNumDims() == keep_dims) return full_set;

  llvm::SmallVector<mlir::AffineExpr> constraints;
  llvm::SmallVector<bool> eq_flags;
  for (unsigned ci = 0; ci < full_set.getNumConstraints(); ++ci) {
    mlir::AffineExpr c = full_set.getConstraint(ci);
    bool uses_high = false;
    c.walk([&](mlir::AffineExpr e) {
      if (auto d = mlir::dyn_cast<mlir::AffineDimExpr>(e))
        if (d.getPosition() >= keep_dims) uses_high = true;
    });
    if (!uses_high) {
      constraints.push_back(c);
      eq_flags.push_back(full_set.isEq(ci));
    }
  }
  if (constraints.empty())
    return mlir::IntegerSet::getEmptySet(keep_dims, 0, ctx);
  return mlir::IntegerSet::get(keep_dims, 0, constraints, eq_flags);
}

/// Evaluate a single-result affine expression over @operands.  If the
/// expression is a pure dimension reference, return the operand directly
/// (avoids emitting a trivial affine.apply).  Otherwise emit an
/// affine.apply.
static mlir::Value evalAffineExpr(mlir::AffineExpr expr,
                                  llvm::ArrayRef<mlir::Value> operands,
                                  mlir::OpBuilder& builder,
                                  mlir::Location loc) {
  if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
    if (dim.getPosition() < operands.size())
      return operands[dim.getPosition()];
  }
  mlir::AffineMap m = mlir::AffineMap::get(
      operands.size(), 0, {expr}, builder.getContext());
  return mlir::affine::AffineApplyOp::create(builder, loc, m, operands);
}

//===----------------------------------------------------------------------===//
// Pass declaration
//===----------------------------------------------------------------------===//

struct IndirectComputeGroupSplitPass
    : public impl::IndirectComputeGroupSplitPassBase<
          IndirectComputeGroupSplitPass> {
  void runOnOperation() final;

 private:
  void processTopLevelModule(mlir::ModuleOp top_module);

  void splitChildModule(mlir::ModuleOp top_module,
                        mlir::ModuleOp orch_module,
                        mlir::ModuleOp child_module);

  /// Build the index-to-address sibling module.  Fills @addr_buf_type and
  /// @idx_func_name for the caller.
  mlir::ModuleOp buildIdxToAddrModule(mlir::ModuleOp top_module,
                                      mlir::func::FuncOp original_func,
                                      mlir::ktdp::ConstructIndirectAccessTilesOp indirect_op,
                                      mlir::MemRefType& addr_buf_type);

  /// Rewrite the gather/scatter child module in place.
  void rewriteGatherScatterModule(
      mlir::ModuleOp child_module,
      mlir::StringRef new_func_name,
      mlir::MemRefType addr_buf_type,
      mlir::ktdp::ConstructIndirectAccessTilesOp indirect_op);
};

//===----------------------------------------------------------------------===//
// buildIdxToAddrModule
//===----------------------------------------------------------------------===//

/// Build the idx_to_addr child module.
///
/// Boundary: all work to *resolve* the indirect subscript lives here.
///  - Clone the index memref's dependencies (memory view + constants).
///  - Build a ktdp.construct_access_tile + ktdp.load over the index memref to
///    get the index tensor.
///  - Extract the base pointer and stride[ind_dim] from the base memref via
///    memref.extract_strided_metadata.
///  - Compute addr_tensor = base_ptr + idx_tensor * stride (element-wise i32).
///  - Allocate a global addr_buf, build a ktdp.construct_access_tile + ktdp.store
///    to write addr_tensor in, and return addr_buf to the orchestrator.
mlir::ModuleOp IndirectComputeGroupSplitPass::buildIdxToAddrModule(
    mlir::ModuleOp top_module,
    mlir::func::FuncOp original_func,
    mlir::ktdp::ConstructIndirectAccessTilesOp indirect_op,
    mlir::MemRefType& addr_buf_type) {
  mlir::MLIRContext* ctx = top_module.getContext();
  mlir::Location loc = top_module.getLoc();

  // ------------------------------------------------------------------
  // Gather information from the indirect op.
  // ------------------------------------------------------------------
  mlir::Value base = indirect_op.getBase();

  // The first indirect_memref is the index memref (e.g. memref<2x32xsi32>).
  mlir::Value idx_memref = indirect_op.getIndirectMemrefs()[0];
  auto idx_memref_type = mlir::cast<mlir::MemRefType>(idx_memref.getType());
  llvm::ArrayRef<int64_t> iab_shape = idx_memref_type.getShape();
  unsigned iab_rank = iab_shape.size();
  mlir::Type idx_elem_type = idx_memref_type.getElementType();

  // The IAB (and addr_buf) shape equals the index memref shape.
  // addr_buf element type is a signless integer (arith index_cast requires
  // signless).  Use the same bit-width as the index element type (typically
  // i32), but always signless.
  mlir::Type addr_elem_type = [&]() -> mlir::Type {
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(idx_elem_type))
      return mlir::IntegerType::get(ctx, it.getWidth());  // signless
    return mlir::IntegerType::get(ctx, 32);
  }();

  addr_buf_type = mlir::MemRefType::get(iab_shape, addr_elem_type,
                                        mlir::StridedLayoutAttr{},
                                        globalMemorySpaceAttr(ctx));

  // Recover the full variable-space set and order from the indirect op.
  mlir::IntegerSet full_vss =
      indirect_op.getVariablesSpaceSet().getValue();
  mlir::AffineMap full_vso = indirect_op.getVariablesSpaceOrder();

  // IAB-scoped set/order: first iab_rank dims of the full var space.
  mlir::IntegerSet iab_set = projectIntegerSet(full_vss, iab_rank);
  mlir::AffineMap iab_order = mlir::AffineMap::get(
      iab_rank, 0,
      full_vso.getResults().take_front(iab_rank), ctx);

  // ------------------------------------------------------------------
  // Scan dependencies to find which block arguments are needed.
  // We need: idx_memref dependencies + base dependencies.
  // ------------------------------------------------------------------
  llvm::SmallVector<mlir::Value> needed_block_args;
  llvm::DenseSet<mlir::Value> args_seen;
  {
    // Use a throw-away builder/mapper to discover needed block args without
    // emitting any ops. The block-arg mappings discovered here are promoted
    // into the real mapper below when the function block is available.
    mlir::OpBuilder scan_builder(ctx);
    mlir::IRMapping scan_map;
    materializeValue(idx_memref, scan_map, scan_builder, needed_block_args,
                     args_seen);
    materializeValue(base, scan_map, scan_builder, needed_block_args,
                     args_seen);
  }

  // ------------------------------------------------------------------
  // Create the idx_to_addr child module.
  // ------------------------------------------------------------------
  mlir::OpBuilder outer_builder(top_module);
  outer_builder.setInsertionPointToEnd(top_module.getBody());
  auto idx_module = mlir::ModuleOp::create(outer_builder, loc,
                                           "local_schedule_idx_to_addr");
  mlir::SymbolTable top_sym(top_module);
  (void)top_sym.renameToUnique(idx_module, {});
  llvm::StringRef idx_func_name = idx_module.getSymName().value();

  // Build function type: (block_arg_types...) -> addr_buf_type
  llvm::SmallVector<mlir::Type> arg_types;
  arg_types.reserve(needed_block_args.size());
  for (mlir::Value v : needed_block_args) arg_types.push_back(v.getType());
  auto func_type =
      mlir::FunctionType::get(ctx, arg_types, {addr_buf_type});

  // ------------------------------------------------------------------
  // Create the function.
  // ------------------------------------------------------------------
  mlir::OpBuilder mod_builder(idx_module.getBodyRegion());
  mod_builder.setInsertionPointToStart(idx_module.getBody());
  auto idx_func = mlir::func::FuncOp::create(loc, idx_func_name, func_type);
  idx_func.setPublic();
  if (auto grid_attr = original_func->getAttr("grid"))
    idx_func->setAttr("grid", grid_attr);
  mod_builder.insert(idx_func);

  mlir::Block* func_block = idx_func.addEntryBlock();
  mlir::OpBuilder builder(func_block, func_block->begin());

  // Map original block args -> new function args.
  mlir::IRMapping mapper;
  for (size_t i = 0; i < needed_block_args.size(); ++i)
    mapper.map(needed_block_args[i], func_block->getArgument(i));

  // ------------------------------------------------------------------
  // Emit the idx_to_addr body.
  // ------------------------------------------------------------------

  mlir::Value c0 = mlir::arith::ConstantIndexOp::create(builder, loc, 0);

  // Allocate the global addr_buf.
  mlir::Value addr_buf = mlir::memref::AllocOp::create(builder, loc, addr_buf_type);

  // Clone idx_memref and base dependencies into this function.
  {
    // Use a local copy of mapper so we can extend it without invalidating
    // the original before the clones are committed.  Move (not copy) the
    // result back to avoid an extra DenseMap allocation.
    mlir::IRMapping local_map(mapper);
    llvm::SmallVector<mlir::Value> dummy_args;
    llvm::DenseSet<mlir::Value> dummy_seen;
    materializeValue(idx_memref, local_map, builder, dummy_args, dummy_seen);
    materializeValue(base, local_map, builder, dummy_args, dummy_seen);
    mapper = std::move(local_map);
  }

  mlir::Value mapped_idx_memref = mapper.lookup(idx_memref);
  mlir::Value mapped_base = mapper.lookup(base);

  // ---- Load the index tensor ----
  //
  // Build ktdp.construct_access_tile over the index memref with zero offsets,
  // then ktdp.load to get the integer index tensor.

  // Access tile result type: same shape as index memref, element type = index.
  // (The AccessTileType encodes the tile *shape*, not the element type of the
  // underlying memory; the load result type has the correct element type.)
  auto iab_at_type =
      mlir::ktdp::AccessTileType::get(iab_shape, mlir::IndexType::get(ctx));

  llvm::SmallVector<mlir::Value> zero_sub(iab_rank, c0);

  // identity base_map for the access tile (no offset beyond the zero subscripts).
  mlir::AffineMap base_map_id = identityMap(ctx, iab_rank);

  auto idx_at = mlir::ktdp::ConstructAccessTilesOp::create(
      builder, loc, iab_at_type, mapped_idx_memref,
      /*base_map=*/base_map_id, /*indices=*/mlir::ValueRange(zero_sub),
      /*symbol_operands=*/mlir::ValueRange{},
      /*access_tile_set=*/iab_set, /*access_tile_order=*/iab_order);

  // ktdp.load: load into the signless addr_elem_type (avoids si32/ui32 tensor
  // issues with arith ops that require signless integer types).
  auto idx_tensor_type = mlir::RankedTensorType::get(iab_shape, addr_elem_type);
  auto idx_tensor = mlir::ktdp::LoadOp::create(
      builder, loc, idx_tensor_type, idx_at);

  // ---- Address computation ----
  //
  // Compute: addr_tensor[i,j] = base_flat_i32 + idx_tensor[i,j] * stride_i32
  //
  // where:
  //   base_flat = aligned_ptr_as_index + flat_offset (from extract_strided_metadata)
  //   stride    = stride[ind_dim] of the base memref (dim 0 = the indirect dim)

  // Extract base pointer and strides from the base memref.
  auto extract_meta =
      mlir::memref::ExtractStridedMetadataOp::create(builder, loc, mapped_base);

  mlir::Value base_ptr_memref = extract_meta.getBaseBuffer();
  mlir::Value base_flat_offset = extract_meta.getOffset();

  // The indirect dimension is dimension 0 of the base memref (by convention
  // from the spec; the first dimension is the one with the `ind()` subscript).
  mlir::Value ind_stride = extract_meta.getStrides()[0];

  // Get the flat base address as an index.
  auto aligned_ptr_idx =
      mlir::memref::ExtractAlignedPointerAsIndexOp::create(
          builder, loc, builder.getIndexType(), base_ptr_memref);

  // base_flat_index = aligned_ptr + offset
  auto base_flat_idx = mlir::arith::AddIOp::create(
      builder, loc, aligned_ptr_idx, base_flat_offset);

  // Truncate/cast scalar values to addr_elem_type (typically i32).
  unsigned addr_bits = mlir::cast<mlir::IntegerType>(addr_elem_type).getWidth();
  auto scalarToAddrType = [&](mlir::Value v) -> mlir::Value {
    if (v.getType() == addr_elem_type) return v;
    if (addr_bits <= 64)
      return mlir::arith::IndexCastUIOp::create(builder, loc, addr_elem_type, v);
    return mlir::arith::IndexCastOp::create(builder, loc, addr_elem_type, v);
  };

  mlir::Value base_scalar = scalarToAddrType(base_flat_idx);
  mlir::Value stride_scalar = scalarToAddrType(ind_stride);

  auto addr_tensor_type = mlir::RankedTensorType::get(iab_shape, addr_elem_type);

  mlir::Value base_splat =
      mlir::tensor::SplatOp::create(builder, loc, addr_tensor_type, base_scalar);
  mlir::Value stride_splat =
      mlir::tensor::SplatOp::create(builder, loc, addr_tensor_type, stride_scalar);

  // idx_tensor is already addr_elem_type (signless); no conversion needed.
  mlir::Value idx_as_addr = idx_tensor;

  // offsets = idx_as_addr * stride_splat
  mlir::Value offsets =
      mlir::arith::MulIOp::create(builder, loc, idx_as_addr, stride_splat);

  // addr_tensor = base_splat + offsets
  mlir::Value addr_tensor =
      mlir::arith::AddIOp::create(builder, loc, base_splat, offsets);

  // ---- Store addr_tensor into addr_buf ----
  //
  // We need an access tile over addr_buf to satisfy the ktdp.store contract.
  // The addr_buf is a memref<...xaddr_elem_type, global>.  We use the same
  // iab_set / iab_order and zero offsets.
  auto addr_buf_at_type =
      mlir::ktdp::AccessTileType::get(iab_shape, mlir::IndexType::get(ctx));

  auto addr_buf_at = mlir::ktdp::ConstructAccessTilesOp::create(
      builder, loc, addr_buf_at_type, addr_buf,
      /*base_map=*/base_map_id, /*indices=*/mlir::ValueRange(zero_sub),
      /*symbol_operands=*/mlir::ValueRange{},
      /*access_tile_set=*/iab_set, /*access_tile_order=*/iab_order);

  mlir::ktdp::StoreOp::create(builder, loc, addr_tensor, addr_buf_at);

  // Return the addr_buf.
  mlir::func::ReturnOp::create(builder, loc, mlir::ValueRange{addr_buf});

  return idx_module;
}

//===----------------------------------------------------------------------===//
// rewriteGatherScatterModule
//===----------------------------------------------------------------------===//

/// Boundary: all work that *consumes* the resolved addresses lives here.
///  - Load the addr_buf into an IAB (memref.alloc with "IAB" memory space).
///  - Replace `ktdp.construct_indirect_access_tile` with the lowered form.
void IndirectComputeGroupSplitPass::rewriteGatherScatterModule(
    mlir::ModuleOp child_module,
    mlir::StringRef new_func_name,
    mlir::MemRefType addr_buf_type,
    mlir::ktdp::ConstructIndirectAccessTilesOp indirect_op) {
  mlir::MLIRContext* ctx = child_module.getContext();
  mlir::Location loc = child_module.getLoc();

  // Find the single function in the child module.
  mlir::func::FuncOp func = nullptr;
  child_module.walk([&](mlir::func::FuncOp f) {
    func = f;
    return mlir::WalkResult::interrupt();
  });
  assert(func && "child module must have a function");

  // ------------------------------------------------------------------
  // Collect the intermediate variable types from the indirect op's region.
  // These are the tile IVs (%arg0..%argN in the region body).  We will
  // add them as new function arguments so the lowered op can reference them
  // from the enclosing function scope (Steps 2+3 will wrap them in loops).
  // ------------------------------------------------------------------
  auto intermediate_vars = indirect_op.getIntermediateVariables();
  llvm::SmallVector<mlir::Type> iv_types;
  iv_types.reserve(intermediate_vars.size());
  for (mlir::Value iv : intermediate_vars)
    iv_types.push_back(iv.getType());  // always index

  // ------------------------------------------------------------------
  // Prepend addr_buf_type + IV types as the first function arguments.
  // ------------------------------------------------------------------
  mlir::Block& entry = func.getBody().front();

  // Insert IV args first (they end up after addr_arg, before old args).
  llvm::SmallVector<mlir::BlockArgument> iv_func_args;
  for (unsigned i = 0; i < iv_types.size(); ++i)
    iv_func_args.push_back(entry.insertArgument(0u, iv_types[i], loc));
  // iv_func_args[0] is the LAST inserted, so they are in reverse order.
  // Reverse to restore original order.
  std::reverse(iv_func_args.begin(), iv_func_args.end());

  mlir::BlockArgument addr_arg =
      entry.insertArgument(0u, addr_buf_type, loc);

  auto old_ftype = func.getFunctionType();
  llvm::SmallVector<mlir::Type> new_arg_types = {addr_buf_type};
  new_arg_types.append(iv_types.begin(), iv_types.end());
  new_arg_types.append(old_ftype.getInputs().begin(), old_ftype.getInputs().end());
  func.setFunctionType(
      mlir::FunctionType::get(ctx, new_arg_types, old_ftype.getResults()));

  // Rename the function and its enclosing module.
  func.setSymName(new_func_name);
  child_module.setSymName(new_func_name);

  // ------------------------------------------------------------------
  // Build IAB setup + lowered op immediately before the indirect op.
  // ------------------------------------------------------------------
  mlir::OpBuilder builder(indirect_op);

  mlir::Value c0 = mlir::arith::ConstantIndexOp::create(builder, loc, 0);

  // IAB shape and type.
  llvm::ArrayRef<int64_t> iab_shape = addr_buf_type.getShape();
  unsigned iab_rank = iab_shape.size();
  mlir::Type addr_elem_type = addr_buf_type.getElementType();

  // Recover IAB set/order from the indirect op (first iab_rank dims).
  mlir::IntegerSet full_vss = indirect_op.getVariablesSpaceSet().getValue();
  mlir::AffineMap full_vso = indirect_op.getVariablesSpaceOrder();
  mlir::IntegerSet iab_set = projectIntegerSet(full_vss, iab_rank);
  mlir::AffineMap iab_order = mlir::AffineMap::get(
      iab_rank, 0,
      full_vso.getResults().take_front(iab_rank), ctx);
  mlir::AffineMap base_map_id = identityMap(ctx, iab_rank);

  llvm::SmallVector<mlir::Value> zero_sub(iab_rank, c0);

  // ---- Load addr_buf into IAB ----
  //
  // IAB memref: same shape as addr_buf, element type = addr_elem_type,
  // memory space = "IAB".  We use memref.alloc to create it.
  llvm::SmallVector<int64_t> iab_strides_static(iab_rank);
  {
    int64_t s = 1;
    for (int i = static_cast<int>(iab_rank) - 1; i >= 0; --i) {
      iab_strides_static[i] = s;
      s *= iab_shape[i];
    }
  }
  auto iab_memref_type = mlir::MemRefType::get(
      iab_shape, addr_elem_type,
      mlir::StridedLayoutAttr::get(ctx, 0, iab_strides_static),
      mlir::StringAttr::get(ctx, "IAB"));

  mlir::Value iab = mlir::memref::AllocOp::create(builder, loc, iab_memref_type);

  // Access tile type for addr_buf / IAB (index element type for the tile).
  auto tile_at_type = mlir::ktdp::AccessTileType::get(
      iab_shape, mlir::IndexType::get(ctx));

  // Construct access tile over the incoming addr_buf argument.
  auto addr_at = mlir::ktdp::ConstructAccessTilesOp::create(
      builder, loc, tile_at_type, addr_arg,
      base_map_id, mlir::ValueRange(zero_sub), mlir::ValueRange{},
      iab_set, iab_order);

  // Load the address values from the global addr_buf.
  auto addr_vals_type =
      mlir::RankedTensorType::get(iab_shape, addr_elem_type);
  auto addr_vals = mlir::ktdp::LoadOp::create(
      builder, loc, addr_vals_type, addr_at);

  // Construct access tile over the IAB buffer.
  auto iab_at = mlir::ktdp::ConstructAccessTilesOp::create(
      builder, loc, tile_at_type, iab,
      base_map_id, mlir::ValueRange(zero_sub), mlir::ValueRange{},
      iab_set, iab_order);

  // Store into the IAB.
  mlir::ktdp::StoreOp::create(builder, loc, addr_vals, iab_at);

  // ---- Build ktdp_lowering.construct_indirect_access_tile ----
  //
  // From the original op:
  //   per_dim_subscript_kinds: bool array (true = indirect)
  //   per_dim_subscript_maps:  affine map per dimension
  //   intermediate_variables:  region block args (the tile IVs)
  //
  // We reconstruct:
  //   iab_subscripts:    the IVs used to index into the index memref
  //                      (i.e. the map operands of the indirect dimension)
  //   direct_subscripts: one value per base dimension; indirect dim -> %c0

  mlir::Value orig_base = indirect_op.getBase();

  auto per_kind_attr =
      mlir::cast<mlir::ArrayAttr>(indirect_op.getPerDimSubscriptKinds());
  auto per_map_attr =
      mlir::cast<mlir::ArrayAttr>(indirect_op.getPerDimSubscriptMaps());

  // Maps in the op use the unified canonical ordering:
  //   dim[0..n_captured-1]  -> captured_variables (external SSA values)
  //   dim[n_captured..end]  -> intermediate_variables (tile IVs)
  //
  // Use the new function block args (iv_func_args) as the tile IV operands —
  // they are valid SSA values in the enclosing function scope and won't be
  // destroyed when the indirect op is erased.
  llvm::SmallVector<mlir::Value> map_operands(
      indirect_op.getCapturedVariables().begin(),
      indirect_op.getCapturedVariables().end());
  for (mlir::BlockArgument ba : iv_func_args)
    map_operands.push_back(ba);

  // Find the indirect dimension.
  int ind_dim_idx = -1;
  for (unsigned i = 0; i < per_kind_attr.size(); ++i) {
    if (mlir::cast<mlir::BoolAttr>(per_kind_attr[i]).getValue()) {
      ind_dim_idx = static_cast<int>(i);
      break;
    }
  }
  assert(ind_dim_idx >= 0 && "no indirect dimension found in op");

  // Extract IAB subscripts from the indirect dimension's affine map.
  auto ind_map =
      mlir::cast<mlir::AffineMapAttr>(per_map_attr[ind_dim_idx]).getAffineMap();
  llvm::SmallVector<mlir::Value> iab_subscripts;
  for (mlir::AffineExpr result_expr : ind_map.getResults())
    iab_subscripts.push_back(evalAffineExpr(result_expr, map_operands, builder, loc));

  // Build direct subscripts for each dimension of the base memref.
  auto base_memref_type = mlir::cast<mlir::MemRefType>(orig_base.getType());
  unsigned base_rank = base_memref_type.getRank();
  llvm::SmallVector<mlir::Value> direct_subscripts;
  direct_subscripts.reserve(base_rank);

  for (unsigned d = 0; d < base_rank; ++d) {
    bool is_indirect =
        mlir::cast<mlir::BoolAttr>(per_kind_attr[d]).getValue();
    if (is_indirect) {
      // Replace the indirect subscript with %c0.
      direct_subscripts.push_back(c0);
    } else {
      auto direct_map =
          mlir::cast<mlir::AffineMapAttr>(per_map_attr[d]).getAffineMap();
      // A direct subscript map has exactly one result (the subscript value).
      assert(direct_map.getNumResults() == 1 &&
             "expected a single-result map for each direct subscript");
      direct_subscripts.push_back(
          evalAffineExpr(direct_map.getResult(0), map_operands, builder, loc));
    }
  }

  // Create the lowered op.
  auto orig_result_type = mlir::cast<mlir::ktdp::AccessTileType>(
      indirect_op.getResult().getType());
  // Use iv_func_args as the intermediate_variables — these are valid SSA
  // values in the enclosing function scope; they won't be destroyed when the
  // original indirect op's body region is deleted on erase.
  llvm::SmallVector<mlir::Value> ivar_values(iv_func_args.begin(),
                                             iv_func_args.end());
  auto lowered_op =
      mlir::ktdp_lowering::ConstructIndirectAccessTileOp::create(
          builder, loc, orig_result_type,
          /*base=*/orig_base,
          /*ind_addr_buf_memref=*/iab,
          /*ind_addr_buf_subscripts=*/mlir::ValueRange(iab_subscripts),
          /*direct_subscripts=*/mlir::ValueRange(direct_subscripts),
          /*intermediate_variables=*/mlir::ValueRange(ivar_values),
          /*variables_space_order=*/indirect_op.getVariablesSpaceOrder(),
          /*variables_space_set=*/indirect_op.getVariablesSpaceSet());

  indirect_op.getResult().replaceAllUsesWith(lowered_op.getResult());
  indirect_op.erase();
}

//===----------------------------------------------------------------------===//
// splitChildModule
//===----------------------------------------------------------------------===//

void IndirectComputeGroupSplitPass::splitChildModule(
    mlir::ModuleOp top_module,
    mlir::ModuleOp orch_module,
    mlir::ModuleOp child_module) {
  mlir::MLIRContext* ctx = top_module.getContext();
  mlir::Location loc = top_module.getLoc();

  // Find the child function and the indirect op.
  mlir::func::FuncOp child_func = nullptr;
  child_module.walk([&](mlir::func::FuncOp f) {
    child_func = f;
    return mlir::WalkResult::interrupt();
  });
  assert(child_func);

  mlir::ktdp::ConstructIndirectAccessTilesOp indirect_op =
      findIndirectOp(child_func);
  if (!indirect_op) return;

  llvm::StringRef orig_func_name = child_func.getSymName();

  // Capture n_ivs now — indirect_op will be erased by rewriteGatherScatterModule.
  unsigned n_ivs =
      indirect_op.getVariablesSpaceSet().getValue().getNumDims();

  // Step 1: Build the idx_to_addr sibling module.
  mlir::MemRefType addr_buf_type;
  mlir::ModuleOp idx_module =
      buildIdxToAddrModule(top_module, child_func, indirect_op, addr_buf_type);
  llvm::StringRef idx_func_name = idx_module.getSymName().value();

  // Step 2: Make the existing child module's symbol unique to avoid clash
  //         with the freshly inserted idx_module.
  mlir::SymbolTable top_sym(top_module);
  (void)top_sym.renameToUnique(child_module, {});
  llvm::StringRef gather_func_name = child_module.getSymName().value();

  // Step 3: Rewrite the gather/scatter child module.
  // NOTE: indirect_op is erased inside here — do not use it after this call.
  rewriteGatherScatterModule(child_module, gather_func_name, addr_buf_type,
                             indirect_op);

  // Step 4: Update the orchestrator.
  //
  // Find the call to @orig_func_name and replace it with:
  //   %addr_buf = call @idx_func_name()
  //   call @gather_func_name(%addr_buf)
  orch_module.walk([&](mlir::func::CallOp call_op) {
    if (call_op.getCallee() != orig_func_name)
      return mlir::WalkResult::advance();

    mlir::OpBuilder orch_builder(call_op);

    // Add forward declaration for idx_to_addr.
    auto idx_fwd_type = mlir::FunctionType::get(ctx, {}, {addr_buf_type});
    auto idx_fwd =
        mlir::func::FuncOp::create(loc, idx_func_name, idx_fwd_type);
    idx_fwd.setPrivate();
    orch_module.push_back(idx_fwd);


    // Update existing forward decl for the gather/scatter function.
    // The forward decl still has the old name (orig_func_name) and old type.
    // Use a symbol-table lookup (O(1)) instead of a linear walk.
    mlir::SymbolTable orch_sym(orch_module);
    if (auto existing_fwd = llvm::dyn_cast_or_null<mlir::func::FuncOp>(
            orch_sym.lookup(orig_func_name))) {
      // New signature: (addr_buf, iv_0, ..., iv_n) -> ()
      llvm::SmallVector<mlir::Type> gather_arg_types;
      gather_arg_types.reserve(1 + n_ivs);
      gather_arg_types.push_back(addr_buf_type);
      for (unsigned i = 0; i < n_ivs; ++i)
        gather_arg_types.push_back(orch_builder.getIndexType());
      auto gather_fwd_type =
          mlir::FunctionType::get(ctx, gather_arg_types, {});
      existing_fwd.setFunctionType(gather_fwd_type);
      existing_fwd.setSymName(gather_func_name);
    }

    // Emit the two replacement calls.
    // IVs are passed as %c0 placeholders; Steps 2+3 will wrap with loops.
    auto idx_call =
        mlir::func::CallOp::create(orch_builder, loc, idx_fwd,
                                   mlir::ValueRange{});
    mlir::Value addr_buf_result = idx_call.getResult(0);

    mlir::Value c0_orch = mlir::arith::ConstantIndexOp::create(
        orch_builder, loc, 0);
    llvm::SmallVector<mlir::Value> gather_args;
    gather_args.reserve(1 + n_ivs);
    gather_args.push_back(addr_buf_result);
    for (unsigned i = 0; i < n_ivs; ++i) gather_args.push_back(c0_orch);

    mlir::func::CallOp::create(
        orch_builder, loc,
        mlir::FlatSymbolRefAttr::get(ctx, gather_func_name),
        mlir::TypeRange{}, mlir::ValueRange(gather_args));

    call_op.erase();
    return mlir::WalkResult::interrupt();
  });
}

//===----------------------------------------------------------------------===//
// processTopLevelModule / runOnOperation
//===----------------------------------------------------------------------===//

void IndirectComputeGroupSplitPass::processTopLevelModule(
    mlir::ModuleOp top_module) {
  // Expected structure after ComputeGroupExtraction:
  //
  //   top_module {
  //     orch_module { func @kernel { call @local_schedule_N() } }
  //     @local_schedule_N { func @local_schedule_N { ... ind op ... } }
  //     ...
  //   }
  //
  // The first child module is the orchestrator; the rest are compute groups.

  mlir::ModuleOp orch_module = nullptr;
  llvm::SmallVector<mlir::ModuleOp> child_modules_with_indirect;

  top_module.walk<mlir::WalkOrder::PreOrder>([&](mlir::ModuleOp mod) {
    if (mod == top_module) return mlir::WalkResult::advance();
    if (!orch_module) {
      orch_module = mod;
      return mlir::WalkResult::skip();
    }
    bool has_indirect = false;
    mod.walk([&](mlir::ktdp::ConstructIndirectAccessTilesOp) {
      has_indirect = true;
      return mlir::WalkResult::interrupt();
    });
    if (has_indirect) child_modules_with_indirect.push_back(mod);
    return mlir::WalkResult::skip();
  });

  for (mlir::ModuleOp child : child_modules_with_indirect)
    splitChildModule(top_module, orch_module, child);
}

void IndirectComputeGroupSplitPass::runOnOperation() {
  LDBG(1) << "========= " PASS_NAME " =========";
  mlir::ModuleOp module_op = getOperation();

  // Fast no-op check.
  bool has_indirect = false;
  module_op.walk([&](mlir::ktdp::ConstructIndirectAccessTilesOp) {
    has_indirect = true;
    return mlir::WalkResult::interrupt();
  });
  if (!has_indirect) return;

  processTopLevelModule(module_op);
}

}  // namespace

std::unique_ptr<mlir::Pass>
scheduler::createIndirectComputeGroupSplitPass() {
  return std::make_unique<IndirectComputeGroupSplitPass>();
}
