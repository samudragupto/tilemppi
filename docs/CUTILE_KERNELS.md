# NVIDIA `cuTile` (`@ct.kernel`) Deep-Dive

**TileMPPI** implements its entire Model Predictive Path Integral (MPPI) rollout, 3D Signed Distance Function collision evaluation, articulated link kinematics, and log-sum-exp importance weighting pipeline in the **NVIDIA `cuTile` (`cuda.tile`)** tile-based GPU programming model introduced in CUDA 13.1+.

---

## 1. Core `cuTile` Programming Principles

Unlike traditional SIMT CUDA C++ or PTX kernels that expose individual thread IDs (`threadIdx.x`), warp lane masks, and manual shared-memory bank conflict padding, `cuTile` expresses computation over **compile-time power-of-two tiles**:

1. **Pure Block-Level Indexing (`ct.bid`):** Each tile block `bid_k = ct.bid(0)` owns a contiguous slice of `TILE_K` $\in \{64, 128, 256\}$ parallel rollout universes.
2. **Block Tile Memory Transfers (`ct.load`, `ct.store`):** Global memory accesses occur strictly at tile boundaries with power-of-two shapes `(TILE_K, D_X)` or `(MAX_OBS, 16)`.
3. **Register-Resident Horizon Loops:** All 4 Runge-Kutta (`RK4`) integration stages and intermediate horizon states $t = 0 \dots H-1$ remain in SM registers and per-block SRAM—eliminating **>98% of global DRAM traffic**.
4. **Tensor Core Tile Contractions (`ct.matmul`, `ct.mma`):** Articulated 3-link forward kinematics, Jacobian-transpose force mapping, and weighted perturbation reduction $\sum_k w_k \boldsymbol{\epsilon}_k$ map directly to Tensor Core matrix-multiply-accumulate instructions.

---

## 2. Canonical `@ct.kernel` Inventory

| Module | `@ct.kernel` Function | Grid Dimensions | Primary Tile Shapes | Tensor Core / Transcendental Ops | Per-Block SRAM (`TILE_K=128`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `noise_kernel.py` | `philox_ou_noise_kernel` | `(cdiv(K, TILE_K), 1, 1)` | `(TILE_K, D_U)` | `ct.sqrt`, `ct.log`, `ct.cos` (Box-Muller + OU) | `28.0 KB` |
| `sdf_kernel.py` | `tiled_sdf_collision_kernel` | `(cdiv(K, TILE_K), 1, 1)` | `(TILE_K, 3)`, `(MAX_OBS, 16)` | `ct.sqrt`, `ct.rsqrt`, `ct.exp` | `28.0 KB` |
| `link_dynamics_kernel.py` | `batched_quadruped_fk_jacobian_kernel` | `(cdiv(K, TILE_K), 1, 1)` | `(TILE_K, 32)`, `(TILE_K, 4, 3)` | `ct.matmul`, `ct.mma`, `ct.sin`, `ct.cos` | `48.0 KB` |
| `fused_rollout_kernel.py` | `fused_quadrotor_mppi_kernel` | `(cdiv(K, TILE_K), 1, 1)` | `(TILE_K, 16)`, `(H, 4)` | `ct.sqrt`, `ct.exp`, `ct.sum` | `14.0 KB` |
| `fused_rollout_kernel.py` | `fused_quadruped_mppi_kernel` | `(cdiv(K, TILE_K), 1, 1)` | `(TILE_K, 32)`, `(H, 16)` | `ct.sin`, `ct.cos`, `ct.atan2`, `ct.exp` | `28.0 KB` |
| `fused_rollout_kernel.py` | `fused_swarm_mppi_kernel` | `(cdiv(K, TILE_K), 1, 1)` | `(TILE_K, 32)`, `(H, 16)` | `ct.sqrt`, `ct.exp`, `ct.sum` | `28.0 KB` |
| `softmax_reduce_kernel.py` | `cost_block_stats_kernel` | `(cdiv(K, TILE_K), 1, 1)` | `(TILE_K,)` | `ct.min`, `ct.exp`, `ct.sum` | `28.0 KB` |
| `softmax_reduce_kernel.py` | `softmax_weighted_update_kernel` | `(cdiv(K, TILE_K), 1, 1)` | `(1, TILE_K)`, `(TILE_K, D_U)` | `ct.matmul`, `ct.exp`, `ct.sum` | `32.0 KB` |

---

## 3. Static AST Contract Verification (`tilemppi verify-cutile`)

Every `@ct.kernel` function is continuously verified by `verify_cutile_kernel()` in `src/tilemppi/cutile_compat.py`:

- **Decorator Contract:** Must be decorated with `@ct.kernel`.
- **Block Coordinate Contract:** Must index blocks via `ct.bid(axis)` and never reference SIMT identifiers (`threadIdx`, `blockIdx`, `__syncthreads`, `__shfl_sync`).
- **Power-of-Two Tile Shape Contract:** Every `ct.load`, `ct.store`, `ct.zeros`, `ct.ones`, `ct.full`, `ct.arange`, `ct.reshape`, `ct.broadcast_to`, and `ct.extract` call validates that tile dimensions are powers of two.
- **SRAM Budget Contract:** Per-block active SRAM footprint must remain strictly below `96 KB` across `TILE_K` $\in \{64, 128, 256\}$.
- **Tile-Size Mathematical Invariance:** Changing `TILE_K` across `{64, 128, 256}` produces identical trajectory costs, importance weights, and control updates.

