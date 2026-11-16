# Changelog

All notable changes to **TileMPPI** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-16

### Added
- Canonical NVIDIA `cuTile` (`@ct.kernel`) fused L2-resident MPPI kernel suite (`philox_ou_noise_kernel`, `tiled_sdf_collision_kernel`, `batched_quadruped_fk_jacobian_kernel`, `fused_quadrotor_mppi_kernel`, `fused_quadruped_mppi_kernel`, `fused_swarm_mppi_kernel`, `cost_block_stats_kernel`, `softmax_weighted_update_kernel`).
- Strict `cutile_compat` tile-granular execution emulator and static AST contract verifier (`verify_cutile_kernel`).
- Unified 5-mode backend selector (`cutile`, `cutile_sim`, `triton`, `torch_gpu`, `numpy_cpu`) with FP32, FP16, and FP8 (`e4m3fn`) precision modes.
- Analytical RTX 4060 (`sm_89`) 32 MB L2 cache budget & Roofline planner (`l2_budget.py`).
- Batched power-of-2 rigid & articulated robot models: 6-DoF SE(3) Quadrotor (`16x4`), 12-DoF Articulated Quadruped (`32x16`), and 4-Drone Coordinated 3D Swarm (`32x16`).
- Exact 3D Signed Distance Function primitives (`Sphere`, `Capsule`, `OrientedBox`, `TorusGate`, `GroundPlane`) with dynamic obstacle trajectories, analytical normals, and smooth barrier costs.
- Asynchronous 3D voxel A* global planner paired with GPU MPPI for concave U-trap escape.
- FastAPI control server with compact `"TMPI"` binary WebSocket telemetry protocol.
- Interactive 120 FPS React + Three.js 3D robotics playground with parallel-universe trajectory ribbons, articulated robot meshes, mouse-shove impulse perturbations, and live L2 cache telemetry HUD.
- Publication-grade SVG benchmark chart generator and full GitHub Actions CI/CD pipeline.
