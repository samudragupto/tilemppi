# TileMPPI Benchmark Report

## 1. Detected Hardware Profile
- **CPU:** AMD Ryzen 7 7840HS w/ Radeon 780M Graphics (16 threads, AVX-512=True)
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (`sm_89`, VRAM=8187.5 MB)
- **L2 Cache:** 32.0 MB (33554432 bytes)
- **Active Backend:** `torch_gpu`

## 2. Rollout Scaling & 32 MB L2 Cache Residency

| Rollouts (K) | Horizon (H) | Backend | p50 (ms) | p95 (ms) | Loop (Hz) | M-Steps/s | GFLOP/s | Fused L2 (MB) | Unfused (MB) | L2 Residency | Traffic Reduction |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 32 | `torch_gpu` | 34.21 | 34.27 | **29.2** | 0.24 | 0.8 | 0.09 | 6.8 | 100% | **76.0x** |
| 1,024 | 32 | `torch_gpu` | 29.99 | 30.37 | **33.4** | 1.09 | 3.5 | 0.28 | 27.2 | 100% | **96.2x** |
| 4,096 | 32 | `torch_gpu` | 31.76 | 31.87 | **31.5** | 4.13 | 13.0 | 1.06 | 108.8 | 100% | **103.0x** |
| 8,192 | 32 | `torch_gpu` | 34.32 | 34.86 | **29.1** | 7.64 | 24.1 | 2.09 | 217.5 | 100% | **104.2x** |
| 16,384 | 32 | `torch_gpu` | 40.66 | 40.91 | **24.6** | 12.89 | 40.8 | 4.15 | 435.1 | 100% | **104.8x** |

## 3. Tile Precision Comparison (12-DoF Articulated Quadruped)

| Precision | Rollouts (K) | p50 (ms) | Loop (Hz) | Fused L2 (MB) | Unfused (MB) | Traffic Reduction | Arithmetic Intensity (FLOP/B) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FP32` | 16,384 | 81.28 | **12.3** | 8.15 | 772.1 | 94.7x | 436.8 |
| `FP16` | 16,384 | 81.83 | **12.2** | 16.15 | 772.1 | 47.8x | 220.4 |
| `FP8` | 16,384 | 83.53 | **12.0** | 8.15 | 772.1 | 94.7x | 436.8 |

## 4. Closed-Loop Impulse Shove Recovery
- **Peak Attitude Tilt Under Shove:** 46.12 deg
- **Final Attitude Tilt:** 13.03 deg
- **Final Position Tracking Error:** 0.336 m
- **Minimum Obstacle SDF Clearance:** 0.349 m (Collision-Free=True)

