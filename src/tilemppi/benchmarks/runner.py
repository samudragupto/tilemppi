"""Throughput, scaling, precision, L2 cache residency, and control quality benchmark suite."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tilemppi.config import EngineConfig, MPPIConfig, SDFConfig
from tilemppi.controller.engine import MPPIEngine
from tilemppi.hardware import BackendSelector, detect_hardware
from tilemppi.l2_budget import compute_engine_l2_budget
from tilemppi.types import BackendKind, PrecisionMode, RobotKind, ScenarioKind


@dataclass(slots=True)
class ScalingEntry:
    """Single benchmark measurement point across (backend, robot, precision, K, H)."""

    backend: str
    robot: str
    precision: str
    num_rollouts: int
    horizon: int
    tile_k: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    control_hz: float
    million_steps_per_sec: float
    gflops_per_sec: float
    fused_working_set_mb: float
    unfused_working_set_mb: float
    l2_residency_pct: float
    traffic_reduction_ratio: float
    fused_arithmetic_intensity: float
    unfused_arithmetic_intensity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ImpulseRecoveryPoint:
    """Single timestep sample of closed-loop shove disturbance rejection."""

    step: int
    time_s: float
    position_error_m: float
    attitude_tilt_deg: float
    min_sdf_clearance_m: float
    impulse_active: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkSuiteReport:
    """Full structured output of the TileMPPI benchmark suite."""

    hardware: dict[str, Any]
    scaling: list[ScalingEntry]
    precision_comparison: list[ScalingEntry]
    impulse_recovery_trace: list[ImpulseRecoveryPoint]
    recovery_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware": self.hardware,
            "scaling": [s.to_dict() for s in self.scaling],
            "precision_comparison": [p.to_dict() for p in self.precision_comparison],
            "impulse_recovery_trace": [t.to_dict() for t in self.impulse_recovery_trace],
            "recovery_summary": self.recovery_summary,
        }


def _measure_config_entry(
    cfg: EngineConfig,
    warmup_steps: int,
    timed_steps: int,
) -> ScalingEntry:
    engine = MPPIEngine(cfg)
    for _ in range(warmup_steps):
        engine.step()

    latencies_ms: list[float] = []
    for _ in range(timed_steps):
        t0 = time.perf_counter()
        engine.step()
        latencies_ms.append(max(0.05, (time.perf_counter() - t0) * 1000.0))

    arr = np.asarray(latencies_ms, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    hz = 1000.0 / max(p50, 0.05)

    l2 = compute_engine_l2_budget(cfg, l2_capacity_bytes=engine.capabilities.l2_cache_bytes)
    total_steps = float(cfg.mppi.num_rollouts * cfg.mppi.horizon)
    m_steps_s = (total_steps / (p50 / 1000.0)) / 1.0e6
    gflops_s = (float(l2.total_flops_per_step) / (p50 / 1000.0)) / 1.0e9

    return ScalingEntry(
        backend=engine.active_backend.value,
        robot=cfg.robot.value,
        precision=cfg.precision.value,
        num_rollouts=cfg.mppi.num_rollouts,
        horizon=cfg.mppi.horizon,
        tile_k=cfg.mppi.tile_k,
        p50_ms=round(p50, 3),
        p95_ms=round(p95, 3),
        p99_ms=round(p99, 3),
        control_hz=round(hz, 2),
        million_steps_per_sec=round(m_steps_s, 3),
        gflops_per_sec=round(gflops_s, 2),
        fused_working_set_mb=l2.fused_working_set_mb,
        unfused_working_set_mb=l2.unfused_working_set_mb,
        l2_residency_pct=l2.l2_residency_pct,
        traffic_reduction_ratio=l2.traffic_reduction_ratio,
        fused_arithmetic_intensity=l2.fused_arithmetic_intensity,
        unfused_arithmetic_intensity=l2.unfused_arithmetic_intensity,
    )


def run_benchmark_suite(quick: bool = False) -> BenchmarkSuiteReport:
    """Execute scaling, precision, roofline, and disturbance-rejection benchmarks."""
    caps = detect_hardware()
    primary_backend = BackendSelector.resolve(BackendKind.AUTO, caps)

    if quick:
        rollout_counts = [256, 1024, 4096, 16384]
        horizon = 16
        warmup = 1
        timed = 2
        recovery_steps = 28
    else:
        rollout_counts = [256, 1024, 4096, 8192, 16384]
        horizon = 32
        warmup = 2
        timed = 5
        recovery_steps = 48

    scaling_entries: list[ScalingEntry] = []
    for k in rollout_counts:
        tile_k = min(128, k)
        cfg = EngineConfig(
            robot=RobotKind.QUADROTOR,
            scenario=ScenarioKind.OBSTACLE_SLALOM,
            backend=primary_backend,
            precision=PrecisionMode.FP32,
            mppi=MPPIConfig(
                num_rollouts=k,
                horizon=horizon,
                tile_k=tile_k,
                num_vis_rollouts=min(64, k),
            ),
            sdf=SDFConfig(max_obstacles=16),
        )
        scaling_entries.append(_measure_config_entry(cfg, warmup_steps=warmup, timed_steps=timed))

    # Precision comparison across FP32, FP16, FP8
    prec_k = 4096 if quick else 16384
    precision_entries: list[ScalingEntry] = []
    for prec in (PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.FP8):
        cfg_p = EngineConfig(
            robot=RobotKind.QUADRUPED,
            scenario=ScenarioKind.OBSTACLE_SLALOM,
            backend=primary_backend,
            precision=prec,
            mppi=MPPIConfig(
                num_rollouts=prec_k,
                horizon=horizon,
                tile_k=128,
                num_vis_rollouts=64,
            ),
            sdf=SDFConfig(max_obstacles=16),
        )
        precision_entries.append(
            _measure_config_entry(cfg_p, warmup_steps=warmup, timed_steps=timed)
        )

    # Closed-loop impulse shove recovery trace
    rec_cfg = EngineConfig(
        robot=RobotKind.QUADROTOR,
        scenario=ScenarioKind.KICK_RECOVERY,
        backend=primary_backend,
        precision=PrecisionMode.FP32,
        mppi=MPPIConfig(
            num_rollouts=512 if quick else 2048,
            horizon=16,
            tile_k=64,
            dt=0.025,
            seed=42,
        ),
        sdf=SDFConfig(max_obstacles=8),
    )
    rec_engine = MPPIEngine(rec_cfg)
    rec_engine.set_goal((0.2, 0.0, float(rec_engine.state[2])))

    trace: list[ImpulseRecoveryPoint] = []
    min_clearance = 1.0e4
    peak_tilt_deg = 0.0

    for s in range(recovery_steps):
        if s == 8:
            rec_engine.apply_impulse(
                force=(3.5, 9.5, 1.5),
                torque=(0.22, -0.18, 0.08),
                duration_steps=4,
            )
        res = rec_engine.step()
        pos_err = float(np.linalg.norm(res.state[0:3] - res.goal))
        qx, qy = float(res.state[4]), float(res.state[5])
        sin_half_tilt = min(1.0, float(np.hypot(qx, qy)))
        tilt_deg = float(np.degrees(2.0 * np.arcsin(sin_half_tilt)))
        peak_tilt_deg = max(peak_tilt_deg, tilt_deg)
        min_clearance = min(min_clearance, res.telemetry.min_sdf_clearance)

        trace.append(
            ImpulseRecoveryPoint(
                step=s,
                time_s=round(s * 0.025, 4),
                position_error_m=round(pos_err, 4),
                attitude_tilt_deg=round(tilt_deg, 3),
                min_sdf_clearance_m=round(res.telemetry.min_sdf_clearance, 4),
                impulse_active=float(np.linalg.norm(res.impulse_vec)) > 0.1,
            )
        )

    recovery_summary = {
        "steps_run": recovery_steps,
        "peak_attitude_tilt_deg": round(peak_tilt_deg, 2),
        "final_attitude_tilt_deg": trace[-1].attitude_tilt_deg,
        "final_position_error_m": trace[-1].position_error_m,
        "min_sdf_clearance_m": round(min_clearance, 4),
        "collision_free": min_clearance > 0.0,
    }

    return BenchmarkSuiteReport(
        hardware=caps.to_dict(),
        scaling=scaling_entries,
        precision_comparison=precision_entries,
        impulse_recovery_trace=trace,
        recovery_summary=recovery_summary,
    )


def format_benchmark_markdown(report: BenchmarkSuiteReport) -> str:
    """Render a publication-grade Markdown table report from a BenchmarkSuiteReport."""
    hw = report.hardware
    lines = [
        "# TileMPPI Benchmark Report",
        "",
        "## 1. Detected Hardware Profile",
        f"- **CPU:** {hw.get('cpu_model')} ({hw.get('cpu_cores_logical')} threads, AVX-512={hw.get('avx512_supported')})",
        f"- **GPU:** {hw.get('gpu_name')} (`{hw.get('compute_capability_str')}`, VRAM={hw.get('vram_total_mb')} MB)",
        f"- **L2 Cache:** {hw.get('l2_cache_mb')} MB ({hw.get('l2_cache_bytes')} bytes)",
        f"- **Active Backend:** `{hw.get('recommended_backend')}`",
        "",
        "## 2. Rollout Scaling & 32 MB L2 Cache Residency",
        "",
        "| Rollouts (K) | Horizon (H) | Backend | p50 (ms) | p95 (ms) | Loop (Hz) | M-Steps/s | GFLOP/s | Fused L2 (MB) | Unfused (MB) | L2 Residency | Traffic Reduction |",
        "| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in report.scaling:
        lines.append(
            f"| {s.num_rollouts:,} | {s.horizon} | `{s.backend}` | {s.p50_ms:.2f} | {s.p95_ms:.2f} | "
            f"**{s.control_hz:.1f}** | {s.million_steps_per_sec:.2f} | {s.gflops_per_sec:.1f} | "
            f"{s.fused_working_set_mb:.2f} | {s.unfused_working_set_mb:.1f} | "
            f"{s.l2_residency_pct:.0f}% | **{s.traffic_reduction_ratio:.1f}x** |"
        )

    lines.extend(
        [
            "",
            "## 3. Tile Precision Comparison (12-DoF Articulated Quadruped)",
            "",
            "| Precision | Rollouts (K) | p50 (ms) | Loop (Hz) | Fused L2 (MB) | Unfused (MB) | Traffic Reduction | Arithmetic Intensity (FLOP/B) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for p in report.precision_comparison:
        lines.append(
            f"| `{p.precision.upper()}` | {p.num_rollouts:,} | {p.p50_ms:.2f} | "
            f"**{p.control_hz:.1f}** | {p.fused_working_set_mb:.2f} | {p.unfused_working_set_mb:.1f} | "
            f"{p.traffic_reduction_ratio:.1f}x | {p.fused_arithmetic_intensity:.1f} |"
        )

    rec = report.recovery_summary
    lines.extend(
        [
            "",
            "## 4. Closed-Loop Impulse Shove Recovery",
            f"- **Peak Attitude Tilt Under Shove:** {rec['peak_attitude_tilt_deg']:.2f} deg",
            f"- **Final Attitude Tilt:** {rec['final_attitude_tilt_deg']:.2f} deg",
            f"- **Final Position Tracking Error:** {rec['final_position_error_m']:.3f} m",
            f"- **Minimum Obstacle SDF Clearance:** {rec['min_sdf_clearance_m']:.3f} m (Collision-Free={rec['collision_free']})",
            "",
        ]
    )
    return "\n".join(lines)
