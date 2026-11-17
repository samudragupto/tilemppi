"""Command-line interface (`tilemppi`) for hardware inspection, cuTile verification, demos, benchmarks, and serving."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.table import Table

from tilemppi.benchmarks import (
    format_benchmark_markdown,
    generate_all_benchmark_svgs,
    run_benchmark_suite,
)
from tilemppi.config import EngineConfig, MPPIConfig, SDFConfig
from tilemppi.dynamics import create_robot_model
from tilemppi.hardware import detect_hardware
from tilemppi.kernels import CuTileMPPIBackend, ReferenceCPUMPPIBackend
from tilemppi.kernels.cutile import verify_all_cutile_kernels
from tilemppi.l2_budget import compute_l2_budget
from tilemppi.planner import HeterogeneousControlRuntime
from tilemppi.scenarios import build_scenario
from tilemppi.types import BackendKind, PrecisionMode, RobotKind, ScenarioKind

app = typer.Typer(
    name="tilemppi",
    help="TileMPPI: 16,384-Parallel-Universe Real-Time Robotics Control Engine in NVIDIA cuTile.",
    add_completion=False,
)
console = Console(force_terminal=False, legacy_windows=False)


@app.command("info")
def cmd_info() -> None:
    """Print detected CPU/GPU hardware profile and 32 MB L2 cache budget for K=16,384."""
    caps = detect_hardware(force_refresh=True)
    hw_table = Table(title="TileMPPI Hardware & Backend Profile", box=box.ASCII)
    hw_table.add_column("Property", style="cyan")
    hw_table.add_column("Detected Value", style="green")

    hw_table.add_row(
        "OS / Python", f"{caps.os_name} {caps.os_release} / Python {caps.python_version}"
    )
    hw_table.add_row(
        "CPU",
        f"{caps.cpu_model} ({caps.cpu_cores_logical} threads, AVX-512={caps.avx512_supported})",
    )
    hw_table.add_row(
        "GPU",
        f"{caps.gpu_name} (sm_{caps.compute_capability[0]}{caps.compute_capability[1]}, {caps.sm_count} SMs, {caps.vram_total_mb} MB VRAM)",
    )
    hw_table.add_row(
        "L2 Cache Capacity",
        f"{caps.l2_cache_mb} MB ({caps.l2_cache_bytes:,} bytes)",
    )
    hw_table.add_row(
        "Available Backends",
        ", ".join(b.value for b in caps.available_backends),
    )
    hw_table.add_row("Recommended Backend", caps.recommended_backend.value)
    console.print(hw_table)

    l2_table = Table(
        title="RTX 4060 32 MB L2 Cache Residency Budget (K=16,384, H=32)",
        box=box.ASCII,
    )
    l2_table.add_column("Robot", style="cyan")
    l2_table.add_column("Precision")
    l2_table.add_column("Fused L2 (MB)", justify="right", style="green")
    l2_table.add_column("Unfused (MB)", justify="right", style="red")
    l2_table.add_column("L2 Residency", justify="right")
    l2_table.add_column("Traffic Reduction", justify="right", style="bold yellow")

    for robot in (RobotKind.QUADROTOR, RobotKind.QUADRUPED, RobotKind.SWARM):
        for prec in (PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.FP8):
            rep = compute_l2_budget(
                num_rollouts=16384,
                horizon=32,
                state_dim=robot.state_dim,
                control_dim=robot.control_dim,
                max_obstacles=32,
                precision=prec,
                l2_capacity_bytes=caps.l2_cache_bytes,
                robot=robot,
            )
            l2_table.add_row(
                robot.value,
                prec.value.upper(),
                f"{rep.fused_working_set_mb:.2f}",
                f"{rep.unfused_working_set_mb:.1f}",
                f"{rep.l2_residency_pct:.0f}%",
                f"{rep.traffic_reduction_ratio:.1f}x",
            )
    console.print(l2_table)


@app.command("verify-cutile")
def cmd_verify_cutile(
    tile_k: int = typer.Option(128, "--tile-k", help="Power-of-two tile block size"),
) -> None:
    """Run static AST contract check, Tile IR report, and golden parity across all @ct.kernel modules."""
    reports = verify_all_cutile_kernels(tile_k=tile_k)

    ast_table = Table(
        title=f"NVIDIA cuTile (@ct.kernel) Static AST & SRAM Verification (TILE_K={tile_k})",
        box=box.ASCII,
    )
    ast_table.add_column("Kernel Name", style="cyan")
    ast_table.add_column("Status")
    ast_table.add_column("Loads", justify="right")
    ast_table.add_column("Stores", justify="right")
    ast_table.add_column("MMA", justify="right")
    ast_table.add_column("Trans.", justify="right")
    ast_table.add_column("SRAM (KB)", justify="right")

    all_valid = True
    for r in reports:
        status = "[OK] PASS" if r.is_valid else "[FAIL]"
        if not r.is_valid:
            all_valid = False
        ast_table.add_row(
            r.kernel_name,
            status,
            str(r.num_tile_loads),
            str(r.num_tile_stores),
            str(r.num_matmul_ops),
            str(r.num_transcendental_ops),
            f"{r.estimated_sram_bytes / 1024.0:.1f}",
        )
    console.print(ast_table)

    if not all_valid:
        for r in reports:
            if not r.is_valid:
                typer.echo(f"Violations in {r.kernel_name}: {r.violations}", err=True)
        raise typer.Exit(code=1)

    # Golden numerical parity across all 3 robots
    mppi_cfg = MPPIConfig(
        num_rollouts=128,
        horizon=16,
        tile_k=min(tile_k, 128),
        num_vis_rollouts=16,
        seed=42,
    )
    sdf_cfg = SDFConfig(max_obstacles=8)

    for robot in (RobotKind.QUADROTOR, RobotKind.QUADRUPED, RobotKind.SWARM):
        scenario = build_scenario(ScenarioKind.OBSTACLE_SLALOM, robot=robot, max_obstacles=8)
        obs_table = scenario.pack_table(8)
        model = create_robot_model(robot)
        x0 = model.default_initial_state(scenario.start_position)
        u_nom = np.repeat(model.nominal_control().reshape(1, -1), mppi_cfg.horizon, axis=0)
        goal = np.asarray(scenario.goal_position, dtype=np.float32)

        ref_b = ReferenceCPUMPPIBackend(robot, mppi_cfg, sdf_cfg, PrecisionMode.FP32)
        ct_b = CuTileMPPIBackend(robot, mppi_cfg, sdf_cfg, PrecisionMode.FP32)

        u_ref, c_ref, w_ref, _, _, _ = ref_b.execute_mppi_step(
            x0, u_nom, goal, goal, obs_table, step_seed=777, temperature=1.2
        )
        u_ct, c_ct, w_ct, _, _, _ = ct_b.execute_mppi_step(
            x0, u_nom, goal, goal, obs_table, step_seed=777, temperature=1.2
        )
        max_cost_rel = float(np.max(np.abs(c_ct - c_ref) / np.maximum(np.abs(c_ref), 1.0)))
        max_u_diff = float(np.max(np.abs(u_ct - u_ref)))
        max_w_diff = float(np.max(np.abs(w_ct - w_ref)))
        typer.echo(
            f"[OK] Golden Parity [{robot.value}]: max_rel_cost={max_cost_rel:.2e}, "
            f"max_du={max_u_diff:.2e}, max_dw={max_w_diff:.2e}"
        )
        if max_cost_rel > 2e-3 or max_u_diff > 2e-3:
            typer.echo(f"[FAIL] Numerical parity exceeded threshold for {robot.value}", err=True)
            raise typer.Exit(code=1)

    typer.echo("\nSample Generated CUDA Tile IR Summary:")
    typer.echo(reports[3].tile_ir_summary)


@app.command("demo")
def cmd_demo(
    robot: RobotKind = typer.Option(RobotKind.QUADRUPED, "--robot", help="Robot platform"),
    scenario: ScenarioKind = typer.Option(
        ScenarioKind.KICK_RECOVERY, "--scenario", help="3D scenario preset"
    ),
    backend: BackendKind = typer.Option(BackendKind.AUTO, "--backend", help="Compute backend"),
    precision: PrecisionMode = typer.Option(
        PrecisionMode.FP32, "--precision", help="Tile precision"
    ),
    rollouts: int = typer.Option(1024, "--rollouts", help="Power-of-two rollout count K"),
    steps: int = typer.Option(60, "--steps", help="Number of closed-loop simulation steps"),
) -> None:
    """Run a headless closed-loop simulation with live impulse shove and print summary metrics."""
    cfg = EngineConfig(
        robot=robot,
        scenario=scenario,
        backend=backend,
        precision=precision,
        mppi=MPPIConfig(
            num_rollouts=rollouts,
            horizon=16,
            tile_k=min(128, rollouts),
            num_vis_rollouts=min(32, rollouts),
        ),
        sdf=SDFConfig(max_obstacles=16),
    )
    runtime = HeterogeneousControlRuntime(cfg)
    init_pos = runtime.engine.model.extract_position(runtime.engine.state.reshape(1, -1))[0]
    init_dist = float(np.linalg.norm(init_pos - runtime.engine.goal))

    latencies: list[float] = []
    min_clearance = 1.0e4

    shove_step = max(2, steps // 3)
    for s in range(steps):
        if s == shove_step:
            runtime.apply_impulse(
                force=(8.0, 22.0, 3.0) if robot == RobotKind.QUADRUPED else (2.5, 6.5, 1.0),
                torque=(0.2, -0.2, 0.1),
                duration_steps=3,
            )
        res = runtime.step()
        latencies.append(res.telemetry.step_ms)
        min_clearance = min(min_clearance, res.telemetry.min_sdf_clearance)

    final_pos = runtime.engine.model.extract_position(runtime.engine.state.reshape(1, -1))[0]
    final_dist = float(np.linalg.norm(final_pos - runtime.engine.goal))
    median_ms = float(np.median(latencies))
    hz = 1000.0 / max(median_ms, 0.05)

    typer.echo(
        f"[OK] Demo Complete | Robot={robot.value} | Scenario={scenario.value} | "
        f"Backend={runtime.engine.active_backend.value} | Steps={steps}\n"
        f"     Median Step={median_ms:.2f} ms ({hz:.1f} Hz) | "
        f"Goal Dist: {init_dist:.2f}m -> {final_dist:.2f}m | "
        f"Min SDF Clearance={min_clearance:.3f}m | L2 Residency={res.telemetry.l2_residency_pct:.0f}%"
    )


@app.command("benchmark")
def cmd_benchmark(
    quick: bool = typer.Option(False, "--quick", help="Run fast CI benchmark suite"),
    output: Path | None = typer.Option(None, "--output", help="Path to write JSON report"),
    markdown: Path | None = typer.Option(None, "--markdown", help="Path to write Markdown report"),
) -> None:
    """Execute scaling, precision, L2 residency, and disturbance-rejection benchmarks."""
    report = run_benchmark_suite(quick=quick)
    md_text = format_benchmark_markdown(report)
    typer.echo(md_text)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        typer.echo(f"[OK] Wrote JSON benchmark report to {output}")

    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(md_text + "\n", encoding="utf-8")
        typer.echo(f"[OK] Wrote Markdown benchmark report to {markdown}")


@app.command("export-plots")
def cmd_export_plots(
    output_dir: Path = typer.Option(
        Path("docs/assets"), "--output-dir", help="Target directory for SVG charts"
    ),
    quick: bool = typer.Option(False, "--quick", help="Use fast benchmark sweep"),
) -> None:
    """Run benchmark suite and generate all 4 publication-grade SVG charts in docs/assets/."""
    report = run_benchmark_suite(quick=quick)
    charts = generate_all_benchmark_svgs(report, output_dir=output_dir)
    for name, path in charts.items():
        typer.echo(f"[OK] Generated SVG chart [{name}]: {path}")


@app.command("serve")
def cmd_serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host address"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
) -> None:
    """Launch the live FastAPI + binary WebSocket server and 120 FPS 3D playground."""
    import uvicorn

    from tilemppi.server.app import create_app

    fastapi_app = create_app()
    typer.echo(f"Starting TileMPPI 3D Playground Server at http://{host}:{port}")
    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
