"""Tests for the TileMPPI CLI (`tilemppi`), benchmark runner, and SVG chart generator."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tilemppi.benchmarks import (
    format_benchmark_markdown,
    generate_all_benchmark_svgs,
    run_benchmark_suite,
)
from tilemppi.cli import app

runner = CliRunner()


def test_cli_info_and_verify_cutile() -> None:
    res_info = runner.invoke(app, ["info"])
    assert res_info.exit_code == 0
    assert "TileMPPI Hardware" in res_info.stdout
    assert "16,384" in res_info.stdout

    res_verify = runner.invoke(app, ["verify-cutile", "--tile-k", "128"])
    assert res_verify.exit_code == 0
    assert "[OK] Golden Parity [quadrotor]" in res_verify.stdout
    assert "[OK] Golden Parity [quadruped]" in res_verify.stdout
    assert "[OK] Golden Parity [swarm]" in res_verify.stdout


def test_cli_demo_benchmark_and_export_plots(tmp_path: Path) -> None:
    res_demo = runner.invoke(
        app,
        [
            "demo",
            "--robot",
            "quadrotor",
            "--scenario",
            "kick_recovery",
            "--rollouts",
            "128",
            "--steps",
            "12",
        ],
    )
    assert res_demo.exit_code == 0
    assert "[OK] Demo Complete" in res_demo.stdout

    json_out = tmp_path / "bench.json"
    md_out = tmp_path / "bench.md"
    res_bench = runner.invoke(
        app,
        [
            "benchmark",
            "--quick",
            "--output",
            str(json_out),
            "--markdown",
            str(md_out),
        ],
    )
    assert res_bench.exit_code == 0
    assert json_out.is_file()
    assert md_out.is_file()

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert "hardware" in data
    assert len(data["scaling"]) >= 3
    assert len(data["precision_comparison"]) == 3
    assert data["recovery_summary"]["steps_run"] > 0

    svg_dir = tmp_path / "assets"
    res_plots = runner.invoke(
        app,
        ["export-plots", "--quick", "--output-dir", str(svg_dir)],
    )
    assert res_plots.exit_code == 0
    for fname in (
        "l2_cache_residency.svg",
        "throughput_scaling.svg",
        "roofline_rtx4060.svg",
        "impulse_recovery.svg",
    ):
        svg_path = svg_dir / fname
        assert svg_path.is_file()
        content = svg_path.read_text(encoding="utf-8")
        assert "<svg" in content and "</svg>" in content


def test_benchmark_suite_direct_api(tmp_path: Path) -> None:
    report = run_benchmark_suite(quick=True)
    md = format_benchmark_markdown(report)
    assert "# TileMPPI Benchmark Report" in md
    charts = generate_all_benchmark_svgs(report, output_dir=tmp_path)
    assert len(charts) == 4
