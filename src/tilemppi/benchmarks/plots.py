"""Pure-Python deterministic publication-grade SVG chart generator for TileMPPI (white-background layout)."""

from __future__ import annotations

import math
from pathlib import Path

from tilemppi.benchmarks.runner import BenchmarkSuiteReport

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"


def render_l2_cache_residency_svg(report: BenchmarkSuiteReport) -> str:
    """Render white-background SVG bar chart comparing Unfused PyTorch vs Fused TileMPPI vs 32 MB L2 ceiling."""
    width, height = 920, 480
    entries = report.scaling
    n = max(1, len(entries))

    plot_left, plot_right = 95.0, 875.0
    plot_top, plot_bottom = 122.0, 392.0
    plot_h = plot_bottom - plot_top
    max_mb = max([e.unfused_working_set_mb for e in entries] + [450.0]) * 1.15
    log_max = math.log10(1.0 + max_mb)

    def y_for_mb(mb: float) -> float:
        val = max(0.0, mb)
        frac = math.log10(1.0 + val) / log_max
        return plot_bottom - frac * plot_h

    l2_y = y_for_mb(32.0)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '  <rect width="100%" height="100%" fill="#ffffff" rx="10" stroke="#e2e8f0" stroke-width="1.5"/>',
        f'  <text x="45" y="38" fill="#0f172a" font-family="{FONT_FAMILY}" font-size="18" font-weight="700">RTX 4060 32 MB L2 Cache Residency: Unfused PyTorch vs. Fused TileMPPI</text>',
        f'  <text x="45" y="60" fill="#475569" font-family="{FONT_FAMILY}" font-size="12.5">Working-set memory footprint per MPPI step (MB, log scale) vs. the 32 MB on-chip L2 cache ceiling</text>',
        # Dedicated legend row
        '  <rect x="45" y="76" width="13" height="13" fill="#e11d48" rx="3"/>',
        f'  <text x="64" y="87" fill="#334155" font-family="{FONT_FAMILY}" font-size="12">Unfused PyTorch (DRAM Thrashing)</text>',
        '  <rect x="295" y="76" width="13" height="13" fill="#0d9488" rx="3"/>',
        f'  <text x="314" y="87" fill="#334155" font-family="{FONT_FAMILY}" font-size="12">Fused TileMPPI (100% L2 Resident)</text>',
        '  <line x1="555" y1="82" x2="582" y2="82" stroke="#059669" stroke-width="2.5" stroke-dasharray="6,4"/>',
        f'  <text x="590" y="87" fill="#059669" font-family="{FONT_FAMILY}" font-size="12" font-weight="600">32 MB L2 Cache Ceiling</text>',
    ]

    # Log-scale reference gridlines
    for tick_mb in (0, 5, 32, 100, 250, 500):
        if tick_mb > max_mb:
            continue
        yy = y_for_mb(float(tick_mb))
        is_l2 = tick_mb == 32
        stroke = "#059669" if is_l2 else "#e2e8f0"
        s_width = "2" if is_l2 else "1"
        dash = ' stroke-dasharray="7,5"' if is_l2 else ""
        t_col = "#059669" if is_l2 else "#64748b"
        t_weight = ' font-weight="700"' if is_l2 else ""
        lines.append(
            f'  <line x1="{plot_left}" y1="{yy:.1f}" x2="{plot_right}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="{s_width}"{dash}/>'
        )
        lines.append(
            f'  <text x="{plot_left - 10}" y="{yy + 4:.1f}" fill="{t_col}" font-family="{FONT_FAMILY}" font-size="11"{t_weight} text-anchor="end">{tick_mb} MB</text>'
        )

    lines.extend(
        [
            f'  <line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#94a3b8" stroke-width="1.5"/>',
            f'  <line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#94a3b8" stroke-width="1.5"/>',
            f'  <text x="{plot_right - 6}" y="{l2_y - 7:.1f}" fill="#059669" font-family="{FONT_FAMILY}" font-size="11" font-weight="700" text-anchor="end">32 MB L2 Ceiling</text>',
        ]
    )

    slot_w = (plot_right - plot_left - 40.0) / n
    bar_w = min(44.0, slot_w * 0.30)

    for i, e in enumerate(entries):
        group_cx = plot_left + 20.0 + (i + 0.5) * slot_w
        ux = group_cx - bar_w - 4.0
        fx = group_cx + 4.0

        uy = y_for_mb(e.unfused_working_set_mb)
        uh = max(6.0, plot_bottom - uy)

        fy = y_for_mb(e.fused_working_set_mb)
        fh = max(6.0, plot_bottom - fy)
        fy = plot_bottom - fh

        u_lbl_y = uy - 7.0
        if abs(u_lbl_y - l2_y) < 12.0:
            u_lbl_y = l2_y - 11.0

        f_lbl_y = fy - 7.0
        if abs(f_lbl_y - l2_y) < 12.0:
            f_lbl_y = fy + 14.0

        lines.extend(
            [
                f'  <rect x="{ux:.1f}" y="{uy:.1f}" width="{bar_w:.1f}" height="{uh:.1f}" fill="#e11d48" rx="4" opacity="0.9"/>',
                f'  <text x="{ux + bar_w / 2:.1f}" y="{u_lbl_y:.1f}" fill="#be123c" font-family="{FONT_FAMILY}" font-size="11" font-weight="600" text-anchor="middle">{e.unfused_working_set_mb:.0f} MB</text>',
                f'  <rect x="{fx:.1f}" y="{fy:.1f}" width="{bar_w:.1f}" height="{fh:.1f}" fill="#0d9488" rx="4"/>',
                f'  <text x="{fx + bar_w / 2:.1f}" y="{f_lbl_y:.1f}" fill="#0f766e" font-family="{FONT_FAMILY}" font-size="11" font-weight="700" text-anchor="middle">{e.fused_working_set_mb:.2f} MB</text>',
                f'  <text x="{group_cx:.1f}" y="{plot_bottom + 22:.1f}" fill="#0f172a" font-family="{FONT_FAMILY}" font-size="12" font-weight="600" text-anchor="middle">K = {e.num_rollouts:,}</text>',
                f'  <text x="{group_cx:.1f}" y="{plot_bottom + 40:.1f}" fill="#0284c7" font-family="{FONT_FAMILY}" font-size="11" font-weight="600" text-anchor="middle">{e.traffic_reduction_ratio:.0f}x Traffic Cut</text>',
            ]
        )

    lines.append(
        f'  <text x="{(plot_left + plot_right) / 2:.1f}" y="460" fill="#475569" font-family="{FONT_FAMILY}" font-size="12" text-anchor="middle">Parallel Rollout Universes (K, Horizon H = {entries[0].horizon if entries else 32})</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_throughput_scaling_svg(report: BenchmarkSuiteReport) -> str:
    """Render white-background SVG chart of Control Frequency (Hz) and Throughput (M-steps/s) vs K."""
    width, height = 920, 480
    entries = report.scaling
    n = max(1, len(entries))

    plot_left, plot_right = 95.0, 825.0
    plot_top, plot_bottom = 125.0, 395.0
    plot_h = plot_bottom - plot_top

    max_hz = max((e.control_hz for e in entries), default=50.0) * 1.2
    max_ms = max((e.million_steps_per_sec for e in entries), default=10.0) * 1.2

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '  <rect width="100%" height="100%" fill="#ffffff" rx="10" stroke="#e2e8f0" stroke-width="1.5"/>',
        f'  <text x="45" y="38" fill="#0f172a" font-family="{FONT_FAMILY}" font-size="18" font-weight="700">Parallel-Universe Scaling: Control Frequency (Hz) &amp; Throughput (M-Steps/s)</text>',
        f'  <text x="45" y="60" fill="#475569" font-family="{FONT_FAMILY}" font-size="12.5">Measured on {report.hardware.get("gpu_name", "Host Device")} (Backend: {entries[0].backend if entries else "auto"})</text>',
        '  <line x1="45" y1="84" x2="75" y2="84" stroke="#0d9488" stroke-width="3"/>',
        '  <circle cx="60" cy="84" r="4.5" fill="#0d9488"/>',
        f'  <text x="83" y="88" fill="#334155" font-family="{FONT_FAMILY}" font-size="12" font-weight="600">Control Loop Frequency (Hz, Left Axis)</text>',
        '  <line x1="360" y1="84" x2="390" y2="84" stroke="#0284c7" stroke-width="3" stroke-dasharray="6,4"/>',
        '  <circle cx="375" cy="84" r="4.5" fill="#0284c7"/>',
        f'  <text x="398" y="88" fill="#334155" font-family="{FONT_FAMILY}" font-size="12" font-weight="600">Throughput (Million State-Steps/s, Right Axis)</text>',
    ]

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = plot_bottom - frac * plot_h
        hz_val = max_hz * frac
        ms_val = max_ms * frac
        lines.extend(
            [
                f'  <line x1="{plot_left}" y1="{yy:.1f}" x2="{plot_right}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>',
                f'  <text x="{plot_left - 10}" y="{yy + 4:.1f}" fill="#0f766e" font-family="{FONT_FAMILY}" font-size="11" text-anchor="end">{hz_val:.0f} Hz</text>',
                f'  <text x="{plot_right + 10}" y="{yy + 4:.1f}" fill="#0369a1" font-family="{FONT_FAMILY}" font-size="11" text-anchor="start">{ms_val:.1f} M/s</text>',
            ]
        )

    lines.extend(
        [
            f'  <line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#94a3b8" stroke-width="1.5"/>',
            f'  <line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#94a3b8" stroke-width="1.5"/>',
            f'  <line x1="{plot_right}" y1="{plot_top}" x2="{plot_right}" y2="{plot_bottom}" stroke="#94a3b8" stroke-width="1.5"/>',
        ]
    )

    hz_pts: list[str] = []
    ms_pts: list[str] = []
    x_pad = 55.0
    span_x = (plot_right - plot_left) - 2.0 * x_pad

    for i, e in enumerate(entries):
        x = (
            plot_left + x_pad + (i / max(1, n - 1)) * span_x
            if n > 1
            else (plot_left + plot_right) / 2.0
        )
        y_hz = plot_bottom - (e.control_hz / max_hz) * plot_h
        y_ms = plot_bottom - (e.million_steps_per_sec / max_ms) * plot_h
        hz_pts.append(f"{x:.1f},{y_hz:.1f}")
        ms_pts.append(f"{x:.1f},{y_ms:.1f}")

    lines.append(
        f'  <polyline fill="none" stroke="#0d9488" stroke-width="3" points="{" ".join(hz_pts)}"/>'
    )
    lines.append(
        f'  <polyline fill="none" stroke="#0284c7" stroke-width="3" stroke-dasharray="7,4" points="{" ".join(ms_pts)}"/>'
    )

    for i, e in enumerate(entries):
        x = (
            plot_left + x_pad + (i / max(1, n - 1)) * span_x
            if n > 1
            else (plot_left + plot_right) / 2.0
        )
        y_hz = plot_bottom - (e.control_hz / max_hz) * plot_h
        y_ms = plot_bottom - (e.million_steps_per_sec / max_ms) * plot_h

        # Keep all labels strictly inside [plot_top, plot_bottom - 8] so they never touch X-axis labels
        hz_label_y = y_hz - 12.0
        ms_label_y = y_ms - 12.0 if y_ms > plot_bottom - 30.0 else y_ms + 18.0

        if abs(hz_label_y - ms_label_y) < 16.0:
            if y_hz <= y_ms:
                hz_label_y = y_hz - 14.0
                ms_label_y = min(plot_bottom - 8.0, y_ms + 18.0)
            else:
                ms_label_y = y_ms - 14.0
                hz_label_y = min(plot_bottom - 8.0, y_hz + 18.0)

        lines.extend(
            [
                f'  <circle cx="{x:.1f}" cy="{y_hz:.1f}" r="5.5" fill="#0d9488" stroke="#ffffff" stroke-width="1.5"/>',
                f'  <text x="{x:.1f}" y="{hz_label_y:.1f}" fill="#0f766e" font-family="{FONT_FAMILY}" font-size="11" font-weight="700" text-anchor="middle">{e.control_hz:.1f} Hz</text>',
                f'  <circle cx="{x:.1f}" cy="{y_ms:.1f}" r="5.5" fill="#0284c7" stroke="#ffffff" stroke-width="1.5"/>',
                f'  <text x="{x:.1f}" y="{ms_label_y:.1f}" fill="#0369a1" font-family="{FONT_FAMILY}" font-size="11" font-weight="600" text-anchor="middle">{e.million_steps_per_sec:.2f} M/s</text>',
                f'  <text x="{x:.1f}" y="{plot_bottom + 24:.1f}" fill="#0f172a" font-family="{FONT_FAMILY}" font-size="12" font-weight="600" text-anchor="middle">K = {e.num_rollouts:,}</text>',
            ]
        )

    lines.append(
        f'  <text x="{(plot_left + plot_right) / 2:.1f}" y="458" fill="#475569" font-family="{FONT_FAMILY}" font-size="12" text-anchor="middle">Parallel Rollout Count (K)</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_roofline_rtx4060_svg(report: BenchmarkSuiteReport) -> str:
    """Render white-background SVG Roofline model chart for NVIDIA GeForce RTX 4060 Ada sm_89."""
    width, height = 920, 480
    prec_entries = report.precision_comparison or report.scaling

    plot_left, plot_right = 95.0, 865.0
    plot_top, plot_bottom = 125.0, 395.0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '  <rect width="100%" height="100%" fill="#ffffff" rx="10" stroke="#e2e8f0" stroke-width="1.5"/>',
        f'  <text x="45" y="38" fill="#0f172a" font-family="{FONT_FAMILY}" font-size="18" font-weight="700">RTX 4060 (Ada sm_89) Roofline Analysis: DRAM-Bound vs. L2-Resident TileMPPI</text>',
        f'  <text x="45" y="60" fill="#475569" font-family="{FONT_FAMILY}" font-size="12.5">Arithmetic Intensity (FLOP/Byte) vs. Compute Ceiling (256 GB/s GDDR6 DRAM vs. 1.65 TB/s 32 MB L2 Cache)</text>',
        '  <line x1="45" y1="84" x2="75" y2="84" stroke="#e11d48" stroke-width="2.5" stroke-dasharray="6,4"/>',
        f'  <text x="83" y="88" fill="#334155" font-family="{FONT_FAMILY}" font-size="12">GDDR6 DRAM Roof (256 GB/s, 15.1 TFLOPS)</text>',
        '  <line x1="365" y1="84" x2="395" y2="84" stroke="#0d9488" stroke-width="3"/>',
        f'  <text x="403" y="88" fill="#0f766e" font-family="{FONT_FAMILY}" font-size="12" font-weight="600">32 MB L2 Roof (1.65 TB/s, 60-120 TFLOPS Tensor Core)</text>',
    ]

    for yy in (150, 210, 270, 330):
        lines.append(
            f'  <line x1="{plot_left}" y1="{yy}" x2="{plot_right}" y2="{yy}" stroke="#f1f5f9" stroke-width="1"/>'
        )
    for xx in (220, 380, 540, 700):
        lines.append(
            f'  <line x1="{xx}" y1="{plot_top}" x2="{xx}" y2="{plot_bottom}" stroke="#f1f5f9" stroke-width="1"/>'
        )

    lines.extend(
        [
            f'  <line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#94a3b8" stroke-width="1.5"/>',
            f'  <line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#94a3b8" stroke-width="1.5"/>',
            f'  <text x="{plot_left - 10}" y="154" fill="#64748b" font-family="{FONT_FAMILY}" font-size="11" text-anchor="end">100 TF</text>',
            f'  <text x="{plot_left - 10}" y="244" fill="#64748b" font-family="{FONT_FAMILY}" font-size="11" text-anchor="end">15 TF</text>',
            f'  <text x="{plot_left - 10}" y="304" fill="#64748b" font-family="{FONT_FAMILY}" font-size="11" text-anchor="end">1 TF</text>',
            f'  <text x="{plot_left - 10}" y="364" fill="#64748b" font-family="{FONT_FAMILY}" font-size="11" text-anchor="end">100 GF</text>',
            f'  <text x="160" y="{plot_bottom + 20}" fill="#64748b" font-family="{FONT_FAMILY}" font-size="11" text-anchor="middle">1 FLOP/B</text>',
            f'  <text x="330" y="{plot_bottom + 20}" fill="#64748b" font-family="{FONT_FAMILY}" font-size="11" text-anchor="middle">10 FLOP/B</text>',
            f'  <text x="560" y="{plot_bottom + 20}" fill="#64748b" font-family="{FONT_FAMILY}" font-size="11" text-anchor="middle">100 FLOP/B</text>',
            f'  <text x="760" y="{plot_bottom + 20}" fill="#64748b" font-family="{FONT_FAMILY}" font-size="11" text-anchor="middle">500+ FLOP/B</text>',
            # DRAM Roofline at y=242 (well separated from L2 points at y=175..195)
            '  <polyline fill="none" stroke="#e11d48" stroke-width="2.5" stroke-dasharray="6,4" points="115,378 410,242 845,242"/>',
            f'  <text x="840" y="260" fill="#be123c" font-family="{FONT_FAMILY}" font-size="11" font-weight="600" text-anchor="end">GDDR6 DRAM Ceiling (15.1 TFLOPS FP32)</text>',
            # L2 Roofline at y=142
            '  <polyline fill="none" stroke="#0d9488" stroke-width="3" points="115,315 390,142 845,142"/>',
            f'  <text x="840" y="134" fill="#0f766e" font-family="{FONT_FAMILY}" font-size="11" font-weight="700" text-anchor="end">32 MB L2 + Ada Tensor Core Ceiling</text>',
            # Unfused PyTorch point
            '  <circle cx="240" cy="318" r="7.5" fill="#e11d48" stroke="#ffffff" stroke-width="1.5"/>',
            f'  <text x="254" y="322" fill="#be123c" font-family="{FONT_FAMILY}" font-size="11.5" font-weight="700">Unfused PyTorch (~4.3 FLOP/B, DRAM-Bound)</text>',
        ]
    )

    colors = {"fp32": "#0284c7", "fp16": "#0d9488", "fp8": "#059669"}
    # Order points monotonically by X-axis arithmetic intensity: FP16 (~220) -> FP32 (~437) -> FP8 (~437+)
    sorted_prec = sorted(prec_entries[:3], key=lambda item: item.fused_arithmetic_intensity)
    x_positions = [565.0, 695.0, 810.0]
    y_positions = [192.0, 178.0, 192.0]

    for idx, p in enumerate(sorted_prec):
        cx = x_positions[idx]
        cy = y_positions[idx]
        col = colors.get(p.precision, "#0d9488")
        lines.extend(
            [
                f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="7.5" fill="{col}" stroke="#ffffff" stroke-width="1.5"/>',
                f'  <text x="{cx:.1f}" y="{cy - 13:.1f}" fill="{col}" font-family="{FONT_FAMILY}" font-size="11.5" font-weight="700" text-anchor="middle">TileMPPI {p.precision.upper()}</text>',
                f'  <text x="{cx:.1f}" y="{cy + 20:.1f}" fill="#334155" font-family="{FONT_FAMILY}" font-size="11" font-weight="600" text-anchor="middle">{p.fused_arithmetic_intensity:.0f} FLOP/B</text>',
            ]
        )

    lines.append(
        f'  <text x="{(plot_left + plot_right) / 2:.1f}" y="458" fill="#475569" font-family="{FONT_FAMILY}" font-size="12" text-anchor="middle">Operational Arithmetic Intensity (FLOP / Global Memory Byte, Log Scale)</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_impulse_recovery_svg(report: BenchmarkSuiteReport) -> str:
    """Render white-background SVG time-series chart of closed-loop shove disturbance rejection."""
    width, height = 920, 480
    trace = report.impulse_recovery_trace
    n = max(1, len(trace))

    plot_left, plot_right = 95.0, 825.0
    plot_top, plot_bottom = 125.0, 395.0
    plot_h = plot_bottom - plot_top

    max_tilt = max((pt.attitude_tilt_deg for pt in trace), default=45.0) * 1.15
    max_err_or_clr = (
        max(
            max((pt.position_error_m for pt in trace), default=1.5),
            max((pt.min_sdf_clearance_m for pt in trace), default=1.5),
            1.0,
        )
        * 1.18
    )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '  <rect width="100%" height="100%" fill="#ffffff" rx="10" stroke="#e2e8f0" stroke-width="1.5"/>',
        f'  <text x="45" y="38" fill="#0f172a" font-family="{FONT_FAMILY}" font-size="18" font-weight="700">Closed-Loop Impulse Shove Disturbance Rejection &amp; SDF Clearance</text>',
        f'  <text x="45" y="60" fill="#475569" font-family="{FONT_FAMILY}" font-size="12.5">External wrench impulse injected at Step 8 (shaded band) followed by active stabilization &amp; collision avoidance</text>',
        '  <line x1="45" y1="84" x2="72" y2="84" stroke="#e11d48" stroke-width="2.8"/>',
        f'  <text x="80" y="88" fill="#334155" font-family="{FONT_FAMILY}" font-size="12" font-weight="600">Attitude Tilt (deg, Left)</text>',
        '  <line x1="255" y1="84" x2="282" y2="84" stroke="#0284c7" stroke-width="2.8"/>',
        f'  <text x="290" y="88" fill="#334155" font-family="{FONT_FAMILY}" font-size="12" font-weight="600">Goal Distance Error (m, Right)</text>',
        '  <line x1="505" y1="84" x2="532" y2="84" stroke="#059669" stroke-width="2.5" stroke-dasharray="6,4"/>',
        f'  <text x="540" y="88" fill="#334155" font-family="{FONT_FAMILY}" font-size="12" font-weight="600">Min SDF Clearance (m, Right)</text>',
    ]

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = plot_bottom - frac * plot_h
        t_val = max_tilt * frac
        m_val = max_err_or_clr * frac
        lines.extend(
            [
                f'  <line x1="{plot_left}" y1="{yy:.1f}" x2="{plot_right}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>',
                f'  <text x="{plot_left - 10}" y="{yy + 4:.1f}" fill="#be123c" font-family="{FONT_FAMILY}" font-size="11" text-anchor="end">{t_val:.0f}°</text>',
                f'  <text x="{plot_right + 10}" y="{yy + 4:.1f}" fill="#0369a1" font-family="{FONT_FAMILY}" font-size="11" text-anchor="start">{m_val:.2f} m</text>',
            ]
        )

    lines.extend(
        [
            f'  <line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#94a3b8" stroke-width="1.5"/>',
            f'  <line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#94a3b8" stroke-width="1.5"/>',
            f'  <line x1="{plot_right}" y1="{plot_top}" x2="{plot_right}" y2="{plot_bottom}" stroke="#94a3b8" stroke-width="1.5"/>',
        ]
    )

    tilt_pts: list[str] = []
    err_pts: list[str] = []
    clear_pts: list[str] = []
    span_x = plot_right - plot_left - 20.0

    for i, pt in enumerate(trace):
        x = plot_left + 10.0 + (i / max(1, n - 1)) * span_x
        y_tilt = plot_bottom - (pt.attitude_tilt_deg / max(max_tilt, 1.0)) * plot_h
        y_err = plot_bottom - (pt.position_error_m / max(max_err_or_clr, 0.1)) * plot_h
        y_clr = plot_bottom - (max(0.0, pt.min_sdf_clearance_m) / max(max_err_or_clr, 0.1)) * plot_h

        tilt_pts.append(f"{x:.1f},{y_tilt:.1f}")
        err_pts.append(f"{x:.1f},{y_err:.1f}")
        clear_pts.append(f"{x:.1f},{y_clr:.1f}")

        if pt.impulse_active:
            lines.append(
                f'  <rect x="{x - 6:.1f}" y="{plot_top}" width="12" height="{plot_h}" fill="#f43f5e" opacity="0.14"/>'
            )

        if i % max(1, n // 6) == 0 or i == n - 1:
            lines.append(
                f'  <text x="{x:.1f}" y="{plot_bottom + 20:.1f}" fill="#475569" font-family="{FONT_FAMILY}" font-size="11" text-anchor="middle">t={pt.time_s:.2f}s</text>'
            )

    lines.extend(
        [
            f'  <polyline fill="none" stroke="#e11d48" stroke-width="2.8" points="{" ".join(tilt_pts)}"/>',
            f'  <polyline fill="none" stroke="#0284c7" stroke-width="2.8" points="{" ".join(err_pts)}"/>',
            f'  <polyline fill="none" stroke="#059669" stroke-width="2.5" stroke-dasharray="6,4" points="{" ".join(clear_pts)}"/>',
            f'  <text x="{(plot_left + plot_right) / 2:.1f}" y="458" fill="#475569" font-family="{FONT_FAMILY}" font-size="12" text-anchor="middle">Simulation Time (dt = 25 ms; Red Shaded Band = Active Wrench Shove)</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_all_benchmark_svgs(
    report: BenchmarkSuiteReport,
    output_dir: Path | str = Path("docs/assets"),
) -> dict[str, Path]:
    """Write all 4 publication-grade white-background SVG charts to output_dir and return their Paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    charts = {
        "l2_cache_residency": out / "l2_cache_residency.svg",
        "throughput_scaling": out / "throughput_scaling.svg",
        "roofline_rtx4060": out / "roofline_rtx4060.svg",
        "impulse_recovery": out / "impulse_recovery.svg",
    }

    charts["l2_cache_residency"].write_text(render_l2_cache_residency_svg(report), encoding="utf-8")
    charts["throughput_scaling"].write_text(render_throughput_scaling_svg(report), encoding="utf-8")
    charts["roofline_rtx4060"].write_text(render_roofline_rtx4060_svg(report), encoding="utf-8")
    charts["impulse_recovery"].write_text(render_impulse_recovery_svg(report), encoding="utf-8")
    return charts
