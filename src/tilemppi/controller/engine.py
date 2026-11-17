"""Unified MPPIEngine integrating cuTile, Triton, PyTorch CUDA, and CPU reference backends."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from tilemppi.config import EngineConfig
from tilemppi.dynamics import create_robot_model
from tilemppi.dynamics.base import RobotModel
from tilemppi.hardware import BackendSelector, detect_hardware
from tilemppi.kernels import (
    CuTileMPPIBackend,
    ReferenceCPUMPPIBackend,
    TorchMPPIBackend,
    TritonMPPIBackend,
)
from tilemppi.l2_budget import compute_engine_l2_budget
from tilemppi.scenarios import ScenarioDefinition, build_scenario
from tilemppi.scenarios.presets import insert_thrown_obstacle
from tilemppi.sdf.primitives import evaluate_sdf_batch, step_dynamic_obstacles_in_place
from tilemppi.types import (
    BackendKind,
    Float32Array,
    ImpulseState,
    ObstacleKind,
    StepResult,
    TelemetrySnapshot,
)


def _query_vram_mb(active_backend: BackendKind) -> float:
    """Query active CUDA VRAM allocation in MB if using a GPU backend."""
    if active_backend in (BackendKind.TORCH_GPU, BackendKind.TRITON, BackendKind.CUTILE):
        try:
            import torch

            if torch.cuda.is_available():
                reserved = float(torch.cuda.memory_reserved(0)) / (1024.0 * 1024.0)
                allocated = float(torch.cuda.memory_allocated(0)) / (1024.0 * 1024.0)
                return round(max(reserved, allocated, 12.5), 2)
        except Exception:  # nosec B110
            pass
    return 0.0


class MPPIEngine:
    """Unified real-time Model Predictive Path Integral (MPPI) control & simulation engine."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.capabilities = detect_hardware()
        self.active_backend: BackendKind = BackendSelector.resolve(
            self.config.backend, self.capabilities
        )
        self.model: RobotModel = create_robot_model(self.config.robot)
        self.scenario_def: ScenarioDefinition = build_scenario(
            self.config.scenario,
            robot=self.config.robot,
            max_obstacles=self.config.sdf.max_obstacles,
        )
        self.l2_report = compute_engine_l2_budget(
            self.config, l2_capacity_bytes=self.capabilities.l2_cache_bytes
        )
        self._backend_impl: Any = self._instantiate_backend()

        self.step_index: int = 0
        self.sim_time: float = 0.0
        self.temperature: float = float(self.config.mppi.temperature)
        self.state: Float32Array = self.model.default_initial_state(
            self.scenario_def.start_position
        )
        self.goal: Float32Array = np.asarray(self.scenario_def.goal_position, dtype=np.float32)
        self.active_waypoint: Float32Array = self.goal.copy()
        self.obstacles_table: Float32Array = self.scenario_def.pack_table(
            self.config.sdf.max_obstacles
        )
        self.nominal_controls: Float32Array = np.repeat(
            self.model.nominal_control().reshape(1, -1),
            self.config.mppi.horizon,
            axis=0,
        )
        self.impulse = ImpulseState()
        self._throw_cursor: int = 0
        self._step_ms_history: deque[float] = deque(maxlen=32)
        self.last_result: StepResult | None = None

    def _instantiate_backend(self) -> Any:
        b = self.active_backend
        r = self.config.robot
        m_cfg = self.config.mppi
        s_cfg = self.config.sdf
        prec = self.config.precision

        if b == BackendKind.CUTILE:
            return CuTileMPPIBackend(r, m_cfg, s_cfg, prec, prefer_native=True)
        if b == BackendKind.CUTILE_SIM:
            return CuTileMPPIBackend(r, m_cfg, s_cfg, prec, prefer_native=False)
        if b == BackendKind.TRITON:
            return TritonMPPIBackend(r, m_cfg, s_cfg, prec)
        if b == BackendKind.TORCH_GPU:
            return TorchMPPIBackend(r, m_cfg, s_cfg, prec, use_cuda=True)
        if b == BackendKind.TORCH_CPU:
            return TorchMPPIBackend(r, m_cfg, s_cfg, prec, use_cuda=False)
        return ReferenceCPUMPPIBackend(r, m_cfg, s_cfg, prec)

    def reset(self, config: EngineConfig | None = None) -> StepResult:
        """Reset simulation state, scenario obstacles, and nominal control sequence."""
        if config is not None:
            self.config = config
            self.active_backend = BackendSelector.resolve(self.config.backend, self.capabilities)
            self.model = create_robot_model(self.config.robot)
            self.scenario_def = build_scenario(
                self.config.scenario,
                robot=self.config.robot,
                max_obstacles=self.config.sdf.max_obstacles,
            )
            self.l2_report = compute_engine_l2_budget(
                self.config, l2_capacity_bytes=self.capabilities.l2_cache_bytes
            )
            self._backend_impl = self._instantiate_backend()

        self.step_index = 0
        self.sim_time = 0.0
        self.temperature = float(self.config.mppi.temperature)
        self.state = self.model.default_initial_state(self.scenario_def.start_position)
        self.goal = np.asarray(self.scenario_def.goal_position, dtype=np.float32)
        self.active_waypoint = self.goal.copy()
        self.obstacles_table = self.scenario_def.pack_table(self.config.sdf.max_obstacles)
        self.nominal_controls = np.repeat(
            self.model.nominal_control().reshape(1, -1),
            self.config.mppi.horizon,
            axis=0,
        )
        self.impulse = ImpulseState()
        self._step_ms_history.clear()
        return self.step()

    def set_goal(self, goal_xyz: Sequence[float] | Float32Array) -> None:
        """Update 3D target goal position."""
        g = np.asarray(goal_xyz, dtype=np.float32).reshape(3)
        self.goal = g.copy()
        self.active_waypoint = g.copy()

    def set_active_waypoint(self, waypoint_xyz: Sequence[float] | Float32Array) -> None:
        """Update intermediate lookahead waypoint (e.g., from global 3D A* planner)."""
        self.active_waypoint = np.asarray(waypoint_xyz, dtype=np.float32).reshape(3).copy()

    def apply_impulse(
        self,
        force: Sequence[float] = (0.0, 12.0, 3.0),
        torque: Sequence[float] = (0.2, -0.2, 0.1),
        duration_steps: int = 4,
    ) -> None:
        """Trigger an external wrench disturbance ('mouse shove') on the robot."""
        self.impulse = ImpulseState(
            force=np.asarray(force, dtype=np.float32).reshape(3),
            torque=np.asarray(torque, dtype=np.float32).reshape(3),
            remaining_steps=max(1, int(duration_steps)),
        )

    def throw_obstacle(
        self,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
        radius: float = 0.42,
        kind: ObstacleKind = ObstacleKind.SPHERE,
    ) -> None:
        """Launch a dynamic obstacle into the live 3D arena."""
        self._throw_cursor = insert_thrown_obstacle(
            table=self.obstacles_table,
            position=position,
            velocity=velocity,
            radius=radius,
            kind=kind,
            slot_cursor=self._throw_cursor,
        )

    def _update_adaptive_temperature(self, ess: float) -> None:
        if not self.config.mppi.adaptive_temperature:
            return
        ess_ratio = ess / float(self.config.mppi.num_rollouts)
        target = float(self.config.mppi.target_ess_ratio)
        if ess_ratio < 0.5 * target:
            self.temperature = min(25.0, self.temperature * 1.08)
        elif ess_ratio > 1.8 * target:
            self.temperature = max(0.08, self.temperature * 0.94)

    def step(self) -> StepResult:
        """Execute one closed-loop MPPI control and physics simulation step."""
        if (
            self.scenario_def.scheduled_impulse_step is not None
            and self.step_index == self.scenario_def.scheduled_impulse_step
        ):
            self.apply_impulse(
                force=self.scenario_def.scheduled_impulse_force,
                torque=self.scenario_def.scheduled_impulse_torque,
                duration_steps=self.scenario_def.scheduled_impulse_duration,
            )

        step_seed = int(self.config.mppi.seed + self.step_index * 7919)
        t0 = time.perf_counter()

        updated_u, costs, weights, vis_xyz, opt_xyz, ess = self._backend_impl.execute_mppi_step(
            state=self.state,
            nominal_controls=self.nominal_controls,
            waypoint=self.active_waypoint,
            goal=self.goal,
            obstacles_table=self.obstacles_table,
            step_seed=step_seed,
            temperature=self.temperature,
        )

        elapsed_ms = max(0.05, (time.perf_counter() - t0) * 1000.0)
        self._step_ms_history.append(elapsed_ms)
        avg_ms = float(np.mean(self._step_ms_history))
        loop_hz = 1000.0 / max(avg_ms, 0.05)

        self._update_adaptive_temperature(ess)

        u_opt = updated_u[0].copy()
        impulse_vec = self.impulse.to_vector6()
        wrench = impulse_vec if self.impulse.is_active() else None

        dt = float(self.config.mppi.dt)
        self.state = self.model.step_rk4(
            self.state.reshape(1, -1),
            u_opt.reshape(1, -1),
            dt=dt,
            external_wrench=wrench,
        )[0]

        if self.impulse.remaining_steps > 0:
            self.impulse.remaining_steps -= 1

        step_dynamic_obstacles_in_place(
            self.obstacles_table,
            dt=dt,
            arena_bounds=self.config.sdf.arena_bounds,
        )

        # Warm-start shift nominal control sequence
        self.nominal_controls[:-1] = updated_u[1:]
        self.nominal_controls[-1] = updated_u[-1]

        robot_pos = self.model.extract_position(self.state.reshape(1, -1))
        min_clearance = float(
            evaluate_sdf_batch(robot_pos, self.obstacles_table, t=0.0)[0]
            - self.config.sdf.robot_radius
        )

        n_vis = vis_xyz.shape[0]
        sampled_w = weights[:n_vis].copy()
        w_sum = float(np.sum(sampled_w))
        if w_sum > 1e-9:
            sampled_w = (sampled_w / np.float32(w_sum)).astype(np.float32)

        total_state_steps = float(self.config.mppi.num_rollouts * self.config.mppi.horizon)
        m_steps_per_sec = (total_state_steps / (elapsed_ms / 1000.0)) / 1.0e6

        telemetry = TelemetrySnapshot(
            step_index=self.step_index,
            sim_time=round(self.sim_time, 4),
            step_ms=round(elapsed_ms, 3),
            loop_hz=round(loop_hz, 2),
            million_state_steps_per_sec=round(m_steps_per_sec, 3),
            ess=round(ess, 2),
            ess_ratio=round(ess / float(self.config.mppi.num_rollouts), 4),
            temperature=round(self.temperature, 4),
            min_cost=round(float(np.min(costs)), 3),
            mean_cost=round(float(np.mean(costs)), 3),
            min_sdf_clearance=round(min_clearance, 4),
            l2_working_set_mb=self.l2_report.fused_working_set_mb,
            l2_residency_pct=self.l2_report.l2_residency_pct,
            vram_mb=_query_vram_mb(self.active_backend),
            rollout_count=self.config.mppi.num_rollouts,
            horizon=self.config.mppi.horizon,
            backend=self.active_backend,
            precision=self.config.precision,
            robot=self.config.robot,
            scenario=self.config.scenario,
        )

        poses = self.model.compute_articulated_world_poses(self.state)
        result = StepResult(
            state=self.state.copy(),
            optimal_control=u_opt,
            nominal_sequence=self.nominal_controls.copy(),
            optimal_trajectory_xyz=opt_xyz,
            sampled_trajectories_xyz=vis_xyz,
            sampled_weights=sampled_w,
            articulated_poses=poses,
            active_waypoint=self.active_waypoint.copy(),
            goal=self.goal.copy(),
            obstacles_table=self.obstacles_table.copy(),
            impulse_vec=impulse_vec,
            costs=costs,
            telemetry=telemetry,
        )
        self.step_index += 1
        self.sim_time += dt
        self.last_result = result
        return result
