"""Cross-backend numerical parity tests across cuTile, PyTorch CUDA/CPU, Triton, and NumPy."""

from __future__ import annotations

import numpy as np
import pytest

from tilemppi.config import EngineConfig, MPPIConfig, SDFConfig
from tilemppi.controller.engine import MPPIEngine
from tilemppi.dynamics import create_robot_model
from tilemppi.hardware import detect_hardware
from tilemppi.kernels import (
    CuTileMPPIBackend,
    ReferenceCPUMPPIBackend,
    TorchMPPIBackend,
    TritonMPPIBackend,
    is_torch_available,
)
from tilemppi.scenarios import build_scenario
from tilemppi.types import BackendKind, PrecisionMode, RobotKind, ScenarioKind


@pytest.mark.parametrize("robot", [RobotKind.QUADROTOR, RobotKind.QUADRUPED, RobotKind.SWARM])
@pytest.mark.parametrize("precision", [PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.FP8])
def test_cross_backend_step_parity(
    robot: RobotKind,
    precision: PrecisionMode,
) -> None:
    mppi_cfg = MPPIConfig(
        num_rollouts=256,
        horizon=16,
        tile_k=64,
        num_vis_rollouts=32,
        seed=88,
    )
    sdf_cfg = SDFConfig(max_obstacles=8)
    scenario = build_scenario(ScenarioKind.OBSTACLE_SLALOM, robot=robot, max_obstacles=8)
    obs_table = scenario.pack_table(max_obstacles=8)

    model = create_robot_model(robot)
    x0 = model.default_initial_state(scenario.start_position)
    u_nom = np.repeat(model.nominal_control().reshape(1, -1), mppi_cfg.horizon, axis=0)
    goal = np.asarray(scenario.goal_position, dtype=np.float32)

    ref_backend = ReferenceCPUMPPIBackend(robot, mppi_cfg, sdf_cfg, precision)
    ct_backend = CuTileMPPIBackend(robot, mppi_cfg, sdf_cfg, precision)

    u_ref, c_ref, w_ref, vis_ref, opt_ref, ess_ref = ref_backend.execute_mppi_step(
        x0, u_nom, goal, goal, obs_table, step_seed=2026, temperature=1.2
    )
    u_ct, c_ct, w_ct, vis_ct, opt_ct, ess_ct = ct_backend.execute_mppi_step(
        x0, u_nom, goal, goal, obs_table, step_seed=2026, temperature=1.2
    )

    rtol = 2e-3 if precision == PrecisionMode.FP32 else 2.5e-2
    atol = 2e-3 if precision == PrecisionMode.FP32 else 2.5e-2

    assert np.allclose(c_ct, c_ref, rtol=rtol, atol=atol)
    assert np.allclose(w_ct, w_ref, rtol=rtol, atol=atol)
    assert np.allclose(u_ct, u_ref, rtol=rtol, atol=atol)
    assert np.allclose(vis_ct, vis_ref, rtol=rtol, atol=atol)
    assert np.allclose(opt_ct, opt_ref, rtol=rtol, atol=atol)
    assert abs(ess_ct - ess_ref) < (0.5 if precision == PrecisionMode.FP32 else 5.0)

    if is_torch_available():
        torch_cpu = TorchMPPIBackend(robot, mppi_cfg, sdf_cfg, precision, use_cuda=False)
        u_tc, c_tc, w_tc, vis_tc, opt_tc, _ = torch_cpu.execute_mppi_step(
            x0, u_nom, goal, goal, obs_table, step_seed=2026, temperature=1.2
        )
        assert np.allclose(c_tc, c_ref, rtol=rtol, atol=atol)
        assert np.allclose(w_tc, w_ref, rtol=rtol, atol=atol)
        assert np.allclose(u_tc, u_ref, rtol=rtol, atol=atol)
        assert np.allclose(vis_tc, vis_ref, rtol=rtol, atol=atol)
        assert np.allclose(opt_tc, opt_ref, rtol=rtol, atol=atol)

        caps = detect_hardware()
        if caps.torch_cuda_available:
            torch_gpu = TorchMPPIBackend(robot, mppi_cfg, sdf_cfg, precision, use_cuda=True)
            u_tg, c_tg, w_tg, vis_tg, opt_tg, _ = torch_gpu.execute_mppi_step(
                x0, u_nom, goal, goal, obs_table, step_seed=2026, temperature=1.2
            )
            assert np.allclose(c_tg, c_ref, rtol=rtol, atol=atol)
            assert np.allclose(w_tg, w_ref, rtol=rtol, atol=atol)
            assert np.allclose(u_tg, u_ref, rtol=rtol, atol=atol)
            assert np.allclose(vis_tg, vis_ref, rtol=rtol, atol=atol)
            assert np.allclose(opt_tg, opt_ref, rtol=rtol, atol=atol)

        triton_b = TritonMPPIBackend(robot, mppi_cfg, sdf_cfg, precision)
        u_tr, c_tr, _, _, _, _ = triton_b.execute_mppi_step(
            x0, u_nom, goal, goal, obs_table, step_seed=2026, temperature=1.2
        )
        assert np.allclose(c_tr, c_ref, rtol=rtol, atol=atol)
        assert np.allclose(u_tr, u_ref, rtol=rtol, atol=atol)


def test_unified_mppi_engine_all_backend_modes() -> None:
    for b_kind in (
        BackendKind.AUTO,
        BackendKind.CUTILE,
        BackendKind.CUTILE_SIM,
        BackendKind.TORCH_GPU,
        BackendKind.TORCH_CPU,
        BackendKind.TRITON,
        BackendKind.NUMPY_CPU,
    ):
        cfg = EngineConfig(
            robot=RobotKind.QUADROTOR,
            scenario=ScenarioKind.OBSTACLE_SLALOM,
            backend=b_kind,
            mppi=MPPIConfig(num_rollouts=128, horizon=8, tile_k=64, num_vis_rollouts=16),
            sdf=SDFConfig(max_obstacles=8),
        )
        engine = MPPIEngine(cfg)
        res = engine.step()
        assert res.state.shape == (16,)
        assert res.optimal_trajectory_xyz.shape == (8, 3)
        assert res.sampled_trajectories_xyz.shape == (16, 8, 3)
        assert res.telemetry.l2_residency_pct == 100.0
