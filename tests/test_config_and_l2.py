"""Tests for Pydantic v2 configurations, hardware probing, and 32 MB L2 cache budget planner."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tilemppi.config import EngineConfig, MPPIConfig, SDFConfig
from tilemppi.hardware import (
    ADA_RTX4060_L2_BYTES,
    BackendSelector,
    HardwareCapabilities,
    detect_hardware,
)
from tilemppi.l2_budget import (
    compute_engine_l2_budget,
    compute_l2_budget,
    recommend_tile_k,
)
from tilemppi.types import (
    BackendKind,
    ImpulseState,
    PrecisionMode,
    RobotKind,
    ScenarioKind,
    is_power_of_two,
    next_power_of_two,
)


def test_power_of_two_helpers() -> None:
    assert is_power_of_two(1)
    assert is_power_of_two(64)
    assert is_power_of_two(16384)
    assert not is_power_of_two(0)
    assert not is_power_of_two(-8)
    assert not is_power_of_two(100)
    assert next_power_of_two(0) == 1
    assert next_power_of_two(1) == 1
    assert next_power_of_two(100) == 128
    assert next_power_of_two(128) == 128


def test_mppi_config_power_of_two_validation() -> None:
    cfg = MPPIConfig(num_rollouts=16384, horizon=32, tile_k=128)
    assert cfg.num_rollouts == 16384
    assert cfg.horizon == 32
    assert cfg.tile_k == 128

    with pytest.raises(ValidationError, match="power of two"):
        MPPIConfig(num_rollouts=1000)

    with pytest.raises(ValidationError, match="power of two"):
        MPPIConfig(horizon=30)

    with pytest.raises(ValidationError, match="power of two"):
        MPPIConfig(tile_k=96)

    with pytest.raises(ValidationError, match="cannot exceed num_rollouts"):
        MPPIConfig(num_rollouts=64, tile_k=128)


def test_mppi_savgol_validation() -> None:
    cfg_disabled = MPPIConfig(savgol_window=0)
    assert cfg_disabled.savgol_window == 0

    with pytest.raises(ValidationError, match="odd integer"):
        MPPIConfig(savgol_window=4)

    with pytest.raises(ValidationError, match="savgol_polyorder"):
        MPPIConfig(savgol_window=3, savgol_polyorder=3)


def test_sdf_config_validation() -> None:
    sdf = SDFConfig(max_obstacles=32)
    assert sdf.max_obstacles == 32

    with pytest.raises(ValidationError, match="power of two"):
        SDFConfig(max_obstacles=24)


def test_enums_and_impulse_state() -> None:
    assert RobotKind.QUADROTOR.state_dim == 16
    assert RobotKind.QUADROTOR.control_dim == 4
    assert RobotKind.QUADRUPED.state_dim == 32
    assert RobotKind.QUADRUPED.control_dim == 16
    assert RobotKind.SWARM.state_dim == 32
    assert RobotKind.SWARM.control_dim == 16

    assert PrecisionMode.FP32.bytes_per_element == 4
    assert PrecisionMode.FP16.bytes_per_element == 2
    assert PrecisionMode.FP8.bytes_per_element == 1

    for b in BackendKind:
        assert isinstance(b.numeric_id, int)

    imp = ImpulseState()
    assert not imp.is_active()
    assert np.allclose(imp.to_vector6(), np.zeros(6, dtype=np.float32))

    imp_active = ImpulseState(
        force=np.array([10.0, -5.0, 2.0], dtype=np.float32),
        torque=np.array([0.5, 0.0, -0.5], dtype=np.float32),
        remaining_steps=4,
    )
    assert imp_active.is_active()
    vec = imp_active.to_vector6()
    assert vec.shape == (6,)
    assert np.allclose(vec[:3], [10.0, -5.0, 2.0])


@pytest.mark.parametrize(
    "precision",
    [PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.FP8],
)
@pytest.mark.parametrize(
    "robot",
    [RobotKind.QUADROTOR, RobotKind.QUADRUPED, RobotKind.SWARM],
)
def test_l2_residency_k16384_fits_in_32mb(
    precision: PrecisionMode,
    robot: RobotKind,
) -> None:
    cfg = EngineConfig(
        robot=robot,
        scenario=ScenarioKind.OBSTACLE_SLALOM,
        precision=precision,
        mppi=MPPIConfig(num_rollouts=16384, horizon=32, tile_k=128),
        sdf=SDFConfig(max_obstacles=32),
    )
    report = compute_engine_l2_budget(cfg)

    # Fused mode must strictly stay under the 32 MB L2 cache limit
    assert report.fused_working_set_bytes < ADA_RTX4060_L2_BYTES
    assert report.fused_working_set_mb < 32.0
    assert report.fits_in_l2 is True
    assert report.l2_residency_pct == 100.0

    # Unfused PyTorch rollout must exceed 100 MB (and ~671 MB for 32-DoF)
    assert report.unfused_working_set_bytes > 100 * 1024 * 1024
    assert report.unfused_working_set_mb > 100.0
    assert report.traffic_reduction_ratio > 20.0
    assert report.fused_arithmetic_intensity > report.unfused_arithmetic_intensity
    assert report.sram_per_block_bytes < 96 * 1024

    serialized = report.to_dict()
    assert serialized["precision"] == precision.value
    assert serialized["fits_in_l2"] is True


def test_l2_budget_inline_noise_under_1mb() -> None:
    report = compute_l2_budget(
        num_rollouts=16384,
        horizon=32,
        state_dim=32,
        control_dim=16,
        max_obstacles=32,
        precision=PrecisionMode.FP32,
        inline_noise=True,
    )
    assert report.fused_working_set_mb < 0.5
    assert report.traffic_reduction_ratio > 500.0
    assert recommend_tile_k(32, 16, 4) == 32


def test_hardware_detection_and_backend_selector() -> None:
    caps = detect_hardware(force_refresh=True)
    assert isinstance(caps, HardwareCapabilities)
    assert caps.cpu_cores_logical >= 1
    assert caps.l2_cache_bytes > 0
    assert BackendKind.CUTILE_SIM in caps.available_backends
    assert BackendKind.NUMPY_CPU in caps.available_backends

    info_dict = caps.to_dict()
    assert "cpu_model" in info_dict
    assert "gpu_name" in info_dict

    assert BackendSelector.resolve(BackendKind.NUMPY_CPU, caps) == BackendKind.NUMPY_CPU
    assert BackendSelector.resolve(BackendKind.CUTILE_SIM, caps) == BackendKind.CUTILE_SIM
    resolved_auto = BackendSelector.resolve(BackendKind.AUTO, caps)
    assert resolved_auto in caps.available_backends

    # Selecting CUTILE resolves to CUTILE on sm_100+ or CUTILE_SIM on sm_89/CPU
    resolved_cutile = BackendSelector.resolve(BackendKind.CUTILE, caps)
    assert resolved_cutile in (BackendKind.CUTILE, BackendKind.CUTILE_SIM)
