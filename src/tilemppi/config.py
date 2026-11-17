"""Pydantic v2 configuration models for TileMPPI."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tilemppi.types import (
    BackendKind,
    PrecisionMode,
    RobotKind,
    ScenarioKind,
    is_power_of_two,
)


class SDFConfig(BaseModel):
    """3D Signed Distance Function and collision barrier configuration."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_obstacles: int = Field(default=32, ge=4, le=128)
    robot_radius: float = Field(default=0.28, gt=0.01, le=2.0)
    safety_margin: float = Field(default=0.28, ge=0.0, le=2.0)
    hard_barrier_weight: float = Field(default=1600.0, ge=0.0, le=100000.0)
    soft_barrier_weight: float = Field(default=45.0, ge=0.0, le=10000.0)
    soft_decay_alpha: float = Field(default=5.5, gt=0.1, le=50.0)
    smooth_min_k: float = Field(default=12.0, gt=0.5, le=100.0)
    arena_bounds: tuple[float, float, float] = Field(default=(8.0, 8.0, 5.0))

    @field_validator("max_obstacles")
    @classmethod
    def _validate_max_obstacles_pow2(cls, value: int) -> int:
        if not is_power_of_two(value):
            msg = f"max_obstacles must be a power of two, got {value}"
            raise ValueError(msg)
        return value


class MPPIConfig(BaseModel):
    """Model Predictive Path Integral (MPPI) rollout and tile kernel parameters."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    num_rollouts: int = Field(default=16384, ge=64, le=65536)
    horizon: int = Field(default=32, ge=8, le=128)
    tile_k: int = Field(default=128, ge=32, le=256)
    dt: float = Field(default=0.025, gt=0.002, le=0.2)
    temperature: float = Field(default=1.2, gt=0.01, le=100.0)
    adaptive_temperature: bool = Field(default=True)
    target_ess_ratio: float = Field(default=0.08, gt=0.005, le=0.8)
    ou_rho: float = Field(default=0.65, ge=0.0, lt=0.999)
    noise_sigma: float = Field(default=0.45, gt=0.01, le=5.0)
    control_weight: float = Field(default=0.15, ge=0.0, le=50.0)
    smoothness_weight: float = Field(default=0.35, ge=0.0, le=50.0)
    cvar_alpha: float = Field(default=0.15, ge=0.0, le=0.5)
    cvar_weight: float = Field(default=2.0, ge=0.0, le=50.0)
    savgol_window: int = Field(default=5, ge=0, le=15)
    savgol_polyorder: int = Field(default=2, ge=1, le=5)
    num_vis_rollouts: int = Field(default=128, ge=8, le=512)
    inline_noise: bool = Field(default=False)
    seed: int = Field(default=42, ge=0)

    @field_validator("num_rollouts", "horizon", "tile_k")
    @classmethod
    def _validate_pow2_fields(cls, value: int, info: object) -> int:
        if not is_power_of_two(value):
            field_name = getattr(info, "field_name", "value")
            msg = f"{field_name} must be a power of two for cuTile kernels, got {value}"
            raise ValueError(msg)
        return value

    @field_validator("savgol_window")
    @classmethod
    def _validate_savgol_window(cls, value: int) -> int:
        if value != 0 and (value < 3 or value % 2 == 0):
            msg = f"savgol_window must be 0 (disabled) or an odd integer >= 3, got {value}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _validate_tile_and_vis_bounds(self) -> MPPIConfig:
        if self.tile_k > self.num_rollouts:
            msg = f"tile_k ({self.tile_k}) cannot exceed num_rollouts ({self.num_rollouts})"
            raise ValueError(msg)
        if self.savgol_window > 0 and self.savgol_polyorder >= self.savgol_window:
            msg = (
                f"savgol_polyorder ({self.savgol_polyorder}) must be less than "
                f"savgol_window ({self.savgol_window})"
            )
            raise ValueError(msg)
        if self.num_vis_rollouts > self.num_rollouts:
            self.num_vis_rollouts = self.num_rollouts
        return self


class PlannerConfig(BaseModel):
    """Global 3D Voxel A* waypoint planner configuration."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = Field(default=True)
    grid_resolution: float = Field(default=0.4, gt=0.1, le=2.0)
    inflation_radius: float = Field(default=0.45, ge=0.0, le=2.0)
    lookahead_distance: float = Field(default=2.2, gt=0.2, le=10.0)
    replan_interval_steps: int = Field(default=6, ge=1, le=100)


class ServerConfig(BaseModel):
    """FastAPI and WebSocket streaming server configuration."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)
    target_fps: int = Field(default=60, ge=5, le=240)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class EngineConfig(BaseModel):
    """Top-level unified configuration for the TileMPPI control engine."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    robot: RobotKind = Field(default=RobotKind.QUADROTOR)
    scenario: ScenarioKind = Field(default=ScenarioKind.OBSTACLE_SLALOM)
    backend: BackendKind = Field(default=BackendKind.AUTO)
    precision: PrecisionMode = Field(default=PrecisionMode.FP32)
    mppi: MPPIConfig = Field(default_factory=MPPIConfig)
    sdf: SDFConfig = Field(default_factory=SDFConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
