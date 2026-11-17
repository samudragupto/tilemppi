"""FastAPI REST + Binary 'TMPI' WebSocket control server and static 3D playground host."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tilemppi import __version__
from tilemppi.config import EngineConfig
from tilemppi.hardware import detect_hardware
from tilemppi.l2_budget import compute_engine_l2_budget
from tilemppi.planner import HeterogeneousControlRuntime
from tilemppi.scenarios import list_scenarios
from tilemppi.server.protocol import encode_tmpi_frame
from tilemppi.types import (
    BackendKind,
    ObstacleKind,
    PrecisionMode,
    RobotKind,
    ScenarioKind,
    is_power_of_two,
)


class ConfigUpdateRequest(BaseModel):
    """Partial or full runtime configuration update payload."""

    model_config = ConfigDict(extra="forbid")

    robot: RobotKind | None = None
    scenario: ScenarioKind | None = None
    backend: BackendKind | None = None
    precision: PrecisionMode | None = None
    num_rollouts: int | None = Field(default=None, ge=64, le=65536)
    horizon: int | None = Field(default=None, ge=8, le=128)
    tile_k: int | None = Field(default=None, ge=32, le=256)
    temperature: float | None = Field(default=None, gt=0.01, le=100.0)
    noise_sigma: float | None = Field(default=None, gt=0.01, le=5.0)
    planner_enabled: bool | None = None

    @field_validator("num_rollouts", "horizon", "tile_k")
    @classmethod
    def _validate_pow2_opt(cls, value: int | None) -> int | None:
        if value is not None and not is_power_of_two(value):
            msg = f"Value must be a power of two, got {value}"
            raise ValueError(msg)
        return value


class ImpulseRequest(BaseModel):
    """External force/torque impulse ('mouse shove') request payload."""

    model_config = ConfigDict(extra="forbid")

    force: tuple[float, float, float] = Field(default=(0.0, 12.0, 3.0))
    torque: tuple[float, float, float] = Field(default=(0.2, -0.2, 0.1))
    duration_steps: int = Field(default=4, ge=1, le=50)


class ThrowObstacleRequest(BaseModel):
    """Dynamic obstacle launch request payload."""

    model_config = ConfigDict(extra="forbid")

    position: tuple[float, float, float] = Field(default=(0.0, 1.5, 1.5))
    velocity: tuple[float, float, float] = Field(default=(0.0, -2.0, 0.0))
    radius: float = Field(default=0.42, gt=0.05, le=2.0)
    kind: ObstacleKind = Field(default=ObstacleKind.SPHERE)


class GoalUpdateRequest(BaseModel):
    """3D goal update request payload."""

    model_config = ConfigDict(extra="forbid")

    goal: tuple[float, float, float]


def _find_frontend_dist() -> Path | None:
    """Locate frontend/dist directory relative to repo root if built."""
    candidates = [
        Path(__file__).resolve().parents[3] / "frontend" / "dist",
        Path.cwd() / "frontend" / "dist",
    ]
    for path in candidates:
        if (path / "index.html").is_file():
            return path
    return None


def create_app(
    initial_config: EngineConfig | None = None,
    frontend_dist_dir: Path | None = None,
) -> FastAPI:
    """Create and configure the TileMPPI FastAPI application."""
    cfg = initial_config or EngineConfig()
    runtime = HeterogeneousControlRuntime(config=cfg)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        runtime.get_latest_snapshot()
        yield

    app = FastAPI(
        title="TileMPPI Control Engine API",
        version=__version__,
        description="16,384-Parallel-Universe Real-Time Robotics Control Engine in NVIDIA cuTile",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def get_health() -> dict[str, Any]:
        snap = runtime.get_latest_snapshot()
        return {
            "status": "ok",
            "version": __version__,
            "backend": snap.telemetry.backend.value,
            "robot": snap.telemetry.robot.value,
            "scenario": snap.telemetry.scenario.value,
            "l2_residency_pct": snap.telemetry.l2_residency_pct,
        }

    @app.get("/api/hardware")
    async def get_hardware() -> dict[str, Any]:
        caps = detect_hardware()
        l2_rep = compute_engine_l2_budget(runtime.config, l2_capacity_bytes=caps.l2_cache_bytes)
        return {
            "hardware": caps.to_dict(),
            "l2_budget": l2_rep.to_dict(),
        }

    @app.get("/api/scenarios")
    async def get_scenarios() -> dict[str, Any]:
        return {
            "robot": runtime.config.robot.value,
            "scenarios": list_scenarios(runtime.config.robot),
        }

    @app.get("/api/state")
    async def get_state() -> dict[str, Any]:
        snap = runtime.get_latest_snapshot()
        tel_dict = asdict(snap.telemetry)
        tel_dict["backend"] = snap.telemetry.backend.value
        tel_dict["precision"] = snap.telemetry.precision.value
        tel_dict["robot"] = snap.telemetry.robot.value
        tel_dict["scenario"] = snap.telemetry.scenario.value
        return {
            "state": snap.state.tolist(),
            "goal": snap.goal.tolist(),
            "active_waypoint": snap.active_waypoint.tolist(),
            "global_path": [wp.tolist() for wp in runtime.get_global_path()],
            "telemetry": tel_dict,
            "config": runtime.config.model_dump(mode="json"),
        }

    @app.post("/api/config")
    async def update_config(req: ConfigUpdateRequest) -> dict[str, Any]:
        curr = runtime.config.model_copy(deep=True)
        if req.robot is not None:
            curr.robot = req.robot
        if req.scenario is not None:
            curr.scenario = req.scenario
        if req.backend is not None:
            curr.backend = req.backend
        if req.precision is not None:
            curr.precision = req.precision
        if req.num_rollouts is not None:
            curr.mppi.num_rollouts = req.num_rollouts
            if curr.mppi.tile_k > req.num_rollouts:
                curr.mppi.tile_k = req.num_rollouts
        if req.horizon is not None:
            curr.mppi.horizon = req.horizon
        if req.tile_k is not None:
            curr.mppi.tile_k = min(req.tile_k, curr.mppi.num_rollouts)
        if req.temperature is not None:
            curr.mppi.temperature = req.temperature
        if req.noise_sigma is not None:
            curr.mppi.noise_sigma = req.noise_sigma
        if req.planner_enabled is not None:
            curr.planner.enabled = req.planner_enabled

        snap = runtime.reset(curr)
        return {
            "status": "updated",
            "active_backend": snap.telemetry.backend.value,
            "config": runtime.config.model_dump(mode="json"),
        }

    @app.post("/api/impulse")
    async def post_impulse(req: ImpulseRequest) -> dict[str, Any]:
        runtime.apply_impulse(
            force=req.force,
            torque=req.torque,
            duration_steps=req.duration_steps,
        )
        snap = runtime.step()
        return {
            "status": "impulse_applied",
            "impulse_vec": snap.impulse_vec.tolist(),
            "step_index": snap.telemetry.step_index,
        }

    @app.post("/api/obstacles/throw")
    async def post_throw_obstacle(req: ThrowObstacleRequest) -> dict[str, Any]:
        runtime.throw_obstacle(
            position=req.position,
            velocity=req.velocity,
            radius=req.radius,
            kind=req.kind,
        )
        snap = runtime.step()
        return {
            "status": "obstacle_thrown",
            "step_index": snap.telemetry.step_index,
        }

    @app.post("/api/goal")
    async def post_goal(req: GoalUpdateRequest) -> dict[str, Any]:
        runtime.set_goal(req.goal)
        snap = runtime.step()
        return {
            "status": "goal_updated",
            "goal": snap.goal.tolist(),
            "active_waypoint": snap.active_waypoint.tolist(),
        }

    @app.post("/api/reset")
    async def post_reset() -> dict[str, Any]:
        snap = runtime.reset()
        return {
            "status": "reset",
            "step_index": snap.telemetry.step_index,
            "state": snap.state.tolist(),
        }

    @app.post("/api/benchmark/quick")
    async def post_quick_benchmark() -> dict[str, Any]:
        latencies: list[float] = []
        for _ in range(4):
            snap = runtime.step()
            latencies.append(snap.telemetry.step_ms)
        p50_ms = float(np.median(latencies))
        hz = 1000.0 / max(p50_ms, 0.05)
        return {
            "status": "ok",
            "backend": snap.telemetry.backend.value,
            "robot": snap.telemetry.robot.value,
            "precision": snap.telemetry.precision.value,
            "num_rollouts": snap.telemetry.rollout_count,
            "horizon": snap.telemetry.horizon,
            "p50_step_ms": round(p50_ms, 3),
            "control_hz": round(hz, 2),
            "l2_residency_pct": snap.telemetry.l2_residency_pct,
        }

    @app.websocket("/ws")
    async def websocket_telemetry(websocket: WebSocket) -> None:
        await websocket.accept()
        paused = False
        target_dt = 1.0 / max(5, runtime.config.server.target_fps)

        # Send immediate initial binary frame on connection
        initial_snap = runtime.get_latest_snapshot()
        await websocket.send_bytes(encode_tmpi_frame(initial_snap))

        try:
            while True:
                # Non-blocking check for incoming client JSON control messages
                try:
                    raw_msg = await asyncio.wait_for(websocket.receive_text(), timeout=target_dt)
                    cmd = json.loads(raw_msg)
                    msg_type = cmd.get("type", "")

                    if msg_type == "pause":
                        paused = bool(cmd.get("paused", not paused))
                    elif msg_type == "impulse":
                        runtime.apply_impulse(
                            force=tuple(cmd.get("force", [0.0, 12.0, 3.0])),
                            torque=tuple(cmd.get("torque", [0.2, -0.2, 0.1])),
                            duration_steps=int(cmd.get("duration_steps", 4)),
                        )
                    elif msg_type == "throw":
                        runtime.throw_obstacle(
                            position=tuple(cmd.get("position", [0.0, 1.5, 1.5])),
                            velocity=tuple(cmd.get("velocity", [0.0, -2.0, 0.0])),
                            radius=float(cmd.get("radius", 0.42)),
                        )
                    elif msg_type == "goal":
                        runtime.set_goal(tuple(cmd.get("goal", [2.5, 0.0, 1.5])))
                    elif msg_type == "reset":
                        runtime.reset()
                    elif msg_type == "step":
                        snap = runtime.step()
                        await websocket.send_bytes(encode_tmpi_frame(snap))
                        continue
                except TimeoutError:  # nosec B110
                    pass

                if not paused:
                    snap = runtime.step()
                    await websocket.send_bytes(encode_tmpi_frame(snap))
        except WebSocketDisconnect:
            return
        except Exception:
            return

    dist_dir = frontend_dist_dir or _find_frontend_dist()
    if dist_dir is not None and (dist_dir / "index.html").is_file():
        assets_dir = dist_dir / "assets"
        if assets_dir.is_dir():
            app.mount(
                "/assets",
                StaticFiles(directory=str(assets_dir)),
                name="frontend_assets",
            )

        @app.get("/", include_in_schema=False)
        async def serve_index() -> FileResponse:
            return FileResponse(dist_dir / "index.html")
    else:

        @app.get("/", include_in_schema=False)
        async def serve_fallback_root() -> JSONResponse:
            return JSONResponse(
                {
                    "app": "TileMPPI",
                    "version": __version__,
                    "message": "TileMPPI API server is running. Build frontend/dist for 3D UI.",
                    "docs": "/docs",
                }
            )

    return app
