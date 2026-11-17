"""Tests for the binary 'TMPI' protocol and FastAPI REST + WebSocket server."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tilemppi.config import EngineConfig, MPPIConfig, SDFConfig
from tilemppi.controller.engine import MPPIEngine
from tilemppi.server import (
    TMPI_HEADER_BYTES,
    TMPI_MAGIC,
    create_app,
    decode_tmpi_frame,
    encode_tmpi_frame,
)
from tilemppi.types import RobotKind, ScenarioKind


@pytest.mark.parametrize("robot", [RobotKind.QUADROTOR, RobotKind.QUADRUPED, RobotKind.SWARM])
def test_tmpi_binary_protocol_roundtrip(robot: RobotKind) -> None:
    cfg = EngineConfig(
        robot=robot,
        scenario=ScenarioKind.OBSTACLE_SLALOM,
        mppi=MPPIConfig(num_rollouts=128, horizon=16, tile_k=64, num_vis_rollouts=16),
        sdf=SDFConfig(max_obstacles=8),
    )
    engine = MPPIEngine(cfg)
    engine.apply_impulse(force=(4.0, -2.0, 1.0), torque=(0.1, 0.2, -0.1), duration_steps=3)
    res = engine.step()

    frame_bytes = encode_tmpi_frame(res)
    assert len(frame_bytes) > TMPI_HEADER_BYTES

    decoded = decode_tmpi_frame(frame_bytes)
    assert decoded.version == 1
    assert decoded.robot_id == robot.numeric_id
    assert decoded.frame_seq == res.telemetry.step_index
    assert decoded.num_vis_rollouts == 16
    assert decoded.horizon == 16
    assert np.allclose(decoded.articulated_poses, res.articulated_poses)
    assert np.allclose(decoded.goal, res.goal)
    assert np.allclose(decoded.active_waypoint, res.active_waypoint)
    assert np.allclose(decoded.optimal_trajectory_xyz, res.optimal_trajectory_xyz)
    assert np.allclose(decoded.sampled_trajectories_xyz, res.sampled_trajectories_xyz)
    assert np.allclose(decoded.sampled_weights, res.sampled_weights)
    assert np.allclose(decoded.impulse_vec, res.impulse_vec)
    assert decoded.telemetry_vec.shape == (8,)


def test_tmpi_protocol_rejects_corrupt_or_truncated_frames() -> None:
    with pytest.raises(ValueError, match="Truncated"):
        decode_tmpi_frame(b"short")

    bad_magic = struct.pack(
        "<IHHIfHHHHff",
        0xDEADBEEF,
        1,
        0,
        0,
        0.0,
        8,
        8,
        0,
        24,
        1.0,
        0.1,
    )
    with pytest.raises(ValueError, match="Invalid TMPI magic"):
        decode_tmpi_frame(bad_magic)

    bad_version = struct.pack(
        "<IHHIfHHHHff",
        TMPI_MAGIC,
        99,
        0,
        0,
        0.0,
        8,
        8,
        0,
        24,
        1.0,
        0.1,
    )
    with pytest.raises(ValueError, match="Unsupported TMPI protocol version"):
        decode_tmpi_frame(bad_version)

    valid_header_truncated_body = struct.pack(
        "<IHHIfHHHHff",
        TMPI_MAGIC,
        1,
        0,
        0,
        0.0,
        8,
        8,
        0,
        24,
        1.0,
        0.1,
    )
    with pytest.raises(ValueError, match="payload byte mismatch"):
        decode_tmpi_frame(valid_header_truncated_body + b"\x00\x00\x00\x00")


def test_fastapi_rest_and_websocket_endpoints(tmp_path: Path) -> None:
    # Create mock frontend/dist directory to verify static mounting
    dist_dir = tmp_path / "dist"
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html><body>TileMPPI 3D</body></html>", encoding="utf-8")
    (assets_dir / "app.js").write_text("console.log('tilemppi');", encoding="utf-8")

    cfg = EngineConfig(
        robot=RobotKind.QUADROTOR,
        scenario=ScenarioKind.OBSTACLE_SLALOM,
        mppi=MPPIConfig(num_rollouts=128, horizon=8, tile_k=64, num_vis_rollouts=16),
        sdf=SDFConfig(max_obstacles=8),
    )
    app = create_app(initial_config=cfg, frontend_dist_dir=dist_dir)

    with TestClient(app) as client:
        r_root = client.get("/")
        assert r_root.status_code == 200
        assert "TileMPPI 3D" in r_root.text

        r_health = client.get("/api/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "ok"

        r_hw = client.get("/api/hardware")
        assert r_hw.status_code == 200
        hw_data = r_hw.json()
        assert "hardware" in hw_data
        assert "l2_budget" in hw_data
        assert hw_data["l2_budget"]["fits_in_l2"] is True

        r_scen = client.get("/api/scenarios")
        assert r_scen.status_code == 200
        assert len(r_scen.json()["scenarios"]) == 5

        r_state = client.get("/api/state")
        assert r_state.status_code == 200
        assert len(r_state.json()["state"]) == 16

        # Invalid non-power-of-two rollout count must return HTTP 422
        r_bad_cfg = client.post("/api/config", json={"num_rollouts": 1000})
        assert r_bad_cfg.status_code == 422

        # Valid config update
        r_cfg = client.post(
            "/api/config",
            json={
                "robot": "quadruped",
                "scenario": "kick_recovery",
                "precision": "fp16",
                "num_rollouts": 128,
                "horizon": 8,
                "tile_k": 64,
                "temperature": 1.1,
                "noise_sigma": 0.5,
                "planner_enabled": True,
            },
        )
        assert r_cfg.status_code == 200
        assert r_cfg.json()["status"] == "updated"

        r_imp = client.post(
            "/api/impulse",
            json={"force": [5.0, 12.0, 2.0], "torque": [0.1, -0.1, 0.0], "duration_steps": 3},
        )
        assert r_imp.status_code == 200
        assert r_imp.json()["status"] == "impulse_applied"

        r_throw = client.post(
            "/api/obstacles/throw",
            json={"position": [0.5, 1.0, 0.5], "velocity": [-1.5, 0.0, 0.0], "radius": 0.35},
        )
        assert r_throw.status_code == 200
        assert r_throw.json()["status"] == "obstacle_thrown"

        r_goal = client.post("/api/goal", json={"goal": [2.2, -0.5, 0.42]})
        assert r_goal.status_code == 200
        assert np.allclose(r_goal.json()["goal"], [2.2, -0.5, 0.42])

        r_bench = client.post("/api/benchmark/quick")
        assert r_bench.status_code == 200
        assert r_bench.json()["control_hz"] > 0.0

        r_reset = client.post("/api/reset")
        assert r_reset.status_code == 200
        assert r_reset.json()["status"] == "reset"

        # WebSocket binary frame stream + interactive JSON commands
        with client.websocket_connect("/ws") as ws:
            frame0 = ws.receive_bytes()
            dec0 = decode_tmpi_frame(frame0)
            assert dec0.robot_id == RobotKind.QUADRUPED.numeric_id

            ws.send_text(json.dumps({"type": "pause", "paused": True}))
            ws.send_text(
                json.dumps(
                    {
                        "type": "impulse",
                        "force": [0.0, 15.0, 2.0],
                        "torque": [0.1, 0.1, 0.0],
                        "duration_steps": 3,
                    }
                )
            )
            ws.send_text(
                json.dumps(
                    {
                        "type": "throw",
                        "position": [1.0, 0.5, 0.5],
                        "velocity": [-1.0, 0.0, 0.0],
                        "radius": 0.4,
                    }
                )
            )
            ws.send_text(json.dumps({"type": "goal", "goal": [3.0, 0.0, 0.42]}))
            ws.send_text(json.dumps({"type": "step"}))

            matched = False
            for _ in range(10):
                frame1 = ws.receive_bytes()
                dec1 = decode_tmpi_frame(frame1)
                if np.allclose(dec1.goal, [3.0, 0.0, 0.42]) and (
                    np.linalg.norm(dec1.impulse_vec) > 1.0
                ):
                    matched = True
                    break
            assert matched, "Did not receive stepped WebSocket frame with updated goal and impulse"
