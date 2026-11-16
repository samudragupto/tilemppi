"""Tests for 3D Signed Distance Function primitives, normals, dynamic kinematics, and barriers."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tilemppi.config import SDFConfig
from tilemppi.sdf.primitives import (
    ObstaclePrimitive,
    compute_barrier_cost,
    evaluate_sdf_and_normals,
    evaluate_sdf_batch,
    evaluate_single_primitive_sdf,
    pack_obstacles_table,
    predict_obstacle_positions,
    smooth_min_sdf,
    step_dynamic_obstacles_in_place,
)


def test_sphere_sdf_surface_interior_exterior() -> None:
    sphere = ObstaclePrimitive.sphere(position=(1.0, 2.0, 3.0), radius=0.5)
    row = sphere.to_row16()
    pts = np.array(
        [
            [1.0, 2.0, 3.0],  # Center: d = -0.5
            [1.5, 2.0, 3.0],  # Surface +X: d = 0.0
            [1.0, 3.5, 3.0],  # Exterior +Y: d = 1.0
        ],
        dtype=np.float32,
    )
    d = evaluate_single_primitive_sdf(pts, row, t=0.0)
    assert np.allclose(d, [-0.5, 0.0, 1.0], atol=1e-5)


def test_capsule_sdf_exact() -> None:
    cap = ObstaclePrimitive.capsule(
        position=(0.0, 0.0, 1.0),
        radius=0.3,
        half_height=0.8,
    )
    row = cap.to_row16()
    pts = np.array(
        [
            [0.0, 0.0, 1.0],  # Mid-axis center: -0.3
            [0.3, 0.0, 1.4],  # Cylinder side surface: 0.0
            [0.0, 0.0, 2.1],  # Top hemisphere cap surface (1.0 + 0.8 + 0.3): 0.0
            [0.0, 0.0, 3.1],  # 1.0m above top cap: 1.0
        ],
        dtype=np.float32,
    )
    d = evaluate_single_primitive_sdf(pts, row)
    assert np.allclose(d, [-0.3, 0.0, 0.0, 1.0], atol=1e-5)


def test_oriented_box_sdf_and_quaternion_rotation() -> None:
    # 90-degree rotation around Z axis: quat = [cos(pi/4), 0, 0, sin(pi/4)]
    c = float(np.cos(np.pi / 4.0))
    s = float(np.sin(np.pi / 4.0))
    box = ObstaclePrimitive.oriented_box(
        position=(0.0, 0.0, 0.0),
        half_extents=(1.0, 0.25, 0.5),
        quat_wxyz=(c, 0.0, 0.0, s),
    )
    row = box.to_row16()
    # After +90 deg Z rotation, local X (half=1.0) aligns with world +Y,
    # and local Y (half=0.25) aligns with world -X.
    pts = np.array(
        [
            [0.0, 0.0, 0.0],  # Center: inside by min(1.0, 0.25, 0.5) = -0.25
            [0.0, 1.0, 0.0],  # Surface on world +Y: 0.0
            [0.25, 0.0, 0.0],  # Surface on world +X: 0.0
            [1.25, 0.0, 0.0],  # 1.0m outside world +X: 1.0
        ],
        dtype=np.float32,
    )
    d = evaluate_single_primitive_sdf(pts, row)
    assert np.allclose(d, [-0.25, 0.0, 0.0, 1.0], atol=1e-4)


def test_torus_gate_fly_through_hole_vs_ring_collision() -> None:
    gate = ObstaclePrimitive.torus_gate(
        position=(0.0, 0.0, 1.5),
        major_radius=1.2,
        tube_radius=0.2,
    )
    row = gate.to_row16()
    pts = np.array(
        [
            [0.0, 0.0, 1.5],  # Exact center of gate hole: safe clearance = R - r = 1.0 > 0!
            [0.0, 1.2, 1.5],  # Inside solid torus tube centerline: d = -r = -0.2 < 0!
            [0.0, 1.0, 1.5],  # Inner rim surface of torus tube: d = 0.0
            [0.0, 1.4, 1.5],  # Outer rim surface of torus tube: d = 0.0
        ],
        dtype=np.float32,
    )
    d = evaluate_single_primitive_sdf(pts, row)
    assert np.allclose(d, [1.0, -0.2, 0.0, 0.0], atol=1e-5)


def test_ground_plane_and_dynamic_kinematics() -> None:
    ground = ObstaclePrimitive.ground_plane(z_height=0.1)
    moving_sphere = ObstaclePrimitive.sphere(
        position=(0.0, 0.0, 2.0),
        radius=0.4,
        velocity=(2.0, -1.0, 0.0),
        acceleration=(0.0, 2.0, 0.0),
    )
    table = pack_obstacles_table([ground, moving_sphere], max_obstacles=8)
    assert table.shape == (8, 16)

    # At t = 1.0s, moving_sphere position = (0 + 2*1, 0 - 1*1 + 0.5*2*1^2, 2.0) = (2.0, 0.0, 2.0)
    pred = predict_obstacle_positions(table, t=1.0)
    assert np.allclose(pred[1], [2.0, 0.0, 2.0], atol=1e-5)

    pts = np.array([[2.0, 0.0, 2.4], [0.0, 0.0, 0.35]], dtype=np.float32)
    d = evaluate_sdf_batch(pts, table, t=1.0)
    # Point 0 is on top surface of sphere at t=1.0 (d=0.0); Point 1 is 0.25m above ground (d=0.25)
    assert np.allclose(d, [0.0, 0.25], atol=1e-4)

    # Arena bounce test
    step_dynamic_obstacles_in_place(table, dt=5.0, arena_bounds=(3.0, 3.0, 4.0))
    assert abs(float(table[1, 1])) <= 3.0


def test_sdf_normals_unit_length_and_analytical_match() -> None:
    sphere = ObstaclePrimitive.sphere(position=(0.0, 0.0, 1.0), radius=0.5)
    table = pack_obstacles_table([sphere], max_obstacles=4)
    pts = np.array(
        [
            [2.0, 0.0, 1.0],
            [0.0, -1.5, 1.0],
            [0.0, 0.0, 2.5],
        ],
        dtype=np.float32,
    )
    sdf_val, normals = evaluate_sdf_and_normals(pts, table, t=0.0)
    norms = np.linalg.norm(normals, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    expected_normals = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(normals, expected_normals, atol=1e-4)
    assert np.all(sdf_val > 0.0)


def test_smooth_min_and_barrier_monotonicity() -> None:
    d1 = np.array([0.2, 0.5, 1.0], dtype=np.float32)
    d2 = np.array([0.22, 2.0, 1.0], dtype=np.float32)
    smin = smooth_min_sdf(d1, d2, k=0.2)
    assert np.all(smin <= np.minimum(d1, d2) + 1e-6)

    cfg = SDFConfig()
    distances = np.linspace(-0.2, 2.0, 32, dtype=np.float32)
    costs = compute_barrier_cost(distances, cfg)
    assert np.all(np.diff(costs) <= 1e-5)


@settings(max_examples=35, deadline=None)
@given(
    px=st.floats(-4.0, 4.0),
    py=st.floats(-4.0, 4.0),
    pz=st.floats(0.2, 4.0),
    dx=st.floats(-0.4, 0.4),
    dy=st.floats(-0.4, 0.4),
    dz=st.floats(-0.4, 0.4),
)
def test_sdf_lipschitz_property_hypothesis(
    px: float,
    py: float,
    pz: float,
    dx: float,
    dy: float,
    dz: float,
) -> None:
    """Any valid Euclidean SDF must satisfy |SDF(x) - SDF(y)| <= ||x - y||_2."""
    obs = [
        ObstaclePrimitive.sphere((0.0, 0.0, 1.5), 0.5),
        ObstaclePrimitive.oriented_box((1.2, -0.5, 1.2), (0.4, 0.4, 0.6)),
        ObstaclePrimitive.torus_gate((-1.0, 0.5, 1.5), 1.0, 0.2),
    ]
    table = pack_obstacles_table(obs, max_obstacles=4)
    p1 = np.array([[px, py, pz]], dtype=np.float32)
    p2 = np.array([[px + dx, py + dy, pz + dz]], dtype=np.float32)
    d1 = float(evaluate_sdf_batch(p1, table)[0])
    d2 = float(evaluate_sdf_batch(p2, table)[0])
    step_norm = float(np.linalg.norm(p2 - p1))
    assert abs(d1 - d2) <= step_norm + 1e-3
