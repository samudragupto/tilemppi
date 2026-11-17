"""3D Inflated-SDF Voxel A* Global Planner with Line-of-Sight Smoothing."""

from __future__ import annotations

import heapq

import numpy as np

from tilemppi.config import PlannerConfig
from tilemppi.sdf.primitives import evaluate_sdf_batch
from tilemppi.types import Float32Array


class GlobalAStarPlanner:
    """Fast 3D voxel A* path planner with SDF obstacle inflation and string-pulling."""

    def __init__(
        self,
        config: PlannerConfig | None = None,
        bounds_xy: tuple[float, float] = (-5.2, 5.2),
        bounds_z: tuple[float, float] = (0.3, 3.2),
    ) -> None:
        self.config = config or PlannerConfig()
        self.bounds_xy = bounds_xy
        self.bounds_z = bounds_z

    def _is_segment_collision_free(
        self,
        p_start: Float32Array,
        p_end: Float32Array,
        obstacles_table: Float32Array,
        clearance_radius: float,
        num_checks: int = 16,
    ) -> bool:
        alphas = np.linspace(0.0, 1.0, num_checks, dtype=np.float32).reshape(-1, 1)
        pts = p_start.reshape(1, 3) * (np.float32(1.0) - alphas) + p_end.reshape(1, 3) * alphas
        d = evaluate_sdf_batch(pts, obstacles_table, t=0.0)
        return bool(np.all(d >= np.float32(clearance_radius)))

    def smooth_path_line_of_sight(
        self,
        waypoints: list[Float32Array],
        obstacles_table: Float32Array,
        clearance_radius: float,
    ) -> list[Float32Array]:
        """Greedy string-pulling shortcutting across collision-free line-of-sight segments."""
        if len(waypoints) <= 2:
            return waypoints

        smoothed: list[Float32Array] = [waypoints[0]]
        curr_idx = 0
        n_pts = len(waypoints)

        while curr_idx < n_pts - 1:
            next_idx = curr_idx + 1
            for candidate_idx in range(n_pts - 1, curr_idx + 1, -1):
                if self._is_segment_collision_free(
                    waypoints[curr_idx],
                    waypoints[candidate_idx],
                    obstacles_table,
                    clearance_radius=clearance_radius,
                ):
                    next_idx = candidate_idx
                    break
            smoothed.append(waypoints[next_idx])
            curr_idx = next_idx

        return smoothed

    def plan_path(
        self,
        start_xyz: Float32Array,
        goal_xyz: Float32Array,
        obstacles_table: Float32Array,
        robot_radius: float = 0.28,
    ) -> list[Float32Array]:
        """Compute a collision-free 3D waypoint path from start_xyz to goal_xyz."""
        start = np.asarray(start_xyz, dtype=np.float32).reshape(3)
        goal = np.asarray(goal_xyz, dtype=np.float32).reshape(3)
        clearance = float(robot_radius + self.config.inflation_radius)

        # Fast path: if straight line is already collision-free, return [start, goal]
        if self._is_segment_collision_free(start, goal, obstacles_table, clearance, num_checks=20):
            return [start.copy(), goal.copy()]

        res = float(self.config.grid_resolution)
        xs = np.arange(self.bounds_xy[0], self.bounds_xy[1] + 0.5 * res, res, dtype=np.float32)
        ys = np.arange(self.bounds_xy[0], self.bounds_xy[1] + 0.5 * res, res, dtype=np.float32)
        # Keep Z plane around start/goal operating altitude plus upper bypass layer
        z_mid = float(0.5 * (start[2] + goal[2]))
        zs = np.array([z_mid], dtype=np.float32)

        nx, ny, nz = len(xs), len(ys), len(zs)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

        sdf_grid = evaluate_sdf_batch(grid_pts, obstacles_table, t=0.0).reshape(nx, ny, nz)
        free_mask = sdf_grid >= np.float32(clearance)

        def to_idx(pt: Float32Array) -> tuple[int, int, int]:
            ix = int(np.clip(np.round((pt[0] - xs[0]) / res), 0, nx - 1))
            iy = int(np.clip(np.round((pt[1] - ys[0]) / res), 0, ny - 1))
            iz = int(np.clip(np.round((pt[2] - zs[0]) / res), 0, nz - 1))
            return ix, iy, iz

        def to_world(idx: tuple[int, int, int]) -> Float32Array:
            return np.array([xs[idx[0]], ys[idx[1]], zs[idx[2]]], dtype=np.float32)

        s_idx = to_idx(start)
        g_idx = to_idx(goal)
        free_mask[s_idx] = True
        free_mask[g_idx] = True

        neighbors_2d = [
            (1, 0, 0, 1.0),
            (-1, 0, 0, 1.0),
            (0, 1, 0, 1.0),
            (0, -1, 0, 1.0),
            (1, 1, 0, 1.4142),
            (1, -1, 0, 1.4142),
            (-1, 1, 0, 1.4142),
            (-1, -1, 0, 1.4142),
        ]

        open_heap: list[tuple[float, tuple[int, int, int]]] = []
        heapq.heappush(open_heap, (0.0, s_idx))
        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        g_score: dict[tuple[int, int, int], float] = {s_idx: 0.0}
        closed: set[tuple[int, int, int]] = set()

        found = False
        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == g_idx:
                found = True
                break
            if current in closed:
                continue
            closed.add(current)

            cx, cy, cz = current
            for dx, dy, dz, step_cost in neighbors_2d:
                nb = (cx + dx, cy + dy, cz + dz)
                if not (0 <= nb[0] < nx and 0 <= nb[1] < ny and 0 <= nb[2] < nz):
                    continue
                if not free_mask[nb] or nb in closed:
                    continue

                # Add gentle proximity cost near obstacles for wider clearance
                obs_clear = float(sdf_grid[nb])
                prox_penalty = 0.35 / max(obs_clear - clearance + 0.1, 0.05)
                tentative_g = g_score[current] + step_cost + prox_penalty

                if tentative_g < g_score.get(nb, 1.0e12):
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    h = float(np.hypot(nb[0] - g_idx[0], nb[1] - g_idx[1]))
                    heapq.heappush(open_heap, (tentative_g + h, nb))

        if not found:
            return [start.copy(), goal.copy()]

        raw_nodes: list[tuple[int, int, int]] = [g_idx]
        curr = g_idx
        while curr in came_from:
            curr = came_from[curr]
            raw_nodes.append(curr)
        raw_nodes.reverse()

        waypoints = [start.copy()]
        for node in raw_nodes[1:-1]:
            waypoints.append(to_world(node))
        waypoints.append(goal.copy())

        return self.smooth_path_line_of_sight(
            waypoints,
            obstacles_table,
            clearance_radius=clearance * 0.85,
        )

    def select_lookahead_waypoint(
        self,
        robot_pos: Float32Array,
        waypoints: list[Float32Array],
    ) -> Float32Array:
        """Select active carrot waypoint along the polyline at lookahead_distance ahead of robot_pos."""
        if not waypoints:
            return np.asarray(robot_pos, dtype=np.float32).reshape(3).copy()
        if len(waypoints) == 1:
            return waypoints[0].copy()

        pos = np.asarray(robot_pos, dtype=np.float32).reshape(3)
        lookahead = float(self.config.lookahead_distance)

        # Find orthogonal projection of robot_pos onto the polyline segments
        best_dist_sq = 1.0e18
        best_seg_idx = 0
        best_proj_pt = waypoints[0].copy()

        for idx in range(len(waypoints) - 1):
            p_a = waypoints[idx]
            p_b = waypoints[idx + 1]
            seg = p_b - p_a
            seg_len_sq = float(np.dot(seg, seg))
            if seg_len_sq < 1e-8:
                proj = p_a
            else:
                t_val = float(np.clip(np.dot(pos - p_a, seg) / seg_len_sq, 0.0, 1.0))
                proj = p_a + np.float32(t_val) * seg
            d_sq = float(np.sum((pos - proj) ** 2))
            if d_sq <= best_dist_sq:
                best_dist_sq = d_sq
                best_seg_idx = idx
                best_proj_pt = np.asarray(proj, dtype=np.float32)

        # Walk forward along the polyline starting from best_proj_pt on best_seg_idx
        accum = 0.0
        curr_pt = best_proj_pt
        for idx in range(best_seg_idx, len(waypoints) - 1):
            p_next = waypoints[idx + 1]
            seg_vec = p_next - curr_pt
            seg_len = float(np.linalg.norm(seg_vec))
            if accum + seg_len >= lookahead and seg_len > 1e-6:
                ratio = (lookahead - accum) / seg_len
                return np.asarray(
                    curr_pt + np.float32(ratio) * seg_vec,
                    dtype=np.float32,
                )
            accum += seg_len
            curr_pt = p_next

        return waypoints[-1].copy()
