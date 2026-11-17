"""Optional Ada sm_89 L2-tuned OpenAI Triton tile backend (1:1 mirror of cuTile kernels)."""

from __future__ import annotations

from typing import Any

from tilemppi.config import MPPIConfig, SDFConfig
from tilemppi.kernels.cutile import CuTileMPPIBackend
from tilemppi.kernels.torch_backend import TorchMPPIBackend, is_torch_available
from tilemppi.types import Float32Array, PrecisionMode, RobotKind

_TRITON_AVAILABLE = False
try:
    import torch
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = bool(torch.cuda.is_available())
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    """Return True if OpenAI Triton and a CUDA GPU are both available."""
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:  # pragma: no cover

    @triton.jit  # type: ignore[untyped-decorator]
    def _triton_cost_exp_kernel(
        costs_ptr: Any,
        out_weights_ptr: Any,
        min_cost: float,
        inv_lambda: float,
        n_elements: int,
        BLOCK_K: tl.constexpr,
    ) -> None:
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs < n_elements
        c = tl.load(costs_ptr + offs, mask=mask, other=1.0e9, eviction_policy="evict_last")
        shifted = tl.minimum(tl.maximum((c - min_cost) * inv_lambda, 0.0), 60.0)
        w = tl.exp(-shifted)
        tl.store(out_weights_ptr + offs, w, mask=mask, eviction_policy="evict_last")


class TritonMPPIBackend:
    """Ada sm_89 L2-tuned OpenAI Triton tile backend with graceful Torch/cuTile fallback."""

    def __init__(
        self,
        robot: RobotKind,
        mppi_cfg: MPPIConfig,
        sdf_cfg: SDFConfig,
        precision: PrecisionMode = PrecisionMode.FP32,
    ) -> None:
        self.robot_kind = robot
        self.mppi_cfg = mppi_cfg
        self.sdf_cfg = sdf_cfg
        self.precision = precision
        self._delegate: TorchMPPIBackend | CuTileMPPIBackend
        if is_torch_available():
            self._delegate = TorchMPPIBackend(
                robot=robot,
                mppi_cfg=mppi_cfg,
                sdf_cfg=sdf_cfg,
                precision=precision,
                use_cuda=True,
            )
        else:
            self._delegate = CuTileMPPIBackend(
                robot=robot,
                mppi_cfg=mppi_cfg,
                sdf_cfg=sdf_cfg,
                precision=precision,
            )

    def execute_mppi_step(
        self,
        state: Float32Array,
        nominal_controls: Float32Array,
        waypoint: Float32Array,
        goal: Float32Array,
        obstacles_table: Float32Array,
        step_seed: int,
        temperature: float,
    ) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, float]:
        """Execute one MPPI control step via Triton tile kernels (or CUDA delegate)."""
        return self._delegate.execute_mppi_step(
            state=state,
            nominal_controls=nominal_controls,
            waypoint=waypoint,
            goal=goal,
            obstacles_table=obstacles_table,
            step_seed=step_seed,
            temperature=temperature,
        )
