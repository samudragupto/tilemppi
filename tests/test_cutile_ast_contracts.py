"""Tests for NVIDIA cuTile (@ct.kernel) static AST verification & tile runtime contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tilemppi import cutile_compat as ct
from tilemppi.kernels.cutile import ALL_CUTILE_KERNELS, verify_all_cutile_kernels


@pytest.mark.parametrize("tile_k", [64, 128, 256])
def test_all_production_cutile_kernels_pass_ast_contracts(tile_k: int) -> None:
    assert len(ALL_CUTILE_KERNELS) == 8
    reports = verify_all_cutile_kernels(tile_k=tile_k)
    for rep in reports:
        assert rep.is_valid, f"Kernel {rep.kernel_name} failed AST check: {rep.violations}"
        assert rep.has_kernel_decorator is True
        assert rep.uses_bid is True
        assert rep.num_tile_loads >= 1
        assert rep.num_tile_stores >= 1
        assert rep.estimated_sram_bytes < 96 * 1024
        assert f"@{rep.kernel_name}_tile_ir" in rep.tile_ir_summary
        assert len(rep.constant_params) >= 1
        rep_dict = rep.to_dict()
        assert rep_dict["is_valid"] is True


def test_ast_verifier_rejects_simt_and_missing_contracts() -> None:
    def bad_kernel_no_decorator(arr: Any, TILE_K: ct.ConstInt) -> None:
        threadIdx = 0  # Forbidden SIMT identifier
        _ = threadIdx

    rep = ct.verify_cutile_kernel(bad_kernel_no_decorator, tile_k=64)
    assert rep.is_valid is False
    assert any("missing @ct.kernel" in v for v in rep.violations)
    assert any("threadIdx" in v for v in rep.violations)
    assert any("ct.bid()" in v for v in rep.violations)

    with pytest.raises(ValueError, match="power of two"):
        ct.verify_cutile_kernel(ALL_CUTILE_KERNELS[0], tile_k=100)


def test_cutile_runtime_enforces_power_of_two_tile_shapes() -> None:
    t_ok = ct.zeros((64, 16), dtype=ct.float32)
    assert t_ok.shape == (64, 16)

    t_ones = ct.ones((32, 4), dtype=ct.float16)
    assert t_ones.shape == (32, 4)

    with pytest.raises(ValueError, match="power of two"):
        ct.zeros((10, 16))

    with pytest.raises(ValueError, match="power of two"):
        ct.arange(0, 12)

    # Test out-of-bounds zero-padding and clamp-padding in ct.load
    src = np.arange(16, dtype=np.float32).reshape(4, 4)
    loaded_zero = ct.load(src, index=(2, 0), shape=(4, 4), padding_mode=ct.PaddingMode.ZERO)
    assert np.allclose(loaded_zero, 0.0)

    loaded_clamp = ct.load(src, index=(2, 0), shape=(4, 4), padding_mode=ct.PaddingMode.CLAMP)
    assert np.allclose(loaded_clamp[0], src[3])
