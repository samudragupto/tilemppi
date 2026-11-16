"""Shared pytest fixtures for the TileMPPI test suite."""

from __future__ import annotations

import numpy as np
import pytest

from tilemppi.hardware import HardwareCapabilities, detect_hardware


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a deterministic NumPy Generator."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def host_hardware() -> HardwareCapabilities:
    """Return detected host hardware capabilities."""
    return detect_hardware()
