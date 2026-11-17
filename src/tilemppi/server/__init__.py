"""FastAPI server and compact binary 'TMPI' WebSocket telemetry protocol for TileMPPI."""

from __future__ import annotations

from tilemppi.server.app import create_app
from tilemppi.server.protocol import (
    TMPI_HEADER_BYTES,
    TMPI_MAGIC,
    TMPI_VERSION,
    DecodedTMPIFrame,
    decode_tmpi_frame,
    encode_tmpi_frame,
)

__all__ = [
    "TMPI_HEADER_BYTES",
    "TMPI_MAGIC",
    "TMPI_VERSION",
    "DecodedTMPIFrame",
    "create_app",
    "decode_tmpi_frame",
    "encode_tmpi_frame",
]
