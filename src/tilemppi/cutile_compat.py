"""NVIDIA cuTile (cuda.tile) compatibility layer, strict Tile Emulator, and Static AST Verifier."""

from __future__ import annotations

import ast
import builtins
import inspect
import shutil
import textwrap
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Generic, TypeAlias, TypeVar

import numpy as np

from tilemppi.types import is_power_of_two

_py_min = builtins.min
_py_max = builtins.max
_py_abs = builtins.abs
_py_sum = builtins.sum

T = TypeVar("T")

# Probe native cuda.tile installation
_NATIVE_CT: Any = None
try:
    import cuda.tile as _native_ct_mod

    _NATIVE_CT = _native_ct_mod
except Exception:
    _NATIVE_CT = None


class _ConstantMeta(type):
    """Metaclass enabling ct.Constant[int] type subscripting."""

    def __getitem__(cls, item: Any) -> Any:
        return item


class Constant(Generic[T], metaclass=_ConstantMeta):
    """Compile-time tile constant annotation matching cuda.tile.Constant[T]."""


ConstInt: TypeAlias = Annotated[int, "ct.Constant[int]"]
ConstFloat: TypeAlias = Annotated[float, "ct.Constant[float]"]


class PaddingMode(StrEnum):
    """Tile boundary load padding modes matching cuda.tile.PaddingMode."""

    ZERO = "zero"
    CLAMP = "clamp"
    REFLECT = "reflect"


class TileDType:
    """Descriptor for cuTile scalar & tile data types."""

    def __init__(self, name: str, np_dtype: Any, itemsize: int) -> None:
        self.name = name
        self.np_dtype = np.dtype(np_dtype)
        self.itemsize = itemsize

    def __repr__(self) -> str:
        return f"ct.{self.name}"


float32 = TileDType("float32", np.float32, 4)
float16 = TileDType("float16", np.float16, 2)
# FP8 E4M3FN represented in emulator as float32 with E4M3 quantization range [-448, 448]
float8_e4m3fn = TileDType("float8_e4m3fn", np.float32, 1)
int32 = TileDType("int32", np.int32, 4)
uint32 = TileDType("uint32", np.uint32, 4)
bool_ = TileDType("bool_", np.bool_, 1)


class _ExecutionContext(threading.local):
    """Thread-local tile block execution state for cutile_sim."""

    def __init__(self) -> None:
        super().__init__()
        self.active: bool = False
        self.bid_coords: tuple[int, int, int] = (0, 0, 0)
        self.grid_dims: tuple[int, int, int] = (1, 1, 1)


_CTX = _ExecutionContext()


def _resolve_np_dtype(dtype: Any) -> np.dtype[Any]:
    if isinstance(dtype, TileDType):
        dt_tile: np.dtype[Any] = dtype.np_dtype
        return dt_tile
    dt_raw: np.dtype[Any] = np.dtype(dtype)
    return dt_raw


def _quantize_fp8_e4m3(arr: np.ndarray) -> np.ndarray:
    """Quantize float array to NVIDIA FP8 E4M3FN precision in float32 container."""
    clipped = np.clip(arr.astype(np.float32), np.float32(-448.0), np.float32(448.0))
    abs_val = np.abs(clipped)
    sign = np.sign(clipped)
    nonzero = abs_val > np.float32(1e-6)
    safe_abs = np.where(nonzero, abs_val, np.float32(1.0))
    exp2 = np.floor(np.log2(safe_abs))
    scale = np.power(np.float32(2.0), exp2 - np.float32(3.0))
    safe_scale = np.maximum(scale, np.float32(1e-5))
    q = sign * np.round(abs_val / safe_scale) * safe_scale
    return np.asarray(np.where(nonzero, q, np.float32(0.0)), dtype=np.float32)


def _validate_tile_shape(shape: tuple[int, ...], op_name: str = "tile") -> None:
    """Enforce that every tile dimension is 1 or a power of two (or 3D spatial axis)."""
    for dim in shape:
        d = int(dim)
        if d <= 0:
            msg = f"Invalid non-positive tile dimension {d} in {op_name} with shape {shape}"
            raise ValueError(msg)
        if d != 3 and not is_power_of_two(d):
            msg = (
                f"cuTile shape dimension {d} in {shape} ({op_name}) "
                f"must be a power of two (or 3D axis)"
            )
            raise ValueError(msg)


def cdiv(a: int, b: int) -> int:
    """Ceiling integer division matching cuda.tile.cdiv(a, b)."""
    return (int(a) + int(b) - 1) // int(b)


def bid(axis: int = 0) -> int:
    """Return current tile block index along axis in {0, 1, 2}."""
    if axis < 0 or axis > 2:
        msg = f"ct.bid axis must be 0, 1, or 2, got {axis}"
        raise ValueError(msg)
    return int(_CTX.bid_coords[axis])


def zeros(shape: tuple[int, ...] | int, dtype: Any = float32) -> np.ndarray:
    """Allocate a zero-initialized tile of the given power-of-two shape."""
    shp = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
    _validate_tile_shape(shp, "ct.zeros")
    return np.zeros(shp, dtype=_resolve_np_dtype(dtype))


def ones(shape: tuple[int, ...] | int, dtype: Any = float32) -> np.ndarray:
    """Allocate a one-initialized tile of the given power-of-two shape."""
    shp = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
    _validate_tile_shape(shp, "ct.ones")
    return np.ones(shp, dtype=_resolve_np_dtype(dtype))


def full(shape: tuple[int, ...] | int, fill_value: Any, dtype: Any = float32) -> np.ndarray:
    """Allocate a constant-filled tile of the given power-of-two shape."""
    shp = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
    _validate_tile_shape(shp, "ct.full")
    out = np.full(shp, fill_value, dtype=_resolve_np_dtype(dtype))
    if dtype is float8_e4m3fn:
        return _quantize_fp8_e4m3(out)
    return out


def arange(start: int, stop: int | None = None, dtype: Any = int32) -> np.ndarray:
    """Create a 1D sequence tile with power-of-two length."""
    if stop is None:
        lo, hi = 0, int(start)
    else:
        lo, hi = int(start), int(stop)
    length = hi - lo
    _validate_tile_shape((length,), "ct.arange")
    return np.arange(lo, hi, dtype=_resolve_np_dtype(dtype))


def load(
    arr: np.ndarray,
    index: tuple[int, ...] | int,
    shape: tuple[int, ...] | int,
    padding_mode: PaddingMode = PaddingMode.ZERO,
) -> np.ndarray:
    """Load a block tile from global array using tile-block coordinates."""
    idx_tuple = (int(index),) if isinstance(index, int) else tuple(int(i) for i in index)
    shp_tuple = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
    _validate_tile_shape(shp_tuple, "ct.load")

    if len(idx_tuple) != len(shp_tuple) or len(shp_tuple) != arr.ndim:
        msg = f"ct.load rank mismatch: arr.ndim={arr.ndim}, index={idx_tuple}, shape={shp_tuple}"
        raise ValueError(msg)

    out = np.zeros(shp_tuple, dtype=arr.dtype)
    src_slices: list[slice] = []
    dst_slices: list[slice] = []

    for dim_i, (blk_idx, tile_len) in enumerate(zip(idx_tuple, shp_tuple, strict=True)):
        start = blk_idx * tile_len
        end = start + tile_len
        arr_limit = int(arr.shape[dim_i])
        if start >= arr_limit or end <= 0:
            if padding_mode == PaddingMode.CLAMP and arr_limit > 0:
                clamped_idx = _py_min(_py_max(start, 0), arr_limit - 1)
                src_slices.append(slice(clamped_idx, clamped_idx + 1))
                dst_slices.append(slice(0, tile_len))
            else:
                return out
        else:
            s_lo = _py_max(0, start)
            s_hi = _py_min(arr_limit, end)
            d_lo = s_lo - start
            d_hi = d_lo + (s_hi - s_lo)
            src_slices.append(slice(s_lo, s_hi))
            dst_slices.append(slice(d_lo, d_hi))

    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


def store(
    arr: np.ndarray,
    index: tuple[int, ...] | int,
    tile: np.ndarray,
) -> None:
    """Store a block tile into global array using tile-block coordinates."""
    t_arr = np.asarray(tile)
    idx_tuple = (int(index),) if isinstance(index, int) else tuple(int(i) for i in index)
    shp_tuple = t_arr.shape
    _validate_tile_shape(shp_tuple, "ct.store")

    if len(idx_tuple) != len(shp_tuple) or len(shp_tuple) != arr.ndim:
        msg = (
            f"ct.store rank mismatch: arr.ndim={arr.ndim}, "
            f"index={idx_tuple}, tile.shape={shp_tuple}"
        )
        raise ValueError(msg)

    dst_slices: list[slice] = []
    src_slices: list[slice] = []
    for dim_i, (blk_idx, tile_len) in enumerate(zip(idx_tuple, shp_tuple, strict=True)):
        start = blk_idx * tile_len
        end = start + tile_len
        arr_limit = int(arr.shape[dim_i])
        if start >= arr_limit or end <= 0:
            return
        d_lo = _py_max(0, start)
        d_hi = _py_min(arr_limit, end)
        s_lo = d_lo - start
        s_hi = s_lo + (d_hi - d_lo)
        dst_slices.append(slice(d_lo, d_hi))
        src_slices.append(slice(s_lo, s_hi))

    arr[tuple(dst_slices)] = t_arr[tuple(src_slices)].astype(arr.dtype, copy=False)


def matmul(a: np.ndarray, b: np.ndarray, acc: np.ndarray | None = None) -> np.ndarray:
    """Tensor Core tile matrix multiplication with FP32 accumulation."""
    a_f32 = np.asarray(a, dtype=np.float32)
    b_f32 = np.asarray(b, dtype=np.float32)
    res = np.matmul(a_f32, b_f32)
    if acc is not None:
        res = res + np.asarray(acc, dtype=np.float32)
    return np.asarray(res, dtype=np.float32)


def mma(a: np.ndarray, b: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Tensor Core matrix multiply-accumulate (MMA): acc + a @ b."""
    return matmul(a, b, acc=acc)


def exp(x: Any) -> np.ndarray:
    return np.asarray(np.exp(x), dtype=np.float32)


def log(x: Any) -> np.ndarray:
    return np.asarray(np.log(x), dtype=np.float32)


def sqrt(x: Any) -> np.ndarray:
    return np.asarray(np.sqrt(x), dtype=np.float32)


def rsqrt(x: Any) -> np.ndarray:
    return np.asarray(np.float32(1.0) / np.sqrt(x), dtype=np.float32)


def sin(x: Any) -> np.ndarray:
    return np.asarray(np.sin(x), dtype=np.float32)


def cos(x: Any) -> np.ndarray:
    return np.asarray(np.cos(x), dtype=np.float32)


def atan2(y: Any, x: Any) -> np.ndarray:
    return np.asarray(np.arctan2(y, x), dtype=np.float32)


def abs(x: Any) -> np.ndarray:
    return np.asarray(np.abs(x))


def maximum(a: Any, b: Any) -> np.ndarray:
    return np.asarray(np.maximum(a, b))


def minimum(a: Any, b: Any) -> np.ndarray:
    return np.asarray(np.minimum(a, b))


def clip(x: Any, a_min: Any, a_max: Any) -> np.ndarray:
    return np.asarray(np.clip(x, a_min, a_max))


def where(condition: Any, x: Any, y: Any) -> np.ndarray:
    return np.asarray(np.where(condition, x, y))


def astype(x: Any, dtype: Any) -> np.ndarray:
    arr = np.asarray(x)
    if dtype is float8_e4m3fn:
        return _quantize_fp8_e4m3(arr)
    if dtype is float16:
        return np.asarray(arr.astype(np.float16).astype(np.float32), dtype=np.float32)
    return np.asarray(arr.astype(_resolve_np_dtype(dtype)))


def reshape(x: Any, shape: tuple[int, ...]) -> np.ndarray:
    shp = tuple(int(s) for s in shape)
    _validate_tile_shape(shp, "ct.reshape")
    return np.asarray(x).reshape(shp)


def transpose(x: Any, axes: tuple[int, ...] | None = None) -> np.ndarray:
    return np.transpose(np.asarray(x), axes=axes)


def broadcast_to(x: Any, shape: tuple[int, ...]) -> np.ndarray:
    shp = tuple(int(s) for s in shape)
    _validate_tile_shape(shp, "ct.broadcast_to")
    return np.asarray(np.broadcast_to(np.asarray(x), shp).copy())


def extract(
    tile: np.ndarray,
    index: tuple[int, ...],
    shape: tuple[int, ...],
) -> np.ndarray:
    """Extract a sub-tile from a register tile using start offsets and power-of-two shape."""
    _validate_tile_shape(shape, "ct.extract")
    slices = tuple(slice(int(i), int(i) + int(s)) for i, s in zip(index, shape, strict=True))
    return np.asarray(tile[slices]).copy()


def cat(tiles: list[np.ndarray] | tuple[np.ndarray, ...], axis: int = -1) -> np.ndarray:
    """Concatenate register tiles along the specified axis."""
    out = np.asarray(np.concatenate(tiles, axis=axis))
    _validate_tile_shape(out.shape, "ct.cat")
    return out


def sum(x: Any, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Any:
    arr = np.asarray(x)
    return np.sum(arr, axis=axis, keepdims=True) if keepdims else np.sum(arr, axis=axis)


def max(x: Any, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Any:
    arr = np.asarray(x)
    return np.max(arr, axis=axis, keepdims=True) if keepdims else np.max(arr, axis=axis)


def min(x: Any, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Any:
    arr = np.asarray(x)
    return np.min(arr, axis=axis, keepdims=True) if keepdims else np.min(arr, axis=axis)


class CuTileKernelWrapper:
    """Callable wrapper produced by @ct.kernel preserving AST inspection & native dispatch."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.py_func = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__qualname__ = fn.__qualname__
        self.__module__ = fn.__module__
        self._is_cutile_kernel = True
        self._native_kernel: Any = None
        if _NATIVE_CT is not None and hasattr(_NATIVE_CT, "kernel"):
            try:
                self._native_kernel = _NATIVE_CT.kernel(fn)
            except Exception:
                self._native_kernel = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.py_func(*args, **kwargs)


def kernel(fn: Callable[..., Any]) -> CuTileKernelWrapper:
    """Decorator marking a function as an NVIDIA cuTile (@ct.kernel) tile program."""
    return CuTileKernelWrapper(fn)


def launch(
    stream: Any,
    grid: tuple[int, ...],
    kernel_fn: CuTileKernelWrapper | Callable[..., Any],
    args: tuple[Any, ...],
    prefer_native: bool = False,
) -> None:
    """Launch a @ct.kernel over a 1D/2D/3D tile grid on either native cuda.tile or cutile_sim."""
    if (
        prefer_native
        and _NATIVE_CT is not None
        and isinstance(kernel_fn, CuTileKernelWrapper)
        and kernel_fn._native_kernel is not None
    ):
        _NATIVE_CT.launch(stream, grid, kernel_fn._native_kernel, args)
        return

    raw_fn = kernel_fn.py_func if isinstance(kernel_fn, CuTileKernelWrapper) else kernel_fn
    gx = int(grid[0]) if len(grid) >= 1 else 1
    gy = int(grid[1]) if len(grid) >= 2 else 1
    gz = int(grid[2]) if len(grid) >= 3 else 1

    if gx <= 0 or gy <= 0 or gz <= 0:
        msg = f"Invalid launch grid dimensions: {grid}"
        raise ValueError(msg)

    _CTX.active = True
    _CTX.grid_dims = (gx, gy, gz)
    try:
        for bz in range(gz):
            for by in range(gy):
                for bx in range(gx):
                    _CTX.bid_coords = (bx, by, bz)
                    raw_fn(*args)
    finally:
        _CTX.active = False
        _CTX.bid_coords = (0, 0, 0)


# ---------------------------------------------------------------------------
# Static AST Contract Verifier & Pseudo-Tile-IR Analyzer
# ---------------------------------------------------------------------------

FORBIDDEN_SIMT_IDENTIFIERS = frozenset(
    {
        "threadIdx",
        "blockIdx",
        "blockDim",
        "gridDim",
        "__syncthreads",
        "__shfl_sync",
        "__shfl_down_sync",
        "__shfl_xor_sync",
        "warpSize",
        "tl.program_id",
    }
)

FORBIDDEN_HOST_CALLS = frozenset({"open", "input", "eval", "exec", "compile"})

CT_TRANSCENDENTAL_OPS = frozenset({"exp", "log", "sqrt", "rsqrt", "sin", "cos", "atan2"})
CT_REDUCTION_OPS = frozenset({"sum", "max", "min"})
CT_MATMUL_OPS = frozenset({"matmul", "mma"})


@dataclass(slots=True)
class CuTileASTReport:
    """Static verification & resource estimation report for a @ct.kernel function."""

    kernel_name: str
    is_valid: bool
    has_kernel_decorator: bool
    uses_bid: bool
    num_tile_loads: int
    num_tile_stores: int
    num_matmul_ops: int
    num_transcendental_ops: int
    num_reduction_ops: int
    num_loops: int
    constant_params: list[str]
    estimated_sram_bytes: int
    tileiras_available: bool
    violations: list[str] = field(default_factory=list)
    tile_ir_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_cutile_kernel(
    kernel_obj: CuTileKernelWrapper | Callable[..., Any],
    tile_k: int = 128,
    state_dim: int = 32,
    control_dim: int = 16,
) -> CuTileASTReport:
    """Statically analyze a @ct.kernel AST to enforce strict NVIDIA cuTile contracts."""
    if not is_power_of_two(tile_k):
        msg = f"tile_k must be a power of two, got {tile_k}"
        raise ValueError(msg)

    raw_fn = kernel_obj.py_func if isinstance(kernel_obj, CuTileKernelWrapper) else kernel_obj
    kernel_name = getattr(raw_fn, "__name__", "unknown_kernel")
    src = textwrap.dedent(inspect.getsource(raw_fn))
    tree = ast.parse(src)

    func_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not func_defs:
        return CuTileASTReport(
            kernel_name=kernel_name,
            is_valid=False,
            has_kernel_decorator=False,
            uses_bid=False,
            num_tile_loads=0,
            num_tile_stores=0,
            num_matmul_ops=0,
            num_transcendental_ops=0,
            num_reduction_ops=0,
            num_loops=0,
            constant_params=[],
            estimated_sram_bytes=0,
            tileiras_available=shutil.which("tileiras") is not None,
            violations=["No FunctionDef found in kernel source"],
        )

    fdef = func_defs[0]
    violations: list[str] = []

    has_decorator = isinstance(kernel_obj, CuTileKernelWrapper)
    for dec in fdef.decorator_list:
        if (isinstance(dec, ast.Attribute) and dec.attr == "kernel") or (
            isinstance(dec, ast.Name) and dec.id == "kernel"
        ):
            has_decorator = True
    if not has_decorator:
        violations.append(f"Kernel '{kernel_name}' is missing @ct.kernel decorator")

    constant_params: list[str] = []
    for arg in fdef.args.args:
        if arg.annotation is not None:
            ann_str = ast.unparse(arg.annotation)
            if "Constant" in ann_str or "ConstInt" in ann_str or "ConstFloat" in ann_str:
                constant_params.append(arg.arg)

    uses_bid = False
    num_loads = 0
    num_stores = 0
    num_matmuls = 0
    num_trans = 0
    num_reducts = 0
    num_loops = 0
    ir_ops: list[str] = []

    for node in ast.walk(fdef):
        if isinstance(node, (ast.For, ast.While)):
            num_loops += 1
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append(f"Forbidden global/nonlocal mutation in '{kernel_name}'")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_SIMT_IDENTIFIERS:
                violations.append(f"Forbidden SIMT identifier '{node.id}' in '{kernel_name}'")
        elif isinstance(node, ast.Call):
            call_str = ast.unparse(node.func)
            if call_str in FORBIDDEN_SIMT_IDENTIFIERS:
                violations.append(f"Forbidden SIMT call '{call_str}' in '{kernel_name}'")
            if call_str in FORBIDDEN_HOST_CALLS:
                violations.append(f"Forbidden host I/O call '{call_str}' in '{kernel_name}'")

            if call_str in ("ct.bid", "bid"):
                uses_bid = True
                ir_ops.append("  %bid = tile.block_id : i32")
            elif call_str in ("ct.load", "load"):
                num_loads += 1
                ir_ops.append("  %t_in = tile.load_block %mem[%idx] : tensor<TILE_Kxf32>")
            elif call_str in ("ct.store", "store"):
                num_stores += 1
                ir_ops.append("  tile.store_block %t_out -> %mem[%idx] : tensor<TILE_Kxf32>")
            else:
                short_op = call_str.split(".")[-1]
                if short_op in CT_MATMUL_OPS:
                    num_matmuls += 1
                    ir_ops.append("  %mma = tile.mma_tensor_core %A, %B : tensor<TILE_Kxf32>")
                elif short_op in CT_TRANSCENDENTAL_OPS:
                    num_trans += 1
                elif short_op in CT_REDUCTION_OPS:
                    num_reducts += 1
                    ir_ops.append(f"  %red = tile.reduce_{short_op} %tile : f32")

    if not uses_bid:
        violations.append(f"Kernel '{kernel_name}' must use ct.bid() for block-level tile indexing")
    if num_loads == 0:
        violations.append(f"Kernel '{kernel_name}' performs zero ct.load operations")
    if num_stores == 0:
        violations.append(f"Kernel '{kernel_name}' performs zero ct.store operations")

    # Estimate per-block shared SRAM footprint (state tile + control tile + MMA/obstacle cache)
    estimated_sram = (
        tile_k * state_dim * 4 + tile_k * control_dim * 4 + num_matmuls * tile_k * 8 * 4 + 4096
    )
    if estimated_sram >= 96 * 1024:
        violations.append(f"Estimated per-block SRAM ({estimated_sram} B) exceeds 96 KB limit")

    tileiras_found = shutil.which("tileiras") is not None
    ir_header = (
        f'module @{kernel_name}_tile_ir attributes {{arch = "sm_89_sm_100", tile_k = {tile_k}}} {{\n'
        f"  // ConstParams: {', '.join(constant_params) or 'none'}\n"
        f"  // Loads: {num_loads}, Stores: {num_stores}, MMA: {num_matmuls}, "
        f"Transcendentals: {num_trans}, Reductions: {num_reducts}\n"
    )
    ir_body = "\n".join(ir_ops[:12])
    if len(ir_ops) > 12:
        ir_body += f"\n  // ... ({len(ir_ops) - 12} additional tile IR instructions)"
    ir_footer = "\n  tile.return\n}"

    return CuTileASTReport(
        kernel_name=kernel_name,
        is_valid=len(violations) == 0,
        has_kernel_decorator=has_decorator,
        uses_bid=uses_bid,
        num_tile_loads=num_loads,
        num_tile_stores=num_stores,
        num_matmul_ops=num_matmuls,
        num_transcendental_ops=num_trans,
        num_reduction_ops=num_reducts,
        num_loops=num_loops,
        constant_params=constant_params,
        estimated_sram_bytes=estimated_sram,
        tileiras_available=tileiras_found,
        violations=violations,
        tile_ir_summary=ir_header + ir_body + ir_footer,
    )
