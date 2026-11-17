"""Hardware capability detector and backend selector for TileMPPI."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess  # nosec B404
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tilemppi.types import BackendKind

ADA_RTX4060_L2_BYTES = 32 * 1024 * 1024  # 33,554,432 bytes (32 MB)


@dataclass(frozen=True, slots=True)
class HardwareCapabilities:
    """Detected host CPU, GPU, cache hierarchy, and backend support."""

    os_name: str
    os_release: str
    python_version: str
    cpu_model: str
    cpu_cores_logical: int
    avx512_supported: bool
    gpu_detected: bool
    gpu_name: str
    compute_capability: tuple[int, int]
    sm_count: int
    vram_total_mb: float
    l2_cache_bytes: int
    l2_cache_mb: float
    dram_bandwidth_gbps: float
    l2_bandwidth_gbps: float
    peak_fp32_tflops: float
    peak_fp16_tflops: float
    peak_fp8_tflops: float
    cutile_module_installed: bool
    cutile_native_hw_ready: bool
    cutile_sim_available: bool
    triton_available: bool
    torch_installed: bool
    torch_cuda_available: bool
    fp8_tensor_core_supported: bool
    available_backends: tuple[BackendKind, ...]
    recommended_backend: BackendKind

    def to_dict(self) -> dict[str, Any]:
        """Serialize hardware profile to a JSON-friendly dictionary."""
        raw = asdict(self)
        raw["available_backends"] = [b.value for b in self.available_backends]
        raw["recommended_backend"] = self.recommended_backend.value
        raw["compute_capability_str"] = (
            f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"
            if self.gpu_detected
            else "cpu_only"
        )
        return raw


def _detect_cpu_info() -> tuple[str, int, bool]:
    """Detect CPU model string, logical thread count, and AVX-512 capability."""
    logical_cores = os.cpu_count() or 1
    cpu_model = platform.processor() or platform.machine() or "Unknown CPU"
    avx512 = False

    if platform.system() == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            try:
                text = cpuinfo.read_text(encoding="utf-8", errors="ignore")
                for line in text.splitlines():
                    if line.lower().startswith("model name") and ":" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
                avx512 = "avx512f" in text.lower()
            except OSError:  # nosec B110
                pass
    elif platform.system() == "Windows":
        env_ident = os.environ.get("PROCESSOR_IDENTIFIER", "")
        if env_ident:
            cpu_model = env_ident
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\Description\System\CentralProcessor\0",
            ) as key:
                val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if isinstance(val, str) and val.strip():
                    cpu_model = val.strip()
        except Exception:  # nosec B110
            pass

    lower_model = cpu_model.lower()
    if any(
        token in lower_model
        for token in ("7840hs", "7940hs", "7735", "zen 4", "zen 5", "ryzen 7 7", "ryzen 9 7")
    ):
        avx512 = True

    return cpu_model, logical_cores, avx512


def _probe_nvidia_smi() -> tuple[bool, str, tuple[int, int], float]:
    """Fallback GPU detection via nvidia-smi when CUDA Torch is not active."""
    smi_path = shutil.which("nvidia-smi")
    if not smi_path:
        return False, "No GPU Detected (CPU Mode)", (0, 0), 0.0
    try:
        proc = subprocess.run(  # nosec B603
            [
                smi_path,
                "--query-gpu=name,compute_cap,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=3.0,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first_line = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 3:
                name = parts[0]
                cc_parts = parts[1].split(".")
                major = int(cc_parts[0]) if cc_parts else 0
                minor = int(cc_parts[1]) if len(cc_parts) > 1 else 0
                vram_mb = float(parts[2])
                return True, name, (major, minor), vram_mb
    except Exception:
        pass
    return False, "No GPU Detected (CPU Mode)", (0, 0), 0.0


def _infer_cache_and_roofline(
    gpu_detected: bool,
    gpu_name: str,
    cc: tuple[int, int],
    reported_l2_bytes: int,
) -> tuple[int, float, float, float, float, float]:
    """Return (l2_bytes, dram_gbps, l2_gbps, fp32_tflops, fp16_tflops, fp8_tflops)."""
    if not gpu_detected:
        return (
            ADA_RTX4060_L2_BYTES,
            64.0,
            512.0,
            0.8,
            1.6,
            1.6,
        )

    l2_bytes = reported_l2_bytes
    lower_name = gpu_name.lower()

    if l2_bytes <= 0:
        if cc >= (10, 0):
            l2_bytes = 64 * 1024 * 1024
        elif cc == (8, 9) or "4060" in lower_name or "4070" in lower_name:
            l2_bytes = ADA_RTX4060_L2_BYTES
        elif cc >= (8, 0):
            l2_bytes = 16 * 1024 * 1024
        else:
            l2_bytes = 6 * 1024 * 1024

    if "4060" in lower_name or cc == (8, 9):
        return (
            l2_bytes,
            256.0,  # 128-bit GDDR6 @ 16 Gbps
            1650.0,  # 32 MB Ada L2 on-chip bandwidth
            15.1,  # Peak FP32 TFLOPS
            60.4,  # Tensor Core FP16 TFLOPS
            120.8,  # 4th-Gen Tensor Core FP8 TOPS/TFLOPS
        )
    if cc >= (10, 0):
        return (l2_bytes, 896.0, 4200.0, 45.0, 180.0, 360.0)
    return (l2_bytes, 320.0, 1200.0, 12.0, 48.0, 48.0)


_CACHED_CAPABILITIES: HardwareCapabilities | None = None


def detect_hardware(force_refresh: bool = False) -> HardwareCapabilities:
    """Probe CPU, GPU, L2 cache size, and available execution backends."""
    global _CACHED_CAPABILITIES
    if _CACHED_CAPABILITIES is not None and not force_refresh:
        return _CACHED_CAPABILITIES

    cpu_model, logical_cores, avx512 = _detect_cpu_info()

    torch_installed = False
    torch_cuda_available = False
    gpu_detected = False
    gpu_name = "No GPU Detected (CPU Mode)"
    cc = (0, 0)
    sm_count = 0
    vram_total_mb = 0.0
    reported_l2_bytes = 0

    try:
        import torch

        torch_installed = True
        if torch.cuda.is_available():
            torch_cuda_available = True
            gpu_detected = True
            props = torch.cuda.get_device_properties(0)
            gpu_name = str(props.name)
            cc = (int(props.major), int(props.minor))
            sm_count = int(getattr(props, "multi_processor_count", 24))
            vram_total_mb = float(props.total_memory) / (1024.0 * 1024.0)
            reported_l2_bytes = int(getattr(props, "L2_cache_size", 0) or 0)
    except Exception:
        pass

    if not gpu_detected:
        smi_found, smi_name, smi_cc, smi_vram = _probe_nvidia_smi()
        if smi_found:
            gpu_detected = True
            gpu_name = smi_name
            cc = smi_cc
            sm_count = 24 if smi_cc == (8, 9) else 16
            vram_total_mb = smi_vram

    cutile_module_installed = False
    try:
        import cuda.tile  # noqa: F401

        cutile_module_installed = True
    except Exception:
        cutile_module_installed = False

    cutile_native_hw_ready = bool(cutile_module_installed and gpu_detected and cc >= (10, 0))

    triton_available = False
    if torch_cuda_available:
        try:
            import triton  # noqa: F401

            triton_available = True
        except Exception:
            triton_available = False

    l2_bytes, dram_gbps, l2_gbps, fp32_tf, fp16_tf, fp8_tf = _infer_cache_and_roofline(
        gpu_detected=gpu_detected,
        gpu_name=gpu_name,
        cc=cc,
        reported_l2_bytes=reported_l2_bytes,
    )

    fp8_supported = bool(gpu_detected and cc >= (8, 9))

    backends: list[BackendKind] = []
    if cutile_native_hw_ready:
        backends.append(BackendKind.CUTILE)
    backends.append(BackendKind.CUTILE_SIM)
    if triton_available:
        backends.append(BackendKind.TRITON)
    if torch_cuda_available:
        backends.append(BackendKind.TORCH_GPU)
    if torch_installed:
        backends.append(BackendKind.TORCH_CPU)
    backends.append(BackendKind.NUMPY_CPU)

    if cutile_native_hw_ready:
        recommended = BackendKind.CUTILE
    elif triton_available:
        recommended = BackendKind.TRITON
    elif torch_cuda_available:
        recommended = BackendKind.TORCH_GPU
    else:
        recommended = BackendKind.NUMPY_CPU

    _CACHED_CAPABILITIES = HardwareCapabilities(
        os_name=platform.system(),
        os_release=platform.release(),
        python_version=platform.python_version(),
        cpu_model=cpu_model,
        cpu_cores_logical=logical_cores,
        avx512_supported=avx512,
        gpu_detected=gpu_detected,
        gpu_name=gpu_name,
        compute_capability=cc,
        sm_count=sm_count,
        vram_total_mb=round(vram_total_mb, 2),
        l2_cache_bytes=l2_bytes,
        l2_cache_mb=round(l2_bytes / (1024.0 * 1024.0), 2),
        dram_bandwidth_gbps=dram_gbps,
        l2_bandwidth_gbps=l2_gbps,
        peak_fp32_tflops=fp32_tf,
        peak_fp16_tflops=fp16_tf,
        peak_fp8_tflops=fp8_tf,
        cutile_module_installed=cutile_module_installed,
        cutile_native_hw_ready=cutile_native_hw_ready,
        cutile_sim_available=True,
        triton_available=triton_available,
        torch_installed=torch_installed,
        torch_cuda_available=torch_cuda_available,
        fp8_tensor_core_supported=fp8_supported,
        available_backends=tuple(backends),
        recommended_backend=recommended,
    )
    return _CACHED_CAPABILITIES


class BackendSelector:
    """Resolves requested BackendKind against detected hardware capabilities."""

    @staticmethod
    def resolve(
        requested: BackendKind | str = BackendKind.AUTO,
        capabilities: HardwareCapabilities | None = None,
    ) -> BackendKind:
        """Select concrete execution backend according to priority rules.

        Priority for AUTO:
          cutile (native HW) -> triton (CUDA) -> torch_gpu (CUDA) -> numpy_cpu
        """
        caps = capabilities or detect_hardware()
        kind = BackendKind(requested) if isinstance(requested, str) else requested

        if kind == BackendKind.AUTO:
            return caps.recommended_backend

        if kind == BackendKind.CUTILE:
            if caps.cutile_native_hw_ready:
                return BackendKind.CUTILE
            return BackendKind.CUTILE_SIM

        if kind == BackendKind.CUTILE_SIM:
            return BackendKind.CUTILE_SIM

        if kind == BackendKind.TRITON:
            if caps.triton_available:
                return BackendKind.TRITON
            if caps.torch_cuda_available:
                return BackendKind.TORCH_GPU
            return BackendKind.NUMPY_CPU

        if kind == BackendKind.TORCH_GPU:
            if caps.torch_cuda_available:
                return BackendKind.TORCH_GPU
            if caps.torch_installed:
                return BackendKind.TORCH_CPU
            return BackendKind.NUMPY_CPU

        if kind == BackendKind.TORCH_CPU:
            if caps.torch_installed:
                return BackendKind.TORCH_CPU
            return BackendKind.NUMPY_CPU

        return BackendKind.NUMPY_CPU
