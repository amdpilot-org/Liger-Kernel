try:
    import peft  # noqa: F401

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

import torch


def is_peft_available():
    return PEFT_AVAILABLE


def infer_comm_backend():
    """
    Get communication backend name based on the environment.
    """
    if torch.distributed.is_nccl_available():
        # Works for Nvidia
        # TODO: nccl may not work for AMD decices that may require use of rccl.
        return "nccl"
    elif is_npu_available():
        # Use Ascend NPU if available (torch.npu)
        # Ascend is not standard torch backend and requires extension.
        # Assume that it is installed if NPUs are being used in
        # multi device environment.
        return "ascend"
    # XPU (Intel) if available
    elif torch.distributed.distributed_c10d.is_xccl_available():
        return "xccl"
    elif torch.distributed.is_mpi_available():
        # CPU backend, first option
        return "mpi"
    elif torch.distributed.is_gloo_available():
        # CPU backend, backup option
        return "gloo"
    else:
        raise RuntimeError("There is no distributed backend available.")


def infer_device():
    """
    Get current device name based on available devices
    """
    if torch.cuda.is_available():  # Works for both Nvidia and AMD
        return "cuda"
    # Use Ascend NPU if available (torch.npu)
    elif is_npu_available():
        return "npu"
    # XPU (Intel) if available
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


def is_npu_available() -> bool:
    """Detect Ascend NPU availability."""
    try:
        from transformers.utils import is_torch_npu_available

        return is_torch_npu_available()
    except Exception:
        return False


def get_device_arch() -> str:
    """Return the current GPU architecture (GCN name) when running on HIP/ROCm, ``default`` otherwise.

    This dispatches between gfx942 (MI300X) and gfx950 (MI325X/MI355X) tuned arms
    (or ``default`` otherwise). It caches the result so it is cheap to call.

    Returns one of:
        ``"gfx942"``  -- AMD Instinct MI300X accelerator (gfx942)
        ``"gfx950"``  -- AMD Instinct MI325X/MI355X accelerator (gfx950)
        ``"gfx9XX"``  -- Other recent AMD gfx architectures (gfx1100, gfx1150, ...)
        ``"default"`` -- Anything else (CPU-only torch, NVIDIA CUDA, Ascend NPU, etc.)
    """
    global _DEVICE_ARCH_CACHE
    if _DEVICE_ARCH_CACHE is not None:
        return _DEVICE_ARCH_CACHE

    _DEVICE_ARCH_CACHE = "default"
    try:
        if torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None:
            try:
                props = torch.cuda.get_device_properties(0)
                gcn = getattr(props, "gcnArchName", "") or ""
                major = getattr(props, "major", None)
                minor = getattr(props, "minor", None)
                if "gfx950" in gcn:
                    _DEVICE_ARCH_CACHE = "gfx950"
                elif "gfx942" in gcn or (major == 9 and minor == 4):
                    _DEVICE_ARCH_CACHE = "gfx942"
                elif "gfx9" in gcn[:4]:
                    _DEVICE_ARCH_CACHE = "gfx9XX"
            except Exception:
                pass
    except Exception:
        pass
    return _DEVICE_ARCH_CACHE


_DEVICE_ARCH_CACHE = None


def transformers_version_dispatch(
    required_version: str,
    before_fn,
    after_fn,
    before_args: tuple = (),
    after_args: tuple = (),
    before_kwargs: dict = None,
    after_kwargs: dict = None,
):
    """
    Dispatches to different functions based on package version comparison.

    Args:
        required_version: Version to compare against (e.g. "4.48.0")
        before_fn: Function to call if package_version < required_version
        after_fn: Function to call if package_version >= required_version
        before_args: Positional arguments for before_fn
        after_args: Positional arguments for after_fn
        before_kwargs: Keyword arguments for before_fn
        after_kwargs: Keyword arguments for after_fn

    Returns:
        Result from either before_fn or after_fn

    Example:
        >>> rotary_emb = transformers_version_dispatch(
        ...     "4.48.0",
        ...     LlamaRotaryEmbedding,
        ...     LlamaRotaryEmbedding,
        ...     before_args=(head_dim,),
        ...     after_args=(LlamaConfig(head_dim=head_dim),),
        ...     before_kwargs={'device': device},
        ...     after_kwargs={'device': device}
        ... )
    """
    from packaging import version
    from transformers import __version__ as transformers_version

    before_kwargs = before_kwargs or {}
    after_kwargs = after_kwargs or {}

    if version.parse(transformers_version) < version.parse(required_version):
        return before_fn(*before_args, **before_kwargs)
    else:
        return after_fn(*after_args, **after_kwargs)


def get_total_gpu_memory() -> int:
    """Returns total GPU memory in GBs."""
    device = infer_device()
    if device == "cuda":
        return torch.cuda.get_device_properties(0).total_memory // (1024**3)
    elif device == "xpu":
        return torch.xpu.get_device_properties(0).total_memory // (1024**3)
    elif device == "npu":
        return torch.npu.get_device_properties(0).total_memory // (1024**3)
    else:
        raise RuntimeError(f"Unsupported device: {device}")
