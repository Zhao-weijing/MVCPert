import gc

import torch


def cuda_memory_snapshot(device=None):
    if not torch.cuda.is_available():
        return {
            "available": False,
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "max_reserved_mb": 0.0,
        }

    if device is None:
        device = torch.cuda.current_device()
    if isinstance(device, torch.device):
        device = device.index if device.index is not None else torch.cuda.current_device()

    return {
        "available": True,
        "allocated_mb": float(torch.cuda.memory_allocated(device) / 1024 ** 2),
        "reserved_mb": float(torch.cuda.memory_reserved(device) / 1024 ** 2),
        "max_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 1024 ** 2),
        "max_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 1024 ** 2),
    }


def log_cuda_memory(stage, device=None, printer=print):
    snapshot = cuda_memory_snapshot(device=device)
    if not snapshot["available"]:
        printer(f"[CUDA memory] {stage}: CUDA unavailable")
        return snapshot

    printer(
        f"[CUDA memory] {stage}: "
        f"allocated={snapshot['allocated_mb']:.1f}MB, "
        f"reserved={snapshot['reserved_mb']:.1f}MB, "
        f"peak_allocated={snapshot['max_allocated_mb']:.1f}MB, "
        f"peak_reserved={snapshot['max_reserved_mb']:.1f}MB"
    )
    return snapshot


def release_cuda_resources(model=None):
    if model is not None:
        model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
