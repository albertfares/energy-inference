import torch


def count_parameters(model: torch.nn.Module) -> int:
    """Count all trainable and non-trainable parameters."""
    return sum(param.numel() for param in model.parameters())


def infer_model_family(model_name: str) -> str:
    """Infer a compact model family label from model name."""
    name = model_name.lower()
    if "resnet" in name:
        return "resnet"
    if "mobilenet" in name:
        return "mobilenet"
    if "efficientnet" in name:
        return "efficientnet"
    if "vit" in name:
        return "vit"
    if "yolo" in name:
        return "yolo"
    return "other"


@torch.no_grad()
def compute_flops(
    model: torch.nn.Module,
    batch: int,
    resolution: int,
    device: torch.device,
) -> tuple[float, float, int]:
    """
    Compute MACs and strict FLOPs with THOP for one input shape.

    Returns:
        macs_total, flops_total_strict, unsupported_ops_count
    """
    try:
        from thop import profile
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics-thop is not installed. Install it with `pip install ultralytics-thop`."
        ) from exc

    x = torch.randn(batch, 3, resolution, resolution, device=device)
    macs_total, _ = profile(model, inputs=(x,), verbose=False)
    macs_total = float(macs_total)
    flops_total_strict = macs_total * 2.0

    # Keep CSV schema backward-compatible.
    unsupported_ops_count = 0

    return macs_total, flops_total_strict, unsupported_ops_count

