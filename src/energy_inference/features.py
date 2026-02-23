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
) -> tuple[float, int]:
    """
    Compute FLOPs with fvcore for one input shape.

    Returns:
        flops_total, unsupported_ops_count
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "fvcore is not installed. Install it with `pip install fvcore`."
        ) from exc

    x = torch.randn(batch, 3, resolution, resolution, device=device)
    analysis = FlopCountAnalysis(model, x)
    # Explicitly silence unsupported-op warnings to keep CLI output clean.
    analysis.unsupported_ops_warnings(False)
    flops_total = float(analysis.total())

    # Project policy: do not track unsupported ops in CSV for now.
    unsupported_ops_count = 0

    return flops_total, unsupported_ops_count

