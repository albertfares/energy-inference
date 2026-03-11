import torch
import torchvision.models as models


class _SSDLiteBatchWrapper(torch.nn.Module):
    """Adapt torchvision SSDLite to accept batched tensor input [B,3,H,W]."""

    def __init__(self) -> None:
        super().__init__()
        self.detector = models.detection.ssdlite320_mobilenet_v3_large(weights=None)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape [B,3,H,W] for SSDLite wrapper, got {tuple(x.shape)}."
            )
        images = [img for img in x]
        return self.detector(images)


def get_model(name: str) -> torch.nn.Module:
    """Return a torchvision model by lowercase name."""
    model_name = name.lower()

    if model_name == "resnet18":
        return models.resnet18(weights=None)
    if model_name == "resnet50":
        return models.resnet50(weights=None)
    if model_name in {"mobilenetv3", "mobilenet_v3_large", "mobilenetv3_large"}:
        return models.mobilenet_v3_large(weights=None)
    if model_name in {"mobilenet_v3_small", "mobilenetv3_small"}:
        return models.mobilenet_v3_small(weights=None)
    if model_name in {"ssdlite", "ssdlite320", "ssdlite320_mobilenet_v3_large"}:
        return _SSDLiteBatchWrapper()
    if model_name in {"vit_b_16", "vit", "vit-b-16"}:
        return models.vit_b_16(weights=None)
    if model_name in {"swin_t", "swin", "swin_tiny", "swin-tiny"}:
        return models.swin_t(weights=None)

    raise ValueError(
        "Unknown model: "
        f"{name}. Supported: resnet18, resnet50, mobilenet_v3_large, "
        "mobilenet_v3_small, ssdlite320_mobilenet_v3_large, vit_b_16, swin_t."
    )

