import torch
import torchvision.models as models


def get_model(name: str) -> torch.nn.Module:
    """Return a torchvision model by lowercase name."""
    model_name = name.lower()

    if model_name == "resnet18":
        return models.resnet18(weights=None)
    if model_name == "resnet50":
        return models.resnet50(weights=None)

    raise ValueError(f"Unknown model: {name}")

