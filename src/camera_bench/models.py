"""Model loading registry for the camera benchmark."""
from __future__ import annotations

import sys

import torch

SUPPORTED_MODELS: list[str] = [
    "ssdlite320_mobilenet_v3_large",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "retinanet_resnet50_fpn_v2",
    "fcos_resnet50_fpn",
    "yolov8n",
    "yolov8s",
]

_TV_REGISTRY: dict | None = None


def _get_tv_registry() -> dict:
    global _TV_REGISTRY
    if _TV_REGISTRY is not None:
        return _TV_REGISTRY
    from torchvision.models.detection import (  # type: ignore
        FCOS_ResNet50_FPN_Weights,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        FasterRCNN_ResNet50_FPN_V2_Weights,
        RetinaNet_ResNet50_FPN_V2_Weights,
        SSDLite320_MobileNet_V3_Large_Weights,
        fcos_resnet50_fpn,
        fasterrcnn_mobilenet_v3_large_320_fpn,
        fasterrcnn_resnet50_fpn_v2,
        retinanet_resnet50_fpn_v2,
        ssdlite320_mobilenet_v3_large,
    )
    _TV_REGISTRY = {
        "ssdlite320_mobilenet_v3_large": (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
        ),
        "fasterrcnn_mobilenet_v3_large_320_fpn": (
            fasterrcnn_mobilenet_v3_large_320_fpn,
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
        ),
        "fasterrcnn_resnet50_fpn_v2": (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        ),
        "retinanet_resnet50_fpn_v2": (
            retinanet_resnet50_fpn_v2,
            RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
        ),
        "fcos_resnet50_fpn": (
            fcos_resnet50_fpn,
            FCOS_ResNet50_FPN_Weights.DEFAULT,
        ),
    }
    return _TV_REGISTRY


def load_detector(
    model_name: str,
    torch_device: torch.device,
    precision: str = "fp32",
) -> dict:
    """
    Load a detection model and return a descriptor dict::

        {
          "backend":    "torchvision" | "ultralytics",
          "name":       str,
          "model":      nn.Module | YOLO,
          "categories": list | dict,
          "precision":  str,
        }
    """
    name = model_name.lower()
    if name.startswith("yolo"):
        return _load_yolo(name, torch_device, precision)

    registry = _get_tv_registry()
    if name not in registry:
        raise ValueError(
            f"Unsupported model: {model_name!r}. Choose from: {SUPPORTED_MODELS}"
        )
    constructor, weights = registry[name]
    model = constructor(weights=weights).to(torch_device).eval()
    model = _apply_precision(model, precision, torch_device)
    return {
        "backend": "torchvision",
        "name": name,
        "model": model,
        "categories": weights.meta.get("categories", []),
        "precision": precision,
    }


def _load_yolo(name: str, torch_device: torch.device, precision: str) -> dict:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "YOLOv8 requires `ultralytics` (pip install ultralytics)."
        ) from exc

    yolo = YOLO(f"{name}.pt")
    yolo.to("cpu" if torch_device.type == "cpu" else "cuda:0")
    # Do NOT call yolo.model.half() here — Ultralytics fuses conv+bn during the
    # first predict() call, and manually casting weights beforehand causes a dtype
    # mismatch (c10::Half vs float) in fuse_conv_and_bn.  Instead we pass
    # half=True to predict() so Ultralytics casts after fusion.
    return {
        "backend": "ultralytics",
        "name": name,
        "model": yolo,
        "categories": yolo.names,
        "precision": precision,
    }


def _apply_precision(
    model: torch.nn.Module,
    precision: str,
    device: torch.device,
) -> torch.nn.Module:
    if precision == "fp16":
        if device.type != "cuda":
            print(
                "WARNING: fp16 requested but device is not CUDA; staying fp32.",
                file=sys.stderr,
            )
            return model
        return model.half()
    if precision == "bf16":
        return model.to(torch.bfloat16)
    return model
