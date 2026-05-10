"""
Export ssdlite320_mobilenet_v3_large to a TensorRT engine for DeepStream.

Steps
-----
1. Load the pretrained torchvision model.
2. Wrap it to bypass the NMS postprocessing and produce fixed-shape outputs
   (required for static TRT engines).
3. Export to ONNX.
4. Call trtexec to build the TRT engine.

The wrapper exports backbone + head only (no pre/post-processing), accepting
a normalised [0, 1] float RGB tensor of shape [1, 3, 320, 320].  Detection
quality is irrelevant — this is a throughput / energy benchmark.

Usage (on Jetson):
    cd /path/to/energy-inference
    python3 scripts/export_ssdlite_to_trt.py

Output:
    models/trt/ssdlite320_320_fp32.engine
    models/trt/ssdlite320_320_fp32.onnx   (intermediate, kept for debugging)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = PROJECT_ROOT / "models" / "trt"
ENGINE_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH   = ENGINE_DIR / "ssdlite320_320_fp32.onnx"
ENGINE_PATH = ENGINE_DIR / "ssdlite320_320_fp32.engine"
# Opset 17 is supported by both modern torch.onnx and TensorRT 10.x; older opsets
# (e.g. 11) trigger an onnxscript downconversion failure and the file is kept at
# whatever opset torch picked anyway.
OPSET       = 17

# JetPack ships trtexec under /usr/src/tensorrt/bin but does not put it on PATH.
TRTEXEC_CANDIDATES = (
    "trtexec",
    "/usr/src/tensorrt/bin/trtexec",
    "/usr/local/tensorrt/bin/trtexec",
)


def _find_trtexec() -> str:
    for cand in TRTEXEC_CANDIDATES:
        resolved = shutil.which(cand) if "/" not in cand else (cand if Path(cand).is_file() else None)
        if resolved:
            return resolved
    raise FileNotFoundError(
        "trtexec not found. Tried: " + ", ".join(TRTEXEC_CANDIDATES)
    )


class _SSDLiteForTRT(nn.Module):
    """
    Wraps ssdlite320_mobilenet_v3_large so it produces two fixed-shape
    output tensors that TensorRT can serialise without dynamic axes.

    Outputs
    -------
    cls_out  : [1, total_anchors * num_classes]  — flattened class logits
    bbox_out : [1, total_anchors * 4]            — flattened box deltas
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, x: torch.Tensor):
        # backbone returns OrderedDict → convert to list for the head
        features = self.backbone(x)
        if isinstance(features, dict):
            features = list(features.values())

        result = self.head(features)

        # torchvision versions differ: some return dict, some return tuple
        if isinstance(result, dict):
            cls_logits    = result["cls_logits"]
            bbox_regression = result["bbox_regression"]
        else:
            cls_logits, bbox_regression = result

        # Each is a List[Tensor] over feature levels → cat into one tensor
        if isinstance(cls_logits, (list, tuple)):
            cls_out  = torch.cat([t.flatten(1) for t in cls_logits],     dim=1)
            bbox_out = torch.cat([t.flatten(1) for t in bbox_regression], dim=1)
        else:
            cls_out  = cls_logits.flatten(1)
            bbox_out = bbox_regression.flatten(1)

        return cls_out, bbox_out


def export_onnx() -> None:
    if ONNX_PATH.exists():
        print(f"ONNX already exists: {ONNX_PATH} — skipping export")
        return

    print("Loading ssdlite320_mobilenet_v3_large (pretrained) …")
    try:
        import torchvision
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights="DEFAULT"
        )
    except TypeError:
        # older torchvision uses pretrained=True
        import torchvision
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            pretrained=True
        )

    model.eval()
    wrapper = _SSDLiteForTRT(model)
    wrapper.eval()

    dummy = torch.zeros(1, 3, 320, 320)

    # Dry-run to confirm shapes
    with torch.no_grad():
        cls_out, bbox_out = wrapper(dummy)
    print(f"  cls_out  shape: {tuple(cls_out.shape)}")
    print(f"  bbox_out shape: {tuple(bbox_out.shape)}")

    print(f"Exporting ONNX → {ONNX_PATH} …")
    torch.onnx.export(
        wrapper,
        dummy,
        str(ONNX_PATH),
        opset_version=OPSET,
        input_names=["input"],
        output_names=["cls_logits", "bbox_regression"],
        dynamic_axes=None,   # fully static — required for TRT
        do_constant_folding=True,
        verbose=False,
    )
    print(f"ONNX saved: {ONNX_PATH}  ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")


def build_engine() -> None:
    if ENGINE_PATH.exists():
        print(f"Engine already exists: {ENGINE_PATH} — skipping trtexec")
        return

    trtexec = _find_trtexec()

    # TensorRT 10 dropped the legacy --workspace and --fp32 flags:
    #   * workspace is now controlled via --memPoolSize=workspace:<size>
    #   * fp32 is the default precision (fp16/int8 are opt-in)
    cmd = [
        trtexec,
        f"--onnx={ONNX_PATH}",
        f"--saveEngine={ENGINE_PATH}",
        "--memPoolSize=workspace:4096MiB",
        "--verbose",
    ]
    print(f"\nRunning trtexec …")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: trtexec failed (exit code {result.returncode})", file=sys.stderr)
        sys.exit(1)

    print(f"\nEngine saved: {ENGINE_PATH}  ({ENGINE_PATH.stat().st_size / 1e6:.1f} MB)")


def verify_engine() -> None:
    try:
        import tensorrt as trt
    except ImportError:
        print("tensorrt Python package not found — skipping verification")
        return

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    data = ENGINE_PATH.read_bytes()
    engine = runtime.deserialize_cuda_engine(data)
    if engine is None:
        print("ERROR: engine failed to deserialise", file=sys.stderr)
        sys.exit(1)
    print(f"Engine verified OK  ({len(data)/1e6:.1f} MB)")


if __name__ == "__main__":
    print("=" * 60)
    print("SSDLite320 → TensorRT export")
    print("=" * 60)
    export_onnx()
    build_engine()
    verify_engine()
    print("\nDone. Next step:")
    print(f"  python3 scripts/run_deepstream_bench.py \\")
    print(f"      --model ssdlite320 --imgsz 320 --precision fp32 \\")
    print(f"      --target-fps 0 --duration 10 --no-energy \\")
    print(f"      --clean-gst-env  # required when running from a conda env")
