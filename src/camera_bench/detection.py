"""Per-stage detection pipeline with explicit stage timing."""
from __future__ import annotations

import time

import torch
import numpy as np


def _sync(device: torch.device) -> None:
    """Synchronize CUDA stream so that end timestamps are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_staged_detection(
    frame_bgr: np.ndarray,
    detector: dict,
    torch_device: torch.device,
    precision: str,
    score_threshold: float,
    iou_threshold: float,
    max_detections: int,
    yolo_imgsz: int,
    cv2,
) -> tuple[dict[str, tuple[int, int]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run one frame through the detection pipeline with per-stage timing.

    Returns:
        stage_times_ns: {stage: (t_start_ns, t_end_ns)}
        boxes:          (N, 4) float CPU tensor
        labels:         (N,) int64 CPU tensor
        scores:         (N,) float CPU tensor

    Stage notes:
      - torchvision backends: preprocess / infer / postprocess / filter
      - ultralytics (YOLO):   infer_fused (pre+infer+post fused) / filter
        The separate preprocess/infer/postprocess entries are set to (0,0) for
        YOLO to keep the CSV schema uniform. This is noted in summary.json.
    """
    stage_times: dict[str, tuple[int, int]] = {}

    if detector["backend"] == "ultralytics":
        return _run_yolo_staged(
            frame_bgr, detector, torch_device,
            score_threshold, iou_threshold, max_detections, yolo_imgsz,
            stage_times,
        )
    return _run_torchvision_staged(
        frame_bgr, detector, torch_device, precision,
        score_threshold, max_detections, stage_times, cv2,
    )


def _run_torchvision_staged(
    frame_bgr: np.ndarray,
    detector: dict,
    device: torch.device,
    precision: str,
    score_threshold: float,
    max_detections: int,
    stage_times: dict[str, tuple[int, int]],
    cv2,
) -> tuple[dict[str, tuple[int, int]], torch.Tensor, torch.Tensor, torch.Tensor]:
    model = detector["model"]

    # Preprocess
    t0 = time.monotonic_ns()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    if precision == "fp16":
        image = image.half()
    elif precision == "bf16":
        image = image.to(torch.bfloat16)
    image = image.to(device)
    _sync(device)
    t1 = time.monotonic_ns()
    stage_times["preprocess"] = (t0, t1)

    # Inference
    t0 = time.monotonic_ns()
    with torch.no_grad():
        raw_outputs = model([image])
    _sync(device)
    t1 = time.monotonic_ns()
    stage_times["infer"] = (t0, t1)

    # Postprocess (D2H transfer + dict extraction)
    t0 = time.monotonic_ns()
    out = raw_outputs[0]
    boxes = out.get("boxes", torch.empty((0, 4), device=device)).detach().cpu().float()
    labels = out.get("labels", torch.empty((0,), device=device)).detach().cpu()
    scores = out.get("scores", torch.empty((0,), device=device)).detach().cpu().float()
    _sync(device)
    t1 = time.monotonic_ns()
    stage_times["postprocess"] = (t0, t1)

    # Filter
    t0 = time.monotonic_ns()
    keep = scores >= score_threshold
    boxes = boxes[keep][:max_detections]
    labels = labels[keep][:max_detections]
    scores = scores[keep][:max_detections]
    t1 = time.monotonic_ns()
    stage_times["filter"] = (t0, t1)

    # Zero-fill YOLO-only stage for uniform CSV schema
    stage_times["infer_fused"] = (0, 0)

    return stage_times, boxes, labels, scores


def _run_yolo_staged(
    frame_bgr: np.ndarray,
    detector: dict,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
    max_detections: int,
    yolo_imgsz: int,
    stage_times: dict[str, tuple[int, int]],
) -> tuple[dict[str, tuple[int, int]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    YOLO via ultralytics.predict().

    preprocess / infer / postprocess are fused inside predict(); we time the whole
    call as "infer_fused". The individual torchvision-style stages are set to (0,0).
    """
    model = detector["model"]

    t0 = time.monotonic_ns()
    results = model.predict(
        source=frame_bgr,
        conf=score_threshold,
        iou=iou_threshold,
        max_det=max_detections,
        imgsz=yolo_imgsz,
        device="cpu" if device.type == "cpu" else 0,
        verbose=False,
    )
    _sync(device)
    t1 = time.monotonic_ns()
    stage_times["infer_fused"] = (t0, t1)

    # Zero-fill torchvision stages for uniform CSV schema
    stage_times["preprocess"] = (0, 0)
    stage_times["infer"] = (0, 0)
    stage_times["postprocess"] = (0, 0)

    # Filter / extract
    t0 = time.monotonic_ns()
    if not results or results[0].boxes is None or results[0].boxes.xyxy is None:
        boxes = torch.empty((0, 4))
        labels = torch.empty((0,), dtype=torch.int64)
        scores = torch.empty((0,))
    else:
        boxes_obj = results[0].boxes
        boxes = boxes_obj.xyxy.detach().cpu().float()
        labels = boxes_obj.cls.detach().cpu().to(torch.int64)
        scores = boxes_obj.conf.detach().cpu().float()
    t1 = time.monotonic_ns()
    stage_times["filter"] = (t0, t1)

    return stage_times, boxes, labels, scores


def resolve_label_name(label_idx, categories: list | dict) -> str:
    label_int = int(label_idx.item()) if hasattr(label_idx, "item") else int(label_idx)
    if isinstance(categories, dict):
        return str(categories.get(label_int, f"class_{label_int}"))
    if isinstance(categories, (list, tuple)) and 0 <= label_int < len(categories):
        return str(categories[label_int])
    return f"class_{label_int}"


def build_annotated_frame(
    frame_bgr: np.ndarray,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    categories: list | dict,
    cv2,
) -> np.ndarray:
    """Draw bounding boxes on a copy of frame_bgr and return the annotated copy."""
    annotated = frame_bgr.copy()
    for box, label_idx, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        label_name = resolve_label_name(label_idx, categories)
        text = f"{label_name} {float(score.item()):.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 220, 60), 2)
        cv2.putText(
            annotated,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 220, 60),
            2,
            cv2.LINE_AA,
        )
    return annotated
