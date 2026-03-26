import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SUPPORTED_DETECTION_MODELS = [
    "ssdlite320_mobilenet_v3_large",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "retinanet_resnet50_fpn_v2",
    "fcos_resnet50_fpn",
    "yolov8n",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run live object detection from a webcam with selectable models."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ssdlite320_mobilenet_v3_large",
        choices=SUPPORTED_DETECTION_MODELS,
        help="Detection model name.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Video device index (default: 0 -> /dev/video0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested capture height.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Requested capture FPS.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.45,
        help="Minimum confidence score for displayed/printed detections.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold used by YOLO NMS.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=8,
        help="Maximum detections per frame after filtering.",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=1.0,
        help="Seconds between terminal detection summaries in headless mode.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if CUDA is available.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show annotated preview window (requires GUI display).",
    )
    parser.add_argument(
        "--serve-mjpeg",
        action="store_true",
        help="Serve annotated live stream as MJPEG over HTTP.",
    )
    parser.add_argument(
        "--mjpeg-host",
        type=str,
        default="127.0.0.1",
        help="MJPEG server bind host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--mjpeg-port",
        type=int,
        default=8080,
        help="MJPEG server port (default: 8080).",
    )
    parser.add_argument(
        "--stream-rtp",
        action="store_true",
        help="Stream annotated video over RTP/H264 using ffmpeg.",
    )
    parser.add_argument(
        "--rtp-host",
        type=str,
        default=None,
        help="RTP destination host/IP (e.g., your Mac IP).",
    )
    parser.add_argument(
        "--rtp-port",
        type=int,
        default=11111,
        help="RTP destination port (default: 11111).",
    )
    parser.add_argument(
        "--rtp-sdp",
        type=str,
        default="jetson.sdp",
        help="Path to write SDP file for ffplay receiver (default: jetson.sdp).",
    )
    parser.add_argument(
        "--enable-energy",
        action="store_true",
        help="Enable live INA3221 power sampling and energy reporting.",
    )
    parser.add_argument(
        "--ina-hz",
        type=int,
        default=1000,
        help="INA3221 sampling frequency in Hz (default: 1000).",
    )
    parser.add_argument(
        "--ina-hw",
        type=str,
        default="all",
        help="INA3221 rails to sample: cpu|gpu|io|both|all (default: all).",
    )
    parser.add_argument(
        "--sampler-exe",
        type=str,
        default="src/energy_inference/tools/sample_ina3221",
        help="Path to INA3221 sampler executable.",
    )
    parser.add_argument(
        "--power-csv",
        type=str,
        default="/tmp/live_detect_power_trace.csv",
        help="Output CSV path for live power samples.",
    )
    parser.add_argument(
        "--energy-print-every",
        type=float,
        default=1.0,
        help="Seconds between live power/energy summaries (default: 1.0).",
    )
    return parser.parse_args()


def _format_detection(label: str, score: float, box) -> str:
    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
    return f"{label}:{score:.2f} [{x1},{y1},{x2},{y2}]"


def _build_annotated_frame(frame_bgr, boxes, labels, scores, categories, cv2):
    annotated = frame_bgr.copy()
    for box, label_idx, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        label_name = _resolve_label_name(label_idx, categories)
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


def _start_mjpeg_server(host: str, port: int, frame_state: dict, frame_lock: threading.Lock):
    class _MJPEGHandler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args) -> None:
            return

        def do_GET(self):  # noqa: N802
            if self.path in {"/", "/index.html"}:
                body = (
                    "<html><head><title>SSDLite Stream</title></head>"
                    "<body style='margin:0;background:#111;'>"
                    "<img src='/stream.mjpg' style='width:100vw;height:auto;'/>"
                    "</body></html>"
                ).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path != "/stream.mjpg":
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()

            try:
                while True:
                    with frame_lock:
                        jpeg = frame_state.get("jpeg")
                    if jpeg is None:
                        time.sleep(0.02)
                        continue

                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                return

    server = ThreadingHTTPServer((host, port), _MJPEGHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _start_rtp_streamer(
    *,
    host: str,
    port: int,
    width: int,
    height: int,
    fps: float,
    sdp_path: str,
):
    target_fps = max(float(fps), 1.0)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{target_fps:.2f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-profile:v",
        "baseline",
        "-g",
        "30",
        "-keyint_min",
        "30",
        "-x264-params",
        "repeat-headers=1:scenecut=0",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rtp",
        "-sdp_file",
        sdp_path,
        "-payload_type",
        "96",
        f"rtp://{host}:{port}?pkt_size=1200",
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)  # noqa: S603


def _safe_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _trapz_energy_j(t_values, p_mw_values):
    if len(t_values) < 2 or len(p_mw_values) < 2:
        return 0.0
    energy_j = 0.0
    for i in range(1, len(t_values)):
        dt = t_values[i] - t_values[i - 1]
        if dt <= 0:
            continue
        p0_w = p_mw_values[i - 1] / 1000.0
        p1_w = p_mw_values[i] / 1000.0
        energy_j += 0.5 * (p0_w + p1_w) * dt
    return energy_j


def _read_live_power_metrics(power_csv_path: str):
    csv_path = Path(power_csv_path)
    if not csv_path.exists():
        return None

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return None

    if len(rows) < 2:
        return None

    has_mono = "mono_ns" in rows[0]
    has_elapsed = "elapsed_ms" in rows[0]
    if not has_mono and not has_elapsed:
        return None

    rail_columns = {
        "cpu": "cpu_power_mW",
        "gpu": "gpu_power_mW",
        "io": "io_power_mW",
    }
    rails_present = [k for k, col in rail_columns.items() if col in rows[0]]
    if not rails_present:
        return None

    t_vals = []
    by_rail = {k: [] for k in rails_present}

    for row in rows:
        if has_mono:
            t_ns = _safe_float(row.get("mono_ns"))
            if t_ns is None:
                continue
            t = t_ns * 1e-9
        else:
            t_ms = _safe_float(row.get("elapsed_ms"))
            if t_ms is None:
                continue
            t = t_ms / 1000.0

        rail_values = {}
        valid_row = True
        for rail in rails_present:
            p = _safe_float(row.get(rail_columns[rail]))
            if p is None:
                valid_row = False
                break
            rail_values[rail] = p
        if not valid_row:
            continue

        t_vals.append(t)
        for rail in rails_present:
            by_rail[rail].append(rail_values[rail])

    if len(t_vals) < 2:
        return None

    latest_w = {rail: by_rail[rail][-1] / 1000.0 for rail in rails_present}
    energy_j = {rail: _trapz_energy_j(t_vals, by_rail[rail]) for rail in rails_present}
    total_power_w = sum(latest_w.values())
    total_energy_j = sum(energy_j.values())
    return {
        "latest_w": latest_w,
        "energy_j": energy_j,
        "total_power_w": total_power_w,
        "total_energy_j": total_energy_j,
        "num_samples": len(t_vals),
    }


def _resolve_label_name(label_idx, categories):
    label_int = int(label_idx.item()) if hasattr(label_idx, "item") else int(label_idx)
    if isinstance(categories, dict):
        return str(categories.get(label_int, f"class_{label_int}"))
    if isinstance(categories, (list, tuple)) and 0 <= label_int < len(categories):
        return str(categories[label_int])
    return f"class_{label_int}"


def _load_detector(args, torch, torchvision_detection, torch_device):
    (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
        RetinaNet_ResNet50_FPN_V2_Weights,
        retinanet_resnet50_fpn_v2,
        FCOS_ResNet50_FPN_Weights,
        fcos_resnet50_fpn,
    ) = torchvision_detection

    model_name = args.model.lower()
    if model_name == "yolov8n":
        try:
            from ultralytics import YOLO
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "YOLOv8 requires `ultralytics` (pip install ultralytics)."
            ) from exc
        yolo = YOLO("yolov8n.pt")
        yolo.to("cpu" if torch_device.type == "cpu" else "cuda:0")
        return {
            "backend": "ultralytics",
            "name": "yolov8n",
            "model": yolo,
            "categories": yolo.names,
        }

    tv_registry = {
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
    if model_name not in tv_registry:
        raise ValueError(f"Unsupported model: {args.model}")

    constructor, weights = tv_registry[model_name]
    model = constructor(weights=weights).to(torch_device).eval()
    categories = weights.meta.get("categories", [])
    return {
        "backend": "torchvision",
        "name": model_name,
        "model": model,
        "categories": categories,
    }


def _run_detection(frame_bgr, detector, args, torch, cv2, torch_device):
    if detector["backend"] == "ultralytics":
        model = detector["model"]
        results = model.predict(
            source=frame_bgr,
            conf=args.score_threshold,
            iou=args.iou_threshold,
            max_det=args.max_detections,
            imgsz=args.yolo_imgsz,
            device="cpu" if torch_device.type == "cpu" else 0,
            verbose=False,
        )
        if not results:
            return (
                torch.empty((0, 4)),
                torch.empty((0,), dtype=torch.int64),
                torch.empty((0,)),
            )
        boxes_obj = results[0].boxes
        if boxes_obj is None or boxes_obj.xyxy is None:
            return (
                torch.empty((0, 4)),
                torch.empty((0,), dtype=torch.int64),
                torch.empty((0,)),
            )
        boxes = boxes_obj.xyxy.detach().cpu()
        labels = boxes_obj.cls.detach().cpu().to(torch.int64)
        scores = boxes_obj.conf.detach().cpu()
        return boxes, labels, scores

    model = detector["model"]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    image = image.to(torch_device)
    with torch.no_grad():
        outputs = model([image])[0]
    boxes = outputs.get("boxes", torch.empty((0, 4), device=torch_device)).detach().cpu()
    labels = outputs.get("labels", torch.empty((0,), device=torch_device)).detach().cpu()
    scores = outputs.get("scores", torch.empty((0,), device=torch_device)).detach().cpu()
    return boxes, labels, scores


def main() -> None:
    args = parse_args()

    try:
        import cv2  # type: ignore
        import torch
        from torchvision.models.detection import (
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
        from energy_inference.tools.INA3221Sampler import INA3221Sampler
    except Exception as exc:  # noqa: BLE001
        print(f"Missing dependency: {exc}", file=sys.stderr)
        print(
            "Install requirements and ensure torch/torchvision/OpenCV are available.",
            file=sys.stderr,
        )
        sys.exit(1)

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    show_window = bool(args.show and has_display)
    serve_mjpeg = bool(args.serve_mjpeg)
    stream_rtp = bool(args.stream_rtp)
    if args.show and not has_display:
        print(
            "GUI display not detected; continuing in headless mode (no preview window).",
            file=sys.stderr,
        )
    if stream_rtp and not args.rtp_host:
        print("--stream-rtp requires --rtp-host (destination IP/hostname).", file=sys.stderr)
        sys.exit(1)

    torch_device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Using torch device: {torch_device}")

    try:
        detector = _load_detector(
            args,
            torch,
            (
                SSDLite320_MobileNet_V3_Large_Weights,
                ssdlite320_mobilenet_v3_large,
                FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                fasterrcnn_mobilenet_v3_large_320_fpn,
                FasterRCNN_ResNet50_FPN_V2_Weights,
                fasterrcnn_resnet50_fpn_v2,
                RetinaNet_ResNet50_FPN_V2_Weights,
                retinanet_resnet50_fpn_v2,
                FCOS_ResNet50_FPN_Weights,
                fcos_resnet50_fpn,
            ),
            torch_device,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load model '{args.model}': {exc}", file=sys.stderr)
        sys.exit(1)
    categories = detector["categories"]
    print(f"Detection model: {detector['name']}")

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(
            f"Failed to open /dev/video{args.device}. Check camera device and permissions.",
            file=sys.stderr,
        )
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera stream: /dev/video{args.device} "
        f"{actual_width}x{actual_height} @ {actual_fps:.2f} FPS"
    )
    mjpeg_server = None
    rtp_proc = None
    power_sampler = None
    power_proc = None
    frame_lock = threading.Lock()
    frame_state: dict[str, bytes | None] = {"jpeg": None}
    if serve_mjpeg:
        mjpeg_server = _start_mjpeg_server(
            args.mjpeg_host, args.mjpeg_port, frame_state, frame_lock
        )
        print(
            f"MJPEG stream available at http://{args.mjpeg_host}:{args.mjpeg_port}/ "
            "(use SSH port-forward if remote)."
        )
    if stream_rtp:
        try:
            rtp_proc = _start_rtp_streamer(
                host=str(args.rtp_host),
                port=args.rtp_port,
                width=actual_width,
                height=actual_height,
                fps=actual_fps if actual_fps > 0 else args.fps,
                sdp_path=args.rtp_sdp,
            )
        except FileNotFoundError:
            cap.release()
            if mjpeg_server is not None:
                mjpeg_server.shutdown()
                mjpeg_server.server_close()
            print("ffmpeg not found. Install ffmpeg to use --stream-rtp.", file=sys.stderr)
            sys.exit(1)

        print(f"RTP stream target: rtp://{args.rtp_host}:{args.rtp_port}")
        print(f"SDP file written to: {args.rtp_sdp}")
        print(
            "On receiver, run: ffplay -protocol_whitelist file,udp,rtp -i "
            f"{args.rtp_sdp}"
        )
    if args.enable_energy:
        power_path = Path(args.power_csv)
        power_path.parent.mkdir(parents=True, exist_ok=True)
        power_sampler = INA3221Sampler(
            exe_candidate=args.sampler_exe,
            hz=args.ina_hz,
            power_csv=str(power_path),
            hw=args.ina_hw,
        )
        power_proc, _ = power_sampler.start()
        if power_proc is None:
            print(
                "Failed to start INA3221 sampler. Continuing without live power data.",
                file=sys.stderr,
            )
            power_sampler = None
        else:
            time.sleep(0.2)
            print(
                f"Live power enabled: hz={args.ina_hz}, hw={args.ina_hw}, csv={args.power_csv}"
            )

    if show_window:
        print("Press 'q' in the preview window to quit.")
    elif not serve_mjpeg and not stream_rtp:
        print("Headless mode active. Press Ctrl+C to stop.")

    last_print_t = 0.0
    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Frame read failed, exiting.", file=sys.stderr)
                break

            frame_idx += 1
            boxes, labels, scores = _run_detection(
                frame_bgr, detector, args, torch, cv2, torch_device
            )

            keep = scores >= args.score_threshold
            boxes = boxes[keep][: args.max_detections]
            labels = labels[keep][: args.max_detections]
            scores = scores[keep][: args.max_detections]

            det_lines: list[str] = []
            for box, label_idx, score in zip(boxes, labels, scores):
                label_name = _resolve_label_name(label_idx, categories)
                det_lines.append(_format_detection(label_name, float(score.item()), box))

            now = time.time()
            print_period = max(args.print_every, 0.1)
            energy_period = max(args.energy_print_every, 0.1)
            should_print = (now - last_print_t) >= min(print_period, energy_period)
            if not show_window and should_print:
                fps_runtime = frame_idx / max(now - t0, 1e-6)
                power_msg = ""
                if args.enable_energy and (now - last_print_t) >= energy_period:
                    metrics = _read_live_power_metrics(args.power_csv)
                    if metrics is not None:
                        latest = metrics["latest_w"]
                        energies = metrics["energy_j"]
                        rails = []
                        for rail in ("cpu", "gpu", "io"):
                            if rail in latest:
                                rails.append(
                                    f"{rail}={latest[rail]:.2f}W/{energies[rail]:.2f}J"
                                )
                        power_msg = (
                            f" | power={metrics['total_power_w']:.2f}W "
                            f"energy={metrics['total_energy_j']:.2f}J "
                            f"({', '.join(rails)})"
                        )
                if det_lines:
                    print(
                        f"[{now:.2f}] fps={fps_runtime:.2f}{power_msg} detections: "
                        + " | ".join(det_lines)
                    )
                else:
                    print(f"[{now:.2f}] fps={fps_runtime:.2f}{power_msg} detections: none")
                last_print_t = now

            annotated = None
            if show_window or serve_mjpeg:
                annotated = _build_annotated_frame(
                    frame_bgr, boxes, labels, scores, categories, cv2
                )
            elif stream_rtp:
                annotated = _build_annotated_frame(
                    frame_bgr, boxes, labels, scores, categories, cv2
                )

            if serve_mjpeg and annotated is not None:
                ok_jpg, jpg_buf = cv2.imencode(
                    ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                if ok_jpg:
                    with frame_lock:
                        frame_state["jpeg"] = jpg_buf.tobytes()

            if stream_rtp and annotated is not None and rtp_proc is not None:
                if rtp_proc.poll() is not None:
                    print("ffmpeg RTP streamer exited unexpectedly.", file=sys.stderr)
                    break
                try:
                    if rtp_proc.stdin is not None:
                        rtp_proc.stdin.write(annotated.tobytes())
                except BrokenPipeError:
                    print("RTP streamer pipe closed unexpectedly.", file=sys.stderr)
                    break

            if show_window and annotated is not None:
                cv2.imshow("SSDLite Live Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if mjpeg_server is not None:
            mjpeg_server.shutdown()
            mjpeg_server.server_close()
        if rtp_proc is not None:
            if rtp_proc.stdin is not None:
                try:
                    rtp_proc.stdin.close()
                except BrokenPipeError:
                    pass
            rtp_proc.terminate()
            try:
                rtp_proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                rtp_proc.kill()
        if power_sampler is not None and power_proc is not None:
            power_sampler.stop(timeout=0.5)
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
