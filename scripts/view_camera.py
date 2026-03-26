import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a live camera preview using OpenCV. "
            "Press 'q' in the preview window to quit."
        )
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Video device index (default: 0 for /dev/video0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional requested capture width (e.g., 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional requested capture height (e.g., 720).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Optional requested capture FPS (e.g., 30).",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="Jetson Camera Preview",
        help="Preview window title.",
    )
    parser.add_argument(
        "--save-frame",
        type=str,
        default=None,
        help=(
            "Optional output path to save one frame and exit. "
            "Useful in headless sessions."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    try:
        import cv2  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(
            "OpenCV is required. Install it first (e.g., `pip install opencv-python`).",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(
            f"Failed to open camera device index {args.device}. "
            f"Check /dev/video{args.device} and camera permissions.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps is not None:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        f"Streaming from /dev/video{args.device} at "
        f"{actual_width}x{actual_height} @ {actual_fps:.2f} FPS"
    )
    if args.save_frame:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            print("Frame read failed; could not save snapshot.", file=sys.stderr)
            sys.exit(1)
        wrote = cv2.imwrite(args.save_frame, frame)
        cap.release()
        if not wrote:
            print(f"Failed to write snapshot to: {args.save_frame}", file=sys.stderr)
            sys.exit(1)
        print(f"Saved one frame to: {args.save_frame}")
        return

    if not has_display:
        cap.release()
        print(
            "No GUI display detected (DISPLAY/WAYLAND_DISPLAY is not set), "
            "so live preview cannot be shown in this session.",
            file=sys.stderr,
        )
        print(
            "Run this from a desktop session, or use --save-frame <path> "
            "to verify camera capture in headless mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Press 'q' in the preview window to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed; stopping stream.", file=sys.stderr)
                break

            cv2.imshow(args.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
