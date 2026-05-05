#!/usr/bin/env bash
# =============================================================================
# setup_deepstream.sh — environment check + TensorRT engine export
#
# Run this ONCE on the Jetson before running the DeepStream sweep.
#
# Usage:
#   cd /path/to/energy-inference
#   bash scripts/setup_deepstream.sh
#
# What it does:
#   1. Verifies DeepStream is installed
#   2. Verifies pyds Python bindings are importable
#   3. Verifies GStreamer DeepStream plugins are present
#   4. Exports TensorRT engines for all sweep configs:
#        yolov8n  imgsz=640  fp32
#        yolov8n  imgsz=640  fp16
#        yolov8n  imgsz=320  fp32
#        yolov8n  imgsz=320  fp16
#   5. Verifies each engine loads correctly
#
# Engine outputs: models/trt/yolov8n_{imgsz}_{prec}.engine
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENGINE_DIR="$PROJECT_ROOT/models/trt"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

echo "====================================================="
echo " DeepStream environment check + TRT engine export"
echo "====================================================="
echo "Project root: $PROJECT_ROOT"
echo

# ── 1. DeepStream version ─────────────────────────────────────────────────────
echo "── 1. DeepStream version ────────────────────────────"
if command -v deepstream-app &>/dev/null; then
    DS_VER=$(deepstream-app --version-all 2>&1 | head -3)
    ok "deepstream-app found:\n$DS_VER"
else
    fail "deepstream-app not found. Install NVIDIA DeepStream SDK."
fi
echo

# ── 2. pyds Python bindings ───────────────────────────────────────────────────
echo "── 2. pyds Python bindings ──────────────────────────"
if python3 -c "import pyds; print('pyds version:', getattr(pyds, '__version__', 'n/a'))" 2>/dev/null; then
    ok "pyds importable"
else
    warn "pyds not importable. Trying common install paths..."
    # DeepStream typically installs pyds to /opt/nvidia/deepstream/...
    PYDS_CANDIDATES=(
        /opt/nvidia/deepstream/deepstream/lib
        /opt/nvidia/deepstream/deepstream-6.4/lib
        /opt/nvidia/deepstream/deepstream-6.3/lib
        /opt/nvidia/deepstream/deepstream-7.0/lib
    )
    FOUND=0
    for d in "${PYDS_CANDIDATES[@]}"; do
        if [ -f "$d/pyds.so" ]; then
            echo "  Found pyds.so at $d"
            echo "  Add to PYTHONPATH: export PYTHONPATH=\$PYTHONPATH:$d"
            FOUND=1
            break
        fi
    done
    if [ $FOUND -eq 0 ]; then
        fail "pyds.so not found. Check your DeepStream installation."
    fi
fi
echo

# ── 3. GStreamer DeepStream plugins ───────────────────────────────────────────
echo "── 3. GStreamer DeepStream plugins ──────────────────"
REQUIRED_PLUGINS=("nvjpegdec" "nvvideoconvert" "nvstreammux" "nvinfer")
ALL_OK=1
for plugin in "${REQUIRED_PLUGINS[@]}"; do
    if gst-inspect-1.0 "$plugin" &>/dev/null; then
        ok "$plugin"
    else
        warn "$plugin NOT found in GStreamer"
        ALL_OK=0
    fi
done
if [ $ALL_OK -eq 0 ]; then
    echo "  Hint: export GST_PLUGIN_PATH=\$GST_PLUGIN_PATH:/opt/nvidia/deepstream/deepstream/lib"
    fail "Missing GStreamer DeepStream plugins."
fi
echo

# ── 4. Export TensorRT engines ────────────────────────────────────────────────
echo "── 4. Export TensorRT engines ───────────────────────"
mkdir -p "$ENGINE_DIR"

export_engine() {
    local IMGSZ=$1
    local PREC=$2
    local ENGINE_NAME="yolov8n_${IMGSZ}_${PREC}.engine"
    local ENGINE_PATH="$ENGINE_DIR/$ENGINE_NAME"

    if [ -f "$ENGINE_PATH" ]; then
        ok "$ENGINE_NAME already exists — skipping export"
        return
    fi

    echo "  Exporting $ENGINE_NAME ..."
    HALF_FLAG=""
    if [ "$PREC" = "fp16" ]; then
        HALF_FLAG="half=True"
    fi

    # Export using ultralytics CLI
    # The engine is created next to the .pt file; we then move it.
    python3 - <<PYEOF
from ultralytics import YOLO
import shutil, pathlib

model = YOLO("yolov8n.pt")
half = ("$PREC" == "fp16")
model.export(
    format="engine",
    imgsz=$IMGSZ,
    device=0,
    half=half,
    verbose=False,
)

# Ultralytics saves as yolov8n.engine next to yolov8n.pt
src = pathlib.Path("yolov8n.engine")
if not src.exists():
    # Some versions save with imgsz suffix
    candidates = list(pathlib.Path(".").glob("yolov8n*.engine"))
    if candidates:
        src = candidates[-1]
    else:
        raise FileNotFoundError("Could not find exported .engine file")

dst = pathlib.Path("$ENGINE_PATH")
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.move(str(src), str(dst))
print(f"Saved: {dst}")
PYEOF

    if [ -f "$ENGINE_PATH" ]; then
        ok "$ENGINE_NAME exported → $ENGINE_PATH"
    else
        fail "Export failed for $ENGINE_NAME"
    fi
}

export_engine 640 fp32
export_engine 640 fp16
export_engine 320 fp32
export_engine 320 fp16
echo

# ── 5. Verify engines load ────────────────────────────────────────────────────
echo "── 5. Verify engines load ───────────────────────────"
python3 - <<'PYEOF'
import tensorrt as trt
from pathlib import Path

engine_dir = Path("models/trt")
engines = sorted(engine_dir.glob("*.engine"))
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

for e in engines:
    with open(e, "rb") as f:
        data = f.read()
    engine = runtime.deserialize_cuda_engine(data)
    if engine is None:
        print(f"  FAIL: {e.name}")
    else:
        print(f"  OK  : {e.name}  ({len(data)/1e6:.1f} MB)")
PYEOF

echo
echo "====================================================="
echo " Setup complete. Ready to run the DeepStream sweep."
echo "====================================================="
echo
echo "Next steps:"
echo "  # Single test run (unbounded FPS, 10 seconds):"
echo "  python3 scripts/run_deepstream_bench.py \\"
echo "      --imgsz 640 --precision fp32 --target-fps 0 --duration 10"
echo
echo "  # Full sweep (84 runs × 3 repeats ≈ 4–5 hours):"
echo "  python3 scripts/run_deepstream_sweep.py --grid grids/deepstream_fps_sweep.json"
