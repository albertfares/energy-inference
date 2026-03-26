# Jetson to Mac Live Detection Streaming

This is the working setup for live `ssdlite` detection from a headless Jetson to a Mac.

You can choose the detection model with `--model`.
Supported options:

- `ssdlite320_mobilenet_v3_large`
- `fasterrcnn_mobilenet_v3_large_320_fpn`
- `fasterrcnn_resnet50_fpn_v2`
- `retinanet_resnet50_fpn_v2`
- `fcos_resnet50_fpn`
- `yolov8n`

## Prerequisites

- Jetson has camera at `/dev/video0`
- Mac has `ffplay` available (`brew install ffmpeg`)
- Script exists: `scripts/live_detect_ssdlite.py`

## 1) Find IP addresses

On Jetson:

```bash
hostname -I
```

Use the Jetson LAN IP (example: `128.179.129.19`).

On Mac:

```bash
ipconfig getifaddr en0
```

If empty, try:

```bash
ipconfig getifaddr en1
```

Use the Mac IP (example: `128.179.197.5`).

## 2) Start sender on Jetson

Run this on Jetson (replace `--rtp-host` with your Mac IP):

```bash
python scripts/live_detect_ssdlite.py \
  --model ssdlite320_mobilenet_v3_large \
  --device 0 \
  --stream-rtp \
  --rtp-host 128.179.197.5 \
  --rtp-port 11111 \
  --rtp-sdp /tmp/jetson.sdp
```

With live power and cumulative energy reporting in terminal:

```bash
python scripts/live_detect_ssdlite.py \
  --model ssdlite320_mobilenet_v3_large \
  --device 0 \
  --stream-rtp \
  --rtp-host 128.179.197.5 \
  --rtp-port 11111 \
  --rtp-sdp /tmp/jetson.sdp \
  --enable-energy \
  --ina-hz 1000 \
  --ina-hw all
```

Example live log line:

```text
[1742311220.55] fps=12.84 | power=9.31W energy=22.41J (cpu=2.11W/5.34J, gpu=6.72W/15.79J, io=0.48W/1.28J) detections: person:0.92 [101,54,304,463]
```

YOLO example with extra hyperparameters:

```bash
python scripts/live_detect_ssdlite.py \
  --model yolov8n \
  --device 0 \
  --score-threshold 0.35 \
  --iou-threshold 0.45 \
  --yolo-imgsz 640 \
  --stream-rtp \
  --rtp-host 128.179.197.5 \
  --rtp-port 11111 \
  --rtp-sdp /tmp/jetson.sdp
```

## 3) Copy SDP from Jetson to Mac

Run this on Mac (replace username/IP if needed):

```bash
scp albertfares@128.179.129.19:/tmp/jetson.sdp /tmp/jetson.sdp
```

Normalize SDP bind address on Mac:

```bash
sed -i '' 's/^c=IN IP4 .*/c=IN IP4 0.0.0.0/' /tmp/jetson.sdp
```

## 4) View stream on Mac

```bash
ffplay -protocol_whitelist file,udp,rtp -analyzeduration 2000000 -probesize 10000000 /tmp/jetson.sdp
```

## Stop

- Jetson sender: `Ctrl+C`
- Mac `ffplay`: `q` (or `Ctrl+C`)

## Quick Troubleshooting

- `zsh: command not found: ffplay`
  - Install FFmpeg on Mac: `brew install ffmpeg`

- `bind failed: Can't assign requested address`
  - Run the `sed` command above to force `c=IN IP4 0.0.0.0`

- `non-existing PPS 0 referenced` or `no frame`
  - Confirm Jetson sender uses the correct Mac IP in `--rtp-host`
  - Re-copy fresh `/tmp/jetson.sdp` after restarting sender

- No video but no errors
  - Check packets on Mac:
    ```bash
    tcpdump -ni any udp port 11111
    ```
  - If no packets, destination IP/port or network path is wrong.
