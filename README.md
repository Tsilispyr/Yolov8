# YOLOv8 Live Detection

Real-time object detection and tracking via webcam using YOLOv8 and Supervision.

## Scripts

**`yoloALG.py`** - basic detection with polygon zone monitoring.

**`yoloALG2.py`** - full version with ByteTrack object tracking, movement traces, FPS counter, and object count overlay.

## Requirements

```bash
pip install ultralytics supervision opencv-python
```

Tested with: `ultralytics 8.3`, `supervision 0.26`, `opencv 4.12`.

## Usage

```bash
# Basic
python yoloALG.py

# With tracking (recommended)
python yoloALG2.py

# Options
python yoloALG2.py --webcam-resolution 1920 1080 --conf 0.5 --model yolov8n.pt
```

| Argument              | Default       | Description                                  |
| --------------------- | ------------- | -------------------------------------------- |
| `--webcam-resolution` | `1280 720`    | Capture resolution                           |
| `--conf`              | `0.45`        | Detection confidence threshold               |
| `--model`             | `yolov8l.pt`  | YOLO model file (auto-downloaded if missing) |

The model is downloaded automatically by ultralytics on first run. Use `yolov8n.pt` for speed, `yolov8x.pt` for accuracy.

## Controls

`ESC` - quit.

## Zone

The red polygon covers the left half of the frame by default. Edit `ZONE_POLYGON` at the top of either script to change it (values are normalised 0–1).
