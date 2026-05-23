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

| Argument | Default | Description |
| --- | --- | --- |
| `--webcam-resolution` | `1280 720` | Capture resolution |
| `--conf` | `0.45` | Detection confidence threshold |
| `--model` | `yolov8l.pt` | YOLO model file (auto-downloaded if missing) |
| `--zone` | `full` | Preset: `full` `left` `right` `top` `bottom` `custom` |

The model is downloaded automatically by ultralytics on first run. Use `yolov8n.pt` for speed, `yolov8x.pt` for accuracy.

## Controls

| Key | Action |
| --- | ------ |
| `ESC` | Quit |
| `Z` | Open interactive zone editor on the current frame |
| `X` (window close) | Quit |

## Zone

The red polygon marks a region of interest — objects inside it are counted in real time.

**Presets** (via `--zone`):

```bash
python yoloALG2.py --zone left     # left half
python yoloALG2.py --zone right    # right half
python yoloALG2.py --zone top      # top half
python yoloALG2.py --zone bottom   # bottom half
python yoloALG2.py --zone full     # whole frame (default)
python yoloALG2.py --zone custom   # draw on startup
```

**Interactive editor** — press `Z` at any time while running:

- Left-click to add points
- Right-click to undo the last point
- Enter to confirm (minimum 3 points)
- Esc to cancel and keep the current zone
