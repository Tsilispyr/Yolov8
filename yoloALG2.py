import cv2
import argparse
import numpy as np
import time
from ultralytics import YOLO
import supervision as sv

ZONE_PRESETS = {
    "full":   np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
    "left":   np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1]], dtype=float),
    "right":  np.array([[0.5, 0], [1, 0], [1, 1], [0.5, 1]], dtype=float),
    "top":    np.array([[0, 0], [1, 0], [1, 0.5], [0, 0.5]], dtype=float),
    "bottom": np.array([[0, 0.5], [1, 0.5], [1, 1], [0, 1]], dtype=float),
}


def draw_zone_interactive(frame: np.ndarray) -> np.ndarray | None:
    """
    Interactive zone polygon editor.
      LClick  — add point
      RClick  — undo last point
      M       — full frame (instant confirm)
      Enter   — confirm (>= 3 points)
      Esc     — cancel, keep current zone
    """
    H, W = frame.shape[:2]
    WIN  = "Draw Zone"
    base = frame.copy()
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    def put_label(img, text, y):
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (6, y - th - 4), (10 + tw, y + bl + 2), (0, 0, 0), -1)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)

    while True:
        display = base.copy()

        if len(points) >= 3:
            pts     = np.array(points, dtype=np.int32)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 200, 80))
            display = cv2.addWeighted(display, 0.82, overlay, 0.18, 0)

        for i, pt in enumerate(points):
            cv2.circle(display, pt, 6, (0, 255, 80), -1)
            if i > 0:
                cv2.line(display, points[i - 1], pt, (0, 255, 80), 2)
        if len(points) >= 2:
            cv2.line(display, points[-1], points[0], (0, 255, 80), 1)

        # Controls at bottom, status above them
        hint = "Enter to confirm" if len(points) >= 3 else f"need {3 - len(points)} more"
        put_label(display, "LClick=add   RClick=undo   M=full frame   Enter=confirm   Esc=cancel",
                  H - 12)
        put_label(display, f"Points: {len(points)}   {hint}", H - 46)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('m'), ord('M')):       # M — snap to full frame
            points = [(0, 0), (W - 1, 0), (W - 1, H - 1), (0, H - 1)]
        elif key == 13 and len(points) >= 3:  # Enter — confirm
            cv2.destroyWindow(WIN)
            return np.array(points, dtype=np.int32)
        elif key == 27:                       # Esc — cancel
            cv2.destroyWindow(WIN)
            return None


def make_zone(polygon: np.ndarray):
    zone = sv.PolygonZone(polygon=polygon)
    annotator = sv.PolygonZoneAnnotator(
        zone=zone, color=sv.Color.RED,
        thickness=2, display_in_zone_count=True,
    )
    return zone, annotator


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live detection with tracking")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    parser.add_argument("--conf",  default=0.45,         type=float, help="Confidence threshold")
    parser.add_argument("--model", default="yolov8l.pt", type=str,   help="YOLO model path")
    parser.add_argument(
        "--zone",
        default="full",
        choices=list(ZONE_PRESETS) + ["custom"],
        help="Zone preset or 'custom' to draw on startup (default: full)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    W, H = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    model   = YOLO(args.model)
    tracker = sv.ByteTrack(frame_rate=30)

    box_annotator   = sv.RoundBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5, text_thickness=1,
        text_padding=4, smart_position=True,
    )
    trace_annotator = sv.TraceAnnotator(thickness=1, trace_length=40)

    # Build initial zone polygon
    if args.zone in ZONE_PRESETS:
        zone_polygon = (ZONE_PRESETS[args.zone] * np.array([W, H])).astype(int)
    else:
        # custom — grab one frame and let the user draw
        ret, first_frame = cap.read()
        drawn = draw_zone_interactive(first_frame) if ret else None
        zone_polygon = drawn if drawn is not None \
                       else (ZONE_PRESETS["full"] * np.array([W, H])).astype(int)

    zone, zone_annotator = make_zone(zone_polygon)

    WIN       = "YOLOv8 + Tracker"
    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results    = model(frame, agnostic_nms=True, conf=args.conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        ids = detections.tracker_id if detections.tracker_id is not None \
              else [-1] * len(detections)
        labels = [
            f"#{tid} {model.names[cid]} {conf:.2f}"
            for tid, cid, conf in zip(ids, detections.class_id, detections.confidence)
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        frame = trace_annotator.annotate(scene=frame, detections=detections)

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        fps       = 1.0 / max(time.perf_counter() - prev_time, 1e-6)
        prev_time = time.perf_counter()
        cv2.putText(frame, f"FPS {fps:.1f}  |  {len(detections)} obj  |  [Z] draw zone",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 200, 0), 2)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key in (ord('z'), ord('Z')):
            drawn = draw_zone_interactive(frame.copy())
            if drawn is not None:
                zone, zone_annotator = make_zone(drawn)
                tracker.reset()

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
