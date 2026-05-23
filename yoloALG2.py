import cv2
import argparse
import numpy as np
import time
from ultralytics import YOLO
import supervision as sv

ZONE_POLYGON = np.array([
    [0,   0],
    [0.5, 0],
    [0.5, 1],
    [0,   1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live detection with tracking")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    parser.add_argument("--conf",  default=0.45,        type=float, help="Detection confidence threshold")
    parser.add_argument("--model", default="yolov8l.pt", type=str,  help="YOLO model path")
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
        text_scale=0.5, text_thickness=1, text_padding=4,
        smart_position=True,
    )
    trace_annotator = sv.TraceAnnotator(thickness=1, trace_length=40)

    zone_polygon   = (ZONE_POLYGON * np.array([W, H])).astype(int)
    zone           = sv.PolygonZone(polygon=zone_polygon)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.RED,
        thickness=2,
        display_in_zone_count=True,
    )

    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results    = model(frame, agnostic_nms=True, conf=args.conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        tracker_ids = detections.tracker_id if detections.tracker_id is not None \
                      else [-1] * len(detections)
        labels = [
            f"#{tid} {model.names[cid]} {conf:.2f}"
            for tid, cid, conf in zip(tracker_ids, detections.class_id, detections.confidence)
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        frame = trace_annotator.annotate(scene=frame, detections=detections)

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        now  = time.perf_counter()
        fps  = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(
            frame,
            f"FPS {fps:.1f}  |  {len(detections)} obj",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2,
        )

        cv2.imshow("YOLOv8 + Tracker", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
