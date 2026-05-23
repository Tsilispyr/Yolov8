import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv

ZONE_POLYGON = np.array([
    [0,   0],
    [0.5, 0],
    [0.5, 1],
    [0,   1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live detection")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    parser.add_argument("--conf", default=0.45, type=float, help="Detection confidence threshold")
    return parser.parse_args()


def main():
    args = parse_arguments()
    W, H = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    model = YOLO("yolov8l.pt")

    box_annotator   = sv.RoundBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=4)

    zone_polygon  = (ZONE_POLYGON * np.array([W, H])).astype(int)
    zone          = sv.PolygonZone(polygon=zone_polygon)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.RED,
        thickness=2,
        display_in_zone_count=True,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results    = model(frame, agnostic_nms=True, conf=args.conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        labels = [
            f"{model.names[cid]} {conf:.2f}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("YOLOv8", frame)
        if cv2.waitKey(1) == 27:
            break
        if cv2.getWindowProperty("YOLOv8", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
