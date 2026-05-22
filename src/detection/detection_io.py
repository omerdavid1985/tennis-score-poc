from __future__ import annotations

import csv
from pathlib import Path

from detection.yolo_ball_detector import YoloBallDetection


def save_ball_detections_csv(
    output_path: Path,
    detections_by_frame: dict[int, list[YoloBallDetection]],
) -> None:
    """
    Save ball detections to CSV so we can reuse AI results
    without running YOLO again every time.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(
            [
                "frame_idx",
                "x",
                "y",
                "radius",
                "confidence",
            ]
        )

        for frame_idx, detections in detections_by_frame.items():
            for detection in detections:
                writer.writerow(
                    [
                        frame_idx,
                        detection.x,
                        detection.y,
                        detection.radius,
                        detection.confidence,
                    ]
                )


def load_ball_detections_csv(
    input_path: Path,
) -> dict[int, list[YoloBallDetection]]:
    """
    Load cached ball detections from CSV.
    """

    detections_by_frame: dict[int, list[YoloBallDetection]] = {}

    if not input_path.exists():
        return detections_by_frame

    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            frame_idx = int(row["frame_idx"])

            detection = YoloBallDetection(
                x=int(row["x"]),
                y=int(row["y"]),
                radius=int(row["radius"]),
                confidence=float(row["confidence"]),
            )

            detections_by_frame.setdefault(frame_idx, []).append(detection)

    return detections_by_frame