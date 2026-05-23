from __future__ import annotations

import csv
from pathlib import Path

from detection.player_detector import PlayerDetection


def save_player_detections_csv(
    output_path: Path,
    detections_by_frame: dict[int, list[PlayerDetection]],
) -> None:
    """
    Save player detections so YOLO person detection does not need
    to run again on every development iteration.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(
            [
                "frame_idx",
                "x1",
                "y1",
                "x2",
                "y2",
                "confidence",
            ]
        )

        for frame_idx, detections in detections_by_frame.items():
            for detection in detections:
                writer.writerow(
                    [
                        frame_idx,
                        detection.x1,
                        detection.y1,
                        detection.x2,
                        detection.y2,
                        detection.confidence,
                    ]
                )


def load_player_detections_csv(
    input_path: Path,
) -> dict[int, list[PlayerDetection]]:
    """
    Load cached player detections from CSV.
    """

    detections_by_frame: dict[int, list[PlayerDetection]] = {}

    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            frame_idx = int(row["frame_idx"])

            detection = PlayerDetection(
                x1=int(float(row["x1"])),
                y1=int(float(row["y1"])),
                x2=int(float(row["x2"])),
                y2=int(float(row["y2"])),
                confidence=float(row["confidence"]),
            )

            detections_by_frame.setdefault(frame_idx, []).append(detection)

    return detections_by_frame