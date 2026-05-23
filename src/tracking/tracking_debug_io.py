from __future__ import annotations

import csv
from pathlib import Path

from tracking.ball_tracker import TrackedBall


def save_tracking_debug_csv(
    output_path: Path,
    rows: list[dict],
) -> None:
    """
    Save per-frame tracking debug data.

    This helps understand why the tracker loses the ball:
    - no YOLO detections
    - motion filter rejected detections
    - tracker predicted only
    - tracker reset / lost state
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(
            [
                "frame_idx",
                "num_ball_detections",
                "num_motion_filtered_detections",
                "tracked_x",
                "tracked_y",
                "tracked_radius",
                "tracked_confidence",
                "tracked_age",
                "tracked_missed_frames",
                "tracked_is_predicted",
            ]
        )

        for row in rows:
            tracked_ball: TrackedBall | None = row["tracked_ball"]

            if tracked_ball is None:
                writer.writerow(
                    [
                        row["frame_idx"],
                        row["num_ball_detections"],
                        row["num_motion_filtered_detections"],
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                continue

            writer.writerow(
                [
                    row["frame_idx"],
                    row["num_ball_detections"],
                    row["num_motion_filtered_detections"],
                    tracked_ball.x,
                    tracked_ball.y,
                    tracked_ball.radius,
                    tracked_ball.confidence,
                    tracked_ball.age,
                    tracked_ball.missed_frames,
                    int(tracked_ball.is_predicted),
                ]
            )