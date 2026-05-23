from __future__ import annotations

import csv
from pathlib import Path

from tracking.ball_tracker import TrackedBall


def save_tracked_trajectory_csv(
    output_path: Path,
    trajectory_by_frame: dict[int, TrackedBall | None],
) -> None:
    """
    Save the final tracked ball trajectory.

    This is different from YOLO detections:
    - YOLO CSV stores raw detector candidates.
    - Trajectory CSV stores the tracker output after Kalman filtering.
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
                "age",
                "missed_frames",
                "is_predicted",
            ]
        )

        for frame_idx, tracked_ball in trajectory_by_frame.items():
            if tracked_ball is None:
                writer.writerow(
                    [
                        frame_idx,
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
                    frame_idx,
                    tracked_ball.x,
                    tracked_ball.y,
                    tracked_ball.radius,
                    tracked_ball.confidence,
                    tracked_ball.age,
                    tracked_ball.missed_frames,
                    int(tracked_ball.is_predicted),
                ]
            )