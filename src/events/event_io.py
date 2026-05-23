from __future__ import annotations

import csv
from pathlib import Path

from events.ball_event_detector import BallEvent


def save_ball_events_csv(
    output_path: Path,
    events: list[BallEvent],
) -> None:
    """
    Save detected ball-event candidates for debugging and tuning.

    This file is intentionally verbose because later tuning will rely on:
    - comparing real bounces
    - comparing false positives
    - analyzing trajectory behavior
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow(
            [
                "frame_idx",
                "event_type",
                "confidence",
                "x",
                "y",
                "court_x",
                "court_y",
                "is_inside_court",
            ]
        )

        for event in events:

            writer.writerow(
                [
                    event.frame_idx,
                    event.event_type,
                    event.confidence,
                    event.x,
                    event.y,
                    event.court_x,
                    event.court_y,
                    int(event.is_inside_court),
                ]
            )