from __future__ import annotations

import csv
from pathlib import Path

from events.shot_detector import ShotEvent


def save_shot_events_csv(
    output_path: Path,
    shots: list[ShotEvent],
) -> None:
    """
    Save shot candidates for debugging and tuning.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(
            [
                "frame_idx",
                "player_role",
                "confidence",
                "x",
                "y",
                "distance_to_player",
            ]
        )

        for shot in shots:
            writer.writerow(
                [
                    shot.frame_idx,
                    shot.player_role,
                    shot.confidence,
                    shot.x,
                    shot.y,
                    shot.distance_to_player,
                ]
            )