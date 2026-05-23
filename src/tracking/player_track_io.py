from __future__ import annotations

import csv
from pathlib import Path

from tracking.player_tracker import TrackedPlayer


def save_player_tracks_csv(
    output_path: Path,
    tracks_by_frame: dict[int, list[TrackedPlayer]],
) -> None:
    """
    Save tracked near/far players per frame.

    This is useful later for:
    - hit attribution
    - rally segmentation
    - player motion analysis
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(
            [
                "frame_idx",
                "role",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
                "confidence",
                "missed_frames",
                "is_predicted",
            ]
        )

        for frame_idx, tracked_players in tracks_by_frame.items():
            for player in tracked_players:
                center_x = 0.5 * (player.x1 + player.x2)
                center_y = 0.5 * (player.y1 + player.y2)

                writer.writerow(
                    [
                        frame_idx,
                        player.role,
                        player.x1,
                        player.y1,
                        player.x2,
                        player.y2,
                        center_x,
                        center_y,
                        player.confidence,
                        player.missed_frames,
                        int(player.is_predicted),
                    ]
                )