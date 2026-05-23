from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from events.ball_event_detector import BallEvent
from tracking.player_tracker import TrackedPlayer


@dataclass
class ShotEvent:
    """
    Candidate racket-hit / shot event.

    This is not yet a confirmed shot.
    It is a first heuristic based on:
    - ball trajectory change
    - proximity to a tracked player
    """

    frame_idx: int
    x: float
    y: float
    player_role: str
    distance_to_player: float
    confidence: float


class ShotDetector:
    """
    Detect shot candidates from ball events and tracked players.
    """

    def __init__(
        self,
        max_distance_to_player_px: float = 180.0,
        cooldown_frames: int = 8,
    ) -> None:
        self.max_distance_to_player_px = max_distance_to_player_px
        self.cooldown_frames = cooldown_frames

        # Avoid duplicate shot detections across nearby frames.
        self.last_shot_frame: int | None = None

    def update(
        self,
        ball_event: BallEvent | None,
        tracked_players: list[TrackedPlayer],
    ) -> ShotEvent | None:
        """
        Convert trajectory-change events near players into shot candidates.
        """

        if ball_event is None:
            return None

        # For now, shots are based on generic trajectory changes.
        # Bounce candidates are handled separately.
        if ball_event.event_type != "trajectory_change_candidate":
            return None

        if self.last_shot_frame is not None:
            if ball_event.frame_idx - self.last_shot_frame < self.cooldown_frames:
                return None

        nearest_player = None
        nearest_distance = float("inf")

        for player in tracked_players:
            distance = self._distance_ball_to_player_box(
                ball_event.x,
                ball_event.y,
                player,
            )

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_player = player

        if nearest_player is None:
            return None

        if nearest_distance > self.max_distance_to_player_px:
            return None

        # Closer ball-player proximity means higher confidence.
        confidence = 1.0 - (
            nearest_distance / self.max_distance_to_player_px
        )

        shot = ShotEvent(
            frame_idx=ball_event.frame_idx,
            x=ball_event.x,
            y=ball_event.y,
            player_role=nearest_player.role,
            distance_to_player=nearest_distance,
            confidence=confidence,
        )

        self.last_shot_frame = ball_event.frame_idx
        return shot

    @staticmethod
    def _distance_ball_to_player_box(
        ball_x: float,
        ball_y: float,
        player: TrackedPlayer,
    ) -> float:
        """
        Distance from ball point to player bounding box.

        If the point is inside the box, distance is zero.
        """

        closest_x = min(
            max(ball_x, player.x1),
            player.x2,
        )

        closest_y = min(
            max(ball_y, player.y1),
            player.y2,
        )

        return float(
            np.hypot(
                ball_x - closest_x,
                ball_y - closest_y,
            )
        )


def draw_shot_events(
    frame: np.ndarray,
    shots: list[ShotEvent],
) -> np.ndarray:
    """
    Draw shot candidates on the video frame.
    """

    output = frame.copy()

    for shot in shots:
        center = (
            int(round(shot.x)),
            int(round(shot.y)),
        )

        cv2.circle(
            output,
            center,
            18,
            (255, 0, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            f"Shot? {shot.player_role} {shot.confidence:.2f}",
            (center[0] + 12, center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output