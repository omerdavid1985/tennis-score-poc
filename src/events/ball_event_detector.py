from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from tracking.ball_tracker import TrackedBall


@dataclass
class BallEvent:
    """
    Candidate ball event.

    Important:
    This is NOT yet a confirmed tennis event.
    Without player detection, we cannot reliably separate:
    - bounce
    - racket hit
    - net contact
    - tracking artifact
    """

    frame_idx: int
    x: float
    y: float
    event_type: str
    confidence: float


@dataclass
class BallTrackPoint:
    frame_idx: int
    x: float
    y: float
    is_predicted: bool


class BallEventDetector:
    """
    Detects trajectory-change candidates from the tracked ball.

    This module intentionally detects candidates only.
    Later, player detection and court-position filtering will improve
    classification into real bounces and racket hits.
    """

    def __init__(
        self,
        min_speed_px: float = 4.0,
        min_angle_change_deg: float = 35.0,
        min_bounce_vy_px: float = 3.0,
        cooldown_frames: int = 8,
    ) -> None:
        self.min_speed_px = min_speed_px
        self.min_angle_change_deg = min_angle_change_deg
        self.min_bounce_vy_px = min_bounce_vy_px
        self.cooldown_frames = cooldown_frames

        # Use a small temporal window:
        # old point -> middle point -> new point
        self.points: deque[BallTrackPoint] = deque(maxlen=5)

        # Prevent drawing/detecting the same event across many nearby frames.
        self.last_event_frame: int | None = None

    def update(
        self,
        frame_idx: int,
        tracked_ball: TrackedBall | None,
    ) -> BallEvent | None:
        """
        Add current tracked ball and return an event candidate if found.
        """

        if tracked_ball is None:
            return None

        # Predicted points are useful for drawing continuity, but they can
        # create artificial trajectory changes. For event detection, prefer
        # real tracker updates based on actual detections.
        if tracked_ball.is_predicted:
            return None

        self.points.append(
            BallTrackPoint(
                frame_idx=frame_idx,
                x=tracked_ball.x,
                y=tracked_ball.y,
                is_predicted=tracked_ball.is_predicted,
            )
        )

        if len(self.points) < 5:
            return None

        if self.last_event_frame is not None:
            if frame_idx - self.last_event_frame < self.cooldown_frames:
                return None

        p_old = self.points[0]
        p_mid = self.points[2]
        p_new = self.points[4]

        v_before = np.array(
            [
                p_mid.x - p_old.x,
                p_mid.y - p_old.y,
            ],
            dtype=float,
        )

        v_after = np.array(
            [
                p_new.x - p_mid.x,
                p_new.y - p_mid.y,
            ],
            dtype=float,
        )

        speed_before = float(np.linalg.norm(v_before))
        speed_after = float(np.linalg.norm(v_after))

        if speed_before < self.min_speed_px or speed_after < self.min_speed_px:
            return None

        angle_change_deg = self._angle_between_vectors(v_before, v_after)

        if angle_change_deg < self.min_angle_change_deg:
            return None

        event_type = "trajectory_change_candidate"
        confidence = min(angle_change_deg / 120.0, 1.0)

        # In image coordinates, y usually increases downward.
        # A simple bounce-like pattern is:
        # ball moving down -> ball moving up.
        if (
            v_before[1] > self.min_bounce_vy_px
            and v_after[1] < -self.min_bounce_vy_px
        ):
            event_type = "bounce_candidate"

        event = BallEvent(
            frame_idx=p_mid.frame_idx,
            x=p_mid.x,
            y=p_mid.y,
            event_type=event_type,
            confidence=confidence,
        )

        self.last_event_frame = frame_idx
        return event

    @staticmethod
    def _angle_between_vectors(
        v1: np.ndarray,
        v2: np.ndarray,
    ) -> float:
        """
        Return angle change in degrees between two motion vectors.
        """

        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        cos_angle = float(np.dot(v1, v2) / (norm1 * norm2))
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))

        return float(np.degrees(np.arccos(cos_angle)))


def draw_ball_events(
    frame: np.ndarray,
    events: list[BallEvent],
) -> np.ndarray:
    """
    Draw detected event candidates on the video frame.
    """

    output = frame.copy()

    for event in events:
        center = (
            int(round(event.x)),
            int(round(event.y)),
        )

        if event.event_type == "bounce_candidate":
            color = (0, 255, 255)
            label = "Bounce?"
        else:
            color = (255, 255, 0)
            label = "Change?"

        cv2.circle(
            output,
            center,
            14,
            color,
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            f"{label} {event.confidence:.2f}",
            (center[0] + 12, center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return output