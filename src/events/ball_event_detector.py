from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from geometry.homography import (
    COURT_LENGTH_M,
    SINGLES_WIDTH_M,
    image_point_to_court,
)
from tracking.ball_tracker import TrackedBall


@dataclass
class BallEvent:
    """
    Candidate ball event.

    This is not yet a confirmed tennis event.
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

    # Court coordinates in meters.
    court_x: float | None = None
    court_y: float | None = None
    is_inside_court: bool = False


@dataclass
class BallTrackPoint:
    """
    One tracked ball point used by the event detector.
    """

    frame_idx: int
    x: float
    y: float
    is_predicted: bool


class BallEventDetector:
    """
    Detects ball trajectory event candidates.

    Current goal:
    - detect possible bounces
    - detect generic sharp trajectory changes

    Important:
    This detector produces candidates only.
    Later we will use player detection and better court logic
    to classify real bounces vs racket hits.
    """

    def __init__(
        self,
        min_speed_px: float = 3.0,
        min_angle_change_deg: float = 45.0,
        min_bounce_vy_px: float = 1.5,
        cooldown_frames: int = 10,
        court_margin_m: float = 1.0,
    ) -> None:
        self.min_speed_px = min_speed_px
        self.min_angle_change_deg = min_angle_change_deg
        self.min_bounce_vy_px = min_bounce_vy_px
        self.cooldown_frames = cooldown_frames

        # Allow tolerance around the court because:
        # - manual calibration is imperfect
        # - tracker still has noise
        # - ball center is not exactly the bounce contact point
        self.court_margin_m = court_margin_m

        # Keep a short history:
        # old point -> middle point -> new point
        #
        # A 7-frame window is slightly smoother than 5 frames and helps
        # catch bounces that appear as smooth direction changes.
        self.points: deque[BallTrackPoint] = deque(maxlen=7)

        # Prevent repeated detections for the same physical event.
        self.last_event_frame: int | None = None

    def update(
        self,
        frame_idx: int,
        tracked_ball: TrackedBall | None,
        image_to_court_h: np.ndarray | None = None,
    ) -> BallEvent | None:
        """
        Add current tracked ball and return an event candidate if found.
        """

        if tracked_ball is None:
            return None

        # Predicted points may create artificial direction changes.
        # For event detection, use only points supported by real detections.
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

        if len(self.points) < 7:
            return None

        if self.last_event_frame is not None:
            if frame_idx - self.last_event_frame < self.cooldown_frames:
                return None

        p_old = self.points[0]
        p_mid = self.points[3]
        p_new = self.points[6]

        # Motion before and after the middle point.
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

        # Ignore very slow motion because tiny tracker jitter can look
        # like a trajectory event.
        if speed_before < self.min_speed_px or speed_after < self.min_speed_px:
            return None

        angle_change_deg = self._angle_between_vectors(v_before, v_after)

        # --------------------------------------------------------
        # First priority: bounce-like vertical turning point
        #
        # In image coordinates, y increases downward.
        #
        # Bounce candidate pattern:
        # - before event: ball moves downward  -> positive y velocity
        # - after event:  ball moves upward    -> negative y velocity
        #
        # This is checked before angle-change because real bounces can
        # look smooth and may not pass a sharp angle threshold.
        # --------------------------------------------------------

        is_vertical_turning_point = (
            v_before[1] > self.min_bounce_vy_px
            and v_after[1] < -self.min_bounce_vy_px
        )

        if is_vertical_turning_point:
            event_type = "bounce_candidate"

            confidence = min(
                (abs(v_before[1]) + abs(v_after[1])) / 40.0,
                1.0,
            )

        else:
            # ----------------------------------------------------
            # Fallback: generic sharp trajectory change
            #
            # This may represent:
            # - racket hit
            # - net contact
            # - unusual bounce
            # - tracking artifact
            # ----------------------------------------------------

            if angle_change_deg < self.min_angle_change_deg:
                return None

            event_type = "trajectory_change_candidate"
            confidence = min(angle_change_deg / 120.0, 1.0)

        court_x = None
        court_y = None
        is_inside_court = False

        if image_to_court_h is not None:
            court_x, court_y = image_point_to_court(
                (p_mid.x, p_mid.y),
                image_to_court_h,
            )

            # Allow margin because calibration and tracking are not perfect.
            is_inside_court = (
                -self.court_margin_m
                <= court_x
                <= SINGLES_WIDTH_M + self.court_margin_m
                and -self.court_margin_m
                <= court_y
                <= COURT_LENGTH_M + self.court_margin_m
            )

            # Bounce candidates should happen on/near the court.
            # This will not remove racket hits inside the court yet,
            # but it does remove impossible off-court bounces.
            if event_type == "bounce_candidate" and not is_inside_court:
                return None

        event = BallEvent(
            frame_idx=p_mid.frame_idx,
            x=p_mid.x,
            y=p_mid.y,
            event_type=event_type,
            confidence=confidence,
            court_x=court_x,
            court_y=court_y,
            is_inside_court=is_inside_court,
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