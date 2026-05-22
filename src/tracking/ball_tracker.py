from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from detection.ball_detector import BallDetection


@dataclass
class TrackedBall:
    x: float
    y: float
    radius: float
    confidence: float
    age: int
    missed_frames: int


class BallTracker:
    """
    Very first simple ball tracker.

    Current strategy:
    - keep only one tracked ball
    - nearest-neighbor association
    - survive short missed detections

    Future upgrades:
    - Kalman filter
    - velocity prediction
    - confidence fusion
    - trajectory validation
    """

    def __init__(
        self,
        max_distance_px: float = 80.0,
        max_missed_frames: int = 8,
    ) -> None:
        self.max_distance_px = max_distance_px
        self.max_missed_frames = max_missed_frames

        self.ball: TrackedBall | None = None

    def update(
        self,
        detections: list[BallDetection],
    ) -> TrackedBall | None:

        # ---------------------------------------------------------------------
        # No existing track:
        # initialize from first detection
        # ---------------------------------------------------------------------

        if self.ball is None:
            if len(detections) == 0:
                return None

            best = detections[0]

            self.ball = TrackedBall(
                x=best.x,
                y=best.y,
                radius=best.radius,
                confidence=best.confidence,
                age=1,
                missed_frames=0,
            )

            return self.ball

        # ---------------------------------------------------------------------
        # Existing track:
        # find nearest detection
        # ---------------------------------------------------------------------

        best_detection = None
        best_distance = float("inf")

        for detection in detections:
            distance = np.hypot(
                detection.x - self.ball.x,
                detection.y - self.ball.y,
            )

            if distance < best_distance:
                best_distance = distance
                best_detection = detection

        # ---------------------------------------------------------------------
        # Valid match found
        # ---------------------------------------------------------------------

        if (
            best_detection is not None
            and best_distance < self.max_distance_px
        ):
            self.ball.x = best_detection.x
            self.ball.y = best_detection.y
            self.ball.radius = best_detection.radius
            self.ball.confidence = best_detection.confidence

            self.ball.age += 1
            self.ball.missed_frames = 0

            return self.ball

        # ---------------------------------------------------------------------
        # No valid match:
        # keep track alive temporarily
        # ---------------------------------------------------------------------

        self.ball.missed_frames += 1

        if self.ball.missed_frames > self.max_missed_frames:
            self.ball = None

        return self.ball


def draw_tracked_ball(
    frame: np.ndarray,
    tracked_ball: TrackedBall | None,
) -> np.ndarray:
    output = frame.copy()

    if tracked_ball is None:
        return output

    center = (
        int(round(tracked_ball.x)),
        int(round(tracked_ball.y)),
    )

    radius = int(round(tracked_ball.radius))

    # Main tracked ball
    cv2.circle(
        output,
        center,
        radius + 4,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )

    # Center point
    cv2.circle(
        output,
        center,
        3,
        (255, 255, 255),
        -1,
    )

    # Debug text
    cv2.putText(
        output,
        f"Tracked ball | age={tracked_ball.age}",
        (center[0] + 10, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return output