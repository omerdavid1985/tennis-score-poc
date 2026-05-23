from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

from detection.yolo_ball_detector import YoloBallDetection


@dataclass
class TrackedBall:
    x: float
    y: float
    radius: float
    confidence: float
    age: int
    missed_frames: int
    is_predicted: bool


class BallTracker:
    """
    Kalman-filter-based ball tracker.

    State vector:
        x, y, vx, vy

    Meaning:
        x, y   = ball position in image pixels
        vx, vy = ball velocity in pixels/frame

    The tracker:
        1. Predicts where the ball should be in the next frame.
        2. Matches YOLO detections to that prediction.
        3. Updates the Kalman filter when a valid detection exists.
        4. Keeps predicting for a few frames when detection is missing.
    """

    def __init__(
        self,
        max_distance_px: float = 120.0,
        max_missed_frames: int = 10,
        trajectory_length: int = 60,
    ) -> None:
        self.max_distance_px = max_distance_px
        self.max_missed_frames = max_missed_frames

        self.kf: KalmanFilter | None = None
        self.ball: TrackedBall | None = None

        self.trajectory: deque[tuple[int, int]] = deque(maxlen=trajectory_length)

    def _create_filter(self, detection: YoloBallDetection) -> KalmanFilter:
        """
        Initialize a constant-velocity Kalman filter from the first detection.
        """

        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition:
        # x'  = x + vx
        # y'  = y + vy
        # vx' = vx
        # vy' = vy
        kf.F = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        # Measurement:
        # we directly observe x, y from YOLO.
        kf.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=float,
        )

        # Initial state
        kf.x = np.array(
            [
                [detection.x],
                [detection.y],
                [0.0],
                [0.0],
            ],
            dtype=float,
        )

        # Initial uncertainty
        kf.P *= 500.0

        # Measurement noise.
        # Higher = trust YOLO less.
        kf.R *= 20.0

        # Process noise.
        # Higher = allow faster/unmodeled motion changes.
        kf.Q *= 5.0

        return kf

    def update(
        self,
        detections: list[YoloBallDetection],
    ) -> TrackedBall | None:
        """
        Update tracker with current-frame detections.
        """

        # ---------------------------------------------------------------------
        # No active track: initialize from best detection
        # ---------------------------------------------------------------------

        if self.kf is None or self.ball is None:
            if not detections:
                return None

            best_detection = max(
                detections,
                key=lambda detection: detection.confidence,
            )

            self.kf = self._create_filter(best_detection)

            self.ball = TrackedBall(
                x=float(best_detection.x),
                y=float(best_detection.y),
                radius=float(best_detection.radius),
                confidence=float(best_detection.confidence),
                age=1,
                missed_frames=0,
                is_predicted=False,
            )

            self.trajectory.clear()
            self.trajectory.append((best_detection.x, best_detection.y))

            return self.ball

        # ---------------------------------------------------------------------
        # Predict next position from previous state
        # ---------------------------------------------------------------------

        self.kf.predict()

        predicted_x = float(self.kf.x[0, 0])
        predicted_y = float(self.kf.x[1, 0])

        # --------------------------------------------------------
        # Association step
        #
        # Goal:
        # Choose the detection that best matches the predicted
        # Kalman position and expected motion direction.
        # --------------------------------------------------------

        best_detection = None

        # Lower score = better candidate
        best_score = float("inf")

        # Keep for debug/validation
        best_distance = float("inf")

        # Current estimated ball velocity
        vx = float(self.kf.x[2, 0])
        vy = float(self.kf.x[3, 0])

        predicted_speed = float(np.hypot(vx, vy))

        # --------------------------------------------------------
        # Evaluate all candidate detections
        # --------------------------------------------------------

        for detection in detections:

            # Vector from predicted position to candidate
            dx = detection.x - predicted_x
            dy = detection.y - predicted_y

            distance = float(np.hypot(dx, dy))

            # ----------------------------------------------------
            # Dynamic gating
            #
            # Faster ball motion allows larger jumps.
            # ----------------------------------------------------

            max_allowed_distance = (
                self.max_distance_px
                + 6.0 * predicted_speed
            )

            # Reject impossible jumps
            if distance > max_allowed_distance:
                continue

            # ----------------------------------------------------
            # Motion direction consistency
            #
            # Penalize candidates that move against the
            # predicted trajectory direction.
            # ----------------------------------------------------

            direction_penalty = 0.0

            if predicted_speed > 1.0 and distance > 1.0:

                velocity_direction = (
                    np.array([vx, vy], dtype=float)
                    / predicted_speed
                )

                detection_direction = (
                    np.array([dx, dy], dtype=float)
                    / distance
                )

                # Dot product:
                # +1 -> same direction
                #  0 -> perpendicular
                # -1 -> opposite direction
                direction_alignment = float(
                    np.dot(
                        velocity_direction,
                        detection_direction,
                    )
                )

                direction_penalty = 1.0 - direction_alignment

            # ----------------------------------------------------
            # Final association score
            #
            # Lower score is better.
            #
            # Components:
            # - distance
            # - direction mismatch penalty
            # - confidence bonus
            # ----------------------------------------------------

            score = (
                distance
                + 25.0 * direction_penalty
                - 40.0 * detection.confidence
            )

            # Keep best candidate
            if score < best_score:

                best_score = score
                best_distance = distance
                best_detection = detection
        # ---------------------------------------------------------------------
        # Valid detection: correct Kalman state
        # ---------------------------------------------------------------------

        if best_detection is not None:
            measurement = np.array(
                [
                    [best_detection.x],
                    [best_detection.y],
                ],
                dtype=float,
            )

            self.kf.update(measurement)

            x = float(self.kf.x[0, 0])
            y = float(self.kf.x[1, 0])

            self.ball = TrackedBall(
                x=x,
                y=y,
                radius=float(best_detection.radius),
                confidence=float(best_detection.confidence),
                age=self.ball.age + 1,
                missed_frames=0,
                is_predicted=False,
            )

            self.trajectory.append((int(round(x)), int(round(y))))

            return self.ball

        # ---------------------------------------------------------------------
        # No valid detection: use prediction for a short time
        # ---------------------------------------------------------------------

        missed_frames = self.ball.missed_frames + 1

        if missed_frames > self.max_missed_frames:
            self.kf = None
            self.ball = None
            self.trajectory.clear()
            return None

        self.ball = TrackedBall(
            x=predicted_x,
            y=predicted_y,
            radius=self.ball.radius,
            confidence=0.0,
            age=self.ball.age + 1,
            missed_frames=missed_frames,
            is_predicted=True,
        )

        self.trajectory.append((int(round(predicted_x)), int(round(predicted_y))))

        return self.ball


def draw_tracked_ball(
    frame: np.ndarray,
    tracked_ball: TrackedBall | None,
    trajectory: deque[tuple[int, int]] | None = None,
) -> np.ndarray:
    """
    Draw tracked ball and optional recent trajectory.
    """

    output = frame.copy()

    if tracked_ball is None:
        return output

    # Draw trajectory first so the current ball remains visible on top.
    if trajectory is not None and len(trajectory) > 1:
        points = list(trajectory)

        for i in range(1, len(points)):
            # Older points are thinner, newer points are thicker.
            thickness = max(1, int(6 * i / len(points)))

            cv2.line(
                output,
                points[i - 1],
                points[i],
                (255, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        # Draw small dots along the trajectory so the history is easier to see.
        for i, point in enumerate(points):
            radius = max(2, int(5 * i / len(points)))

            cv2.circle(
                output,
                point,
                radius,
                (255, 0, 0),
                -1,
                cv2.LINE_AA,
            )

    center = (
        int(round(tracked_ball.x)),
        int(round(tracked_ball.y)),
    )

    radius = int(round(tracked_ball.radius))

    color = (0, 0, 255) if not tracked_ball.is_predicted else (255, 0, 255)

    cv2.circle(
        output,
        center,
        radius + 5,
        color,
        3,
        cv2.LINE_AA,
    )

    cv2.circle(
        output,
        center,
        3,
        (255, 255, 255),
        -1,
    )

    label = "Tracked" if not tracked_ball.is_predicted else "Predicted"

    cv2.putText(
        output,
        f"{label} ball | age={tracked_ball.age}",
        (center[0] + 10, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )

    return output