from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BallDetection:
    x: int
    y: int
    radius: int
    confidence: float


class BallDetector:
    """
    First simple baseline detector.

    This is NOT the final AI detector.
    Initial goal:
    detect bright/small moving circular objects.

    Later:
    - YOLO
    - TrackNet
    - hybrid tracking
    """

    def __init__(self) -> None:
        pass

    def detect(self, frame: np.ndarray) -> list[BallDetection]:
        detections: list[BallDetection] = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=100,
            param2=18,
            minRadius=2,
            maxRadius=12,
        )

        if circles is None:
            return detections

        circles = np.round(circles[0, :]).astype(int)

        for circle in circles:
            x, y, radius = circle

            detections.append(
                BallDetection(
                    x=x,
                    y=y,
                    radius=radius,
                    confidence=1.0,
                )
            )

        return detections


def draw_ball_detections(
    frame: np.ndarray,
    detections: list[BallDetection],
) -> np.ndarray:
    output = frame.copy()

    for detection in detections:
        cv2.circle(
            output,
            (detection.x, detection.y),
            detection.radius,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.circle(
            output,
            (detection.x, detection.y),
            2,
            (0, 0, 255),
            -1,
        )

    return output