from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class YoloBallDetection:
    x: int
    y: int
    radius: int
    confidence: float


class YoloBallDetector:
    """
    YOLO-based ball detector.

    Step 1:
        Use generic YOLO model as a quick baseline.

    Later:
        Replace yolov8n.pt with a tennis-ball-specific best.pt model.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.15,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> list[YoloBallDetection]:
        detections: list[YoloBallDetection] = []

        results = self.model.predict(
            frame,
            verbose=False,
            conf=self.confidence_threshold,
        )

        if len(results) == 0:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:
            confidence = float(box.conf[0])

            if confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            width = x2 - x1
            height = y2 - y1

            # Generic YOLO probably will NOT classify tennis balls well.
            # For now, keep only very small boxes as ball candidates.
            if width > 80 or height > 80:
                continue

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            radius = int(max(width, height) / 2)

            detections.append(
                YoloBallDetection(
                    x=center_x,
                    y=center_y,
                    radius=max(radius, 3),
                    confidence=confidence,
                )
            )

        return detections


def draw_yolo_ball_detections(
    frame: np.ndarray,
    detections: list[YoloBallDetection],
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

        cv2.putText(
            output,
            f"{detection.confidence:.2f}",
            (detection.x + 8, detection.y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return output