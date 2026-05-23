from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class PlayerDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class PlayerDetector:
    """
    Detect tennis players using YOLO person detections.

    This is only the first structural version.
    Later we will:
    - assign near/far player
    - track players over time
    - use players to classify hits vs bounces
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
    ) -> None:
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[PlayerDetection]:
        results = self.model(frame, verbose=False)

        detections: list[PlayerDetection] = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # COCO class 0 = person
                if class_id != 0:
                    continue

                if confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                detections.append(
                    PlayerDetection(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        confidence=confidence,
                    )
                )

        return detections


def draw_player_detections(
    frame: np.ndarray,
    tracked_players,
) -> np.ndarray:
    """
    Draw tracked near/far players.
    """

    output = frame.copy()

    for player in tracked_players:

        if player.role == "Near":
            color = (0, 255, 255)
        else:
            color = (255, 255, 0)

        # Predicted boxes are drawn differently.
        thickness = 1 if player.is_predicted else 2

        cv2.rectangle(
            output,
            (player.x1, player.y1),
            (player.x2, player.y2),
            color,
            thickness,
        )

        label = player.role

        if player.is_predicted:
            label += " (pred)"

        cv2.putText(
            output,
            label,
            (player.x1, player.y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return output