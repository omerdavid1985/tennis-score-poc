from __future__ import annotations

import cv2
import numpy as np

from detection.yolo_ball_detector import YoloBallDetection


class MotionFilter:
    """
    Frame-difference motion filter.

    Purpose:
        YOLO may detect several ball-like candidates.
        This filter keeps only detections that overlap with moving pixels.

    This helps reject:
        - static court/fence false positives
        - fixed bright objects
        - false detections that do not move between frames
    """

    def __init__(
        self,
        threshold: int = 25,
        min_motion_pixels: int = 8,
        roi_padding_px: int = 8,
    ) -> None:
        self.threshold = threshold
        self.min_motion_pixels = min_motion_pixels
        self.roi_padding_px = roi_padding_px

        self.previous_gray: np.ndarray | None = None

    def build_motion_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Build a binary mask where white pixels represent motion.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.previous_gray is None:
            self.previous_gray = gray
            return np.zeros_like(gray)

        frame_diff = cv2.absdiff(gray, self.previous_gray)

        _, motion_mask = cv2.threshold(
            frame_diff,
            self.threshold,
            255,
            cv2.THRESH_BINARY,
        )

        # Clean small noise.
        kernel = np.ones((3, 3), dtype=np.uint8)

        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        motion_mask = cv2.dilate(
            motion_mask,
            kernel,
            iterations=1,
        )

        self.previous_gray = gray

        return motion_mask

    def filter_detections(
        self,
        detections: list[YoloBallDetection],
        motion_mask: np.ndarray,
    ) -> list[YoloBallDetection]:
        """
        Keep only detections that overlap with enough moving pixels.
        """

        filtered: list[YoloBallDetection] = []

        height, width = motion_mask.shape[:2]

        for detection in detections:
            radius = max(detection.radius, 3)

            x1 = max(0, detection.x - radius - self.roi_padding_px)
            y1 = max(0, detection.y - radius - self.roi_padding_px)
            x2 = min(width, detection.x + radius + self.roi_padding_px)
            y2 = min(height, detection.y + radius + self.roi_padding_px)

            roi = motion_mask[y1:y2, x1:x2]

            motion_pixels = int(cv2.countNonZero(roi))

            if motion_pixels >= self.min_motion_pixels:
                filtered.append(detection)

        return filtered


def draw_motion_mask_preview(
    frame: np.ndarray,
    motion_mask: np.ndarray,
    origin: tuple[int, int] = (30, 380),
    width_px: int = 220,
) -> np.ndarray:
    """
    Draw a small motion-mask preview on the video for debugging.
    """

    output = frame.copy()

    mask_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

    h, w = motion_mask.shape[:2]
    height_px = int(width_px * h / w)

    preview = cv2.resize(mask_bgr, (width_px, height_px))

    x0, y0 = origin
    x1 = x0 + width_px
    y1 = y0 + height_px

    output[y0:y1, x0:x1] = preview

    cv2.rectangle(output, (x0, y0), (x1, y1), (255, 255, 255), 1)

    cv2.putText(
        output,
        "Motion mask",
        (x0, y0 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output