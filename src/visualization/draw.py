from __future__ import annotations

import cv2
import numpy as np


def draw_mini_top_down_court(
    frame: np.ndarray,
    origin: tuple[int, int] = (30, 90),
    height_px: int = 260,
) -> np.ndarray:
    """
    Draw a mini top-down singles tennis court on top of the video frame.
    """

    output = frame.copy()

    court_length_m = 23.77
    singles_width_m = 8.23
    net_y_m = court_length_m / 2.0
    service_distance_m = 6.40

    visual_width_scale = 1.8
    width_px = int(
        height_px
        * singles_width_m
        / court_length_m
        * visual_width_scale
    )
    
    x0, y0 = origin
    x1 = x0 + width_px
    y1 = y0 + height_px

    def to_px(x_m: float, y_m: float) -> tuple[int, int]:
        x = int(x0 + (x_m / singles_width_m) * width_px)
        y = int(y0 + (y_m / court_length_m) * height_px)
        return x, y

    # Background
    cv2.rectangle(output, (x0 - 10, y0 - 10), (x1 + 10, y1 + 10), (30, 30, 30), -1)

    # Outer court
    cv2.rectangle(output, (x0, y0), (x1, y1), (0, 255, 255), 2)

    # Net
    net_left = to_px(0.0, net_y_m)
    net_right = to_px(singles_width_m, net_y_m)
    cv2.line(output, net_left, net_right, (255, 255, 0), 2)

    # Service lines
    near_service_y = net_y_m - service_distance_m
    far_service_y = net_y_m + service_distance_m

    cv2.line(
        output,
        to_px(0.0, near_service_y),
        to_px(singles_width_m, near_service_y),
        (0, 165, 255),
        1,
    )

    cv2.line(
        output,
        to_px(0.0, far_service_y),
        to_px(singles_width_m, far_service_y),
        (0, 165, 255),
        1,
    )

    # Center service line
    center_x = singles_width_m / 2.0
    cv2.line(
        output,
        to_px(center_x, near_service_y),
        to_px(center_x, far_service_y),
        (0, 165, 255),
        1,
    )

    cv2.putText(
        output,
        "Top-down court",
        (x0 - 5, y0 - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output