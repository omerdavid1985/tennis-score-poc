from __future__ import annotations

import cv2
import numpy as np

from events.ball_event_detector import BallEvent


def draw_ball_events_on_mini_court(
    frame: np.ndarray,
    events: list[BallEvent],
    origin: tuple[int, int] = (30, 90),
    height_px: int = 260,
) -> np.ndarray:
    """
    Draw detected ball-event candidates on the mini top-down court.
    """

    output = frame.copy()

    court_length_m = 23.77
    singles_width_m = 8.23

    visual_width_scale = 1.8

    width_px = int(
        height_px
        * singles_width_m
        / court_length_m
        * visual_width_scale
    )

    x0, y0 = origin

    for event in events:

        if event.court_x is None or event.court_y is None:
            continue

        if not event.is_inside_court:
            continue

        px = int(
            x0
            + (event.court_x / singles_width_m) * width_px
        )

        py = int(
            y0
            + ((court_length_m - event.court_y) / court_length_m) * height_px
        )

        if event.event_type == "bounce_candidate":
            color = (0, 255, 255)
            radius = 6
        else:
            color = (255, 255, 0)
            radius = 4

        cv2.circle(
            output,
            (px, py),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )

    return output