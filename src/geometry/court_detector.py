from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


COURT_POINT_NAMES = [
    "near_left",
    "near_right",
    "far_right",
    "far_left",
]


def select_court_corners(first_frame: np.ndarray) -> list[dict]:
    """
    Ask the user to click the 4 outer court corners.

    Click order:
    1. near-left corner
    2. near-right corner
    3. far-right corner
    4. far-left corner
    """

    points: list[tuple[int, int]] = []
    display = first_frame.copy()

    window_name = "Court calibration - click 4 court corners"

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal display

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(points) >= 4:
            return

        points.append((x, y))

        point_index = len(points) - 1
        point_name = COURT_POINT_NAMES[point_index]

        cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(
            display,
            f"{point_index + 1}: {point_name}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        if len(points) > 1:
            cv2.line(
                display,
                points[-2],
                points[-1],
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if len(points) == 4:
            cv2.line(
                display,
                points[-1],
                points[0],
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        preview = display.copy()

        cv2.putText(
            preview,
            "Click 4 outer court corners: near-left, near-right, far-right, far-left",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            preview,
            "Press R to reset | ENTER to accept | ESC to cancel",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # ENTER
            if len(points) == 4:
                break

        elif key == 27:  # ESC
            cv2.destroyWindow(window_name)
            raise RuntimeError("Court calibration cancelled by user")

        elif key in (ord("r"), ord("R")):
            points.clear()
            display = first_frame.copy()

    cv2.destroyWindow(window_name)

    corners: list[dict] = []

    for name, point in zip(COURT_POINT_NAMES, points):
        corners.append(
            {
                "name": name,
                "x": int(point[0]),
                "y": int(point[1]),
            }
        )

    return corners


def save_court_calibration(corners: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "type": "manual_4_corners_v1",
        "click_order": COURT_POINT_NAMES,
        "corners": corners,
    }

    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_court_calibration(input_path: Path) -> list[dict]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    return data["corners"]


def draw_court_lines(frame: np.ndarray, corners: list[dict]) -> np.ndarray:
    output = frame.copy()

    points = [(corner["x"], corner["y"]) for corner in corners]

    if len(points) != 4:
        return output

    near_left, near_right, far_right, far_left = points

    # Draw only the real selected outer court polygon.
    # Do not draw net/service/middle lines here because image-space midpoints
    # are wrong when the camera is not centered behind the court.
    court_lines = [
        (near_left, near_right),
        (near_right, far_right),
        (far_right, far_left),
        (far_left, near_left),
    ]

    for p1, p2 in court_lines:
        cv2.line(output, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)

    # Draw corner labels
    for corner in corners:
        point = (corner["x"], corner["y"])
        cv2.circle(output, point, 7, (0, 0, 255), -1)
        cv2.putText(
            output,
            corner["name"],
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output