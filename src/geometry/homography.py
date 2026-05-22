from __future__ import annotations

import cv2
import numpy as np


COURT_LENGTH_M = 23.77
SINGLES_WIDTH_M = 8.23

NET_Y_M = COURT_LENGTH_M / 2.0
SERVICE_LINE_DISTANCE_FROM_NET_M = 6.40

NEAR_SERVICE_Y_M = NET_Y_M - SERVICE_LINE_DISTANCE_FROM_NET_M
FAR_SERVICE_Y_M = NET_Y_M + SERVICE_LINE_DISTANCE_FROM_NET_M
CENTER_X_M = SINGLES_WIDTH_M / 2.0


def build_court_model_points() -> np.ndarray:
    """
    Real court model points in meters.

    Must match click order:
    1. near-left
    2. near-right
    3. far-right
    4. far-left
    """

    return np.array(
        [
            [0.0, 0.0],
            [SINGLES_WIDTH_M, 0.0],
            [SINGLES_WIDTH_M, COURT_LENGTH_M],
            [0.0, COURT_LENGTH_M],
        ],
        dtype=np.float32,
    )


def corners_to_image_points(corners: list[dict]) -> np.ndarray:
    return np.array(
        [[corner["x"], corner["y"]] for corner in corners],
        dtype=np.float32,
    )


def compute_image_to_court_homography(corners: list[dict]) -> np.ndarray:
    image_points = corners_to_image_points(corners)
    court_points = build_court_model_points()

    homography, _ = cv2.findHomography(image_points, court_points)

    if homography is None:
        raise RuntimeError("Failed to compute image-to-court homography")

    return homography


def compute_court_to_image_homography(corners: list[dict]) -> np.ndarray:
    court_points = build_court_model_points()
    image_points = corners_to_image_points(corners)

    homography, _ = cv2.findHomography(court_points, image_points)

    if homography is None:
        raise RuntimeError("Failed to compute court-to-image homography")

    return homography


def image_point_to_court(
    image_point: tuple[float, float],
    homography: np.ndarray,
) -> tuple[float, float]:
    point = np.array(
        [[[image_point[0], image_point[1]]]],
        dtype=np.float32,
    )

    transformed = cv2.perspectiveTransform(point, homography)

    x_m = float(transformed[0][0][0])
    y_m = float(transformed[0][0][1])

    return x_m, y_m


def court_point_to_image(
    court_point: tuple[float, float],
    homography: np.ndarray,
) -> tuple[int, int]:
    point = np.array(
        [[[court_point[0], court_point[1]]]],
        dtype=np.float32,
    )

    transformed = cv2.perspectiveTransform(point, homography)

    x_px = int(round(float(transformed[0][0][0])))
    y_px = int(round(float(transformed[0][0][1])))

    return x_px, y_px


def get_projected_court_lines(
    court_to_image_homography: np.ndarray,
) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
    """
    Returns projected tennis court lines in image coordinates.

    Output:
    [
        ((x1, y1), (x2, y2), "line_name"),
        ...
    ]
    """

    court_lines_m = [
        # Outer court
        ((0.0, 0.0), (SINGLES_WIDTH_M, 0.0), "near_baseline"),
        ((SINGLES_WIDTH_M, 0.0), (SINGLES_WIDTH_M, COURT_LENGTH_M), "right_sideline"),
        ((SINGLES_WIDTH_M, COURT_LENGTH_M), (0.0, COURT_LENGTH_M), "far_baseline"),
        ((0.0, COURT_LENGTH_M), (0.0, 0.0), "left_sideline"),

        # Net
        ((0.0, NET_Y_M), (SINGLES_WIDTH_M, NET_Y_M), "net"),

        # Service lines
        ((0.0, NEAR_SERVICE_Y_M), (SINGLES_WIDTH_M, NEAR_SERVICE_Y_M), "near_service_line"),
        ((0.0, FAR_SERVICE_Y_M), (SINGLES_WIDTH_M, FAR_SERVICE_Y_M), "far_service_line"),

        # Center service line
        ((CENTER_X_M, NEAR_SERVICE_Y_M), (CENTER_X_M, FAR_SERVICE_Y_M), "center_service_line"),
    ]

    projected_lines = []

    for p1_m, p2_m, name in court_lines_m:
        p1_img = court_point_to_image(p1_m, court_to_image_homography)
        p2_img = court_point_to_image(p2_m, court_to_image_homography)

        projected_lines.append((p1_img, p2_img, name))

    return projected_lines