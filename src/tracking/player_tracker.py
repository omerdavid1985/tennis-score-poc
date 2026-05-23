from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter

from detection.player_detector import PlayerDetection
from detection.player_assignment import AssignedPlayers


@dataclass
class TrackedPlayer:
    """
    Stable tracked player box.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    role: str
    missed_frames: int
    is_predicted: bool


class SinglePlayerTracker:
    """
    Kalman tracker for one player role: near or far.
    """

    def __init__(
        self,
        role: str,
        max_missed_frames: int = 15,
        max_match_distance_px: float = 180.0,
    ) -> None:
        self.role = role
        self.max_missed_frames = max_missed_frames
        self.max_match_distance_px = max_match_distance_px

        self.initialized = False
        self.missed_frames = 0

        # Keep box size outside the Kalman state.
        self.width = 0.0
        self.height = 0.0
        self.confidence = 0.0

        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # State: center_x, center_y, velocity_x, velocity_y
        self.kf.F = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        # Measurement: center_x, center_y
        self.kf.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=float,
        )

        # Initial uncertainty / noise tuning.
        self.kf.P *= 500.0
        self.kf.R *= 25.0
        self.kf.Q *= 2.0

    def update(
        self,
        detection: PlayerDetection | None,
    ) -> TrackedPlayer | None:
        """
        Update tracker with one assigned detection.
        If detection is missing, predict for a short time.
        """

        if detection is None:
            return self._update_missing()

        cx, cy = self._detection_center(detection)

        if not self.initialized:
            self._initialize(detection, cx, cy)
            return self._to_tracked_player(is_predicted=False)

        self.kf.predict()

        predicted_cx = float(self.kf.x[0, 0])
        predicted_cy = float(self.kf.x[1, 0])

        distance = float(
            np.hypot(
                cx - predicted_cx,
                cy - predicted_cy,
            )
        )

        # Reject detections too far from the current predicted player.
        if distance > self.max_match_distance_px:
            return self._update_missing()

        self.kf.update(
            np.array(
                [
                    [cx],
                    [cy],
                ],
                dtype=float,
            )
        )

        # Smooth box size slightly to reduce visual flicker.
        new_width = detection.x2 - detection.x1
        new_height = detection.y2 - detection.y1

        self.width = 0.8 * self.width + 0.2 * new_width
        self.height = 0.8 * self.height + 0.2 * new_height
        self.confidence = detection.confidence
        self.missed_frames = 0

        return self._to_tracked_player(is_predicted=False)

    def _update_missing(self) -> TrackedPlayer | None:
        """
        Predict player location when detector misses temporarily.
        """

        if not self.initialized:
            return None

        self.kf.predict()
        self.missed_frames += 1

        if self.missed_frames > self.max_missed_frames:
            self.initialized = False
            return None

        return self._to_tracked_player(is_predicted=True)

    def _initialize(
        self,
        detection: PlayerDetection,
        cx: float,
        cy: float,
    ) -> None:
        """
        Initialize Kalman state from first detection.
        """

        self.kf.x = np.array(
            [
                [cx],
                [cy],
                [0.0],
                [0.0],
            ],
            dtype=float,
        )

        self.width = float(detection.x2 - detection.x1)
        self.height = float(detection.y2 - detection.y1)
        self.confidence = detection.confidence
        self.missed_frames = 0
        self.initialized = True

    def _to_tracked_player(
        self,
        is_predicted: bool,
    ) -> TrackedPlayer:
        """
        Convert current Kalman state into a bounding box.
        """

        cx = float(self.kf.x[0, 0])
        cy = float(self.kf.x[1, 0])

        half_w = 0.5 * self.width
        half_h = 0.5 * self.height

        return TrackedPlayer(
            x1=int(round(cx - half_w)),
            y1=int(round(cy - half_h)),
            x2=int(round(cx + half_w)),
            y2=int(round(cy + half_h)),
            confidence=float(self.confidence),
            role=self.role,
            missed_frames=self.missed_frames,
            is_predicted=is_predicted,
        )

    @staticmethod
    def _detection_center(
        detection: PlayerDetection,
    ) -> tuple[float, float]:
        """
        Center of detection box.
        """

        return (
            0.5 * (detection.x1 + detection.x2),
            0.5 * (detection.y1 + detection.y2),
        )


class PlayerTracker:
    """
    Tracks near and far player separately.
    """

    def __init__(self) -> None:
        self.near_tracker = SinglePlayerTracker(role="Near")
        self.far_tracker = SinglePlayerTracker(role="Far")

    def update(
        self,
        assigned_players: AssignedPlayers,
    ) -> list[TrackedPlayer]:
        """
        Update both near/far trackers and return visible tracked players.
        """

        tracked_players: list[TrackedPlayer] = []

        near = self.near_tracker.update(assigned_players.near_player)
        far = self.far_tracker.update(assigned_players.far_player)

        if near is not None:
            tracked_players.append(near)

        if far is not None:
            tracked_players.append(far)

        return tracked_players