from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tracking.ball_tracker import TrackedBall

from enum import Enum

class RallyState(str, Enum):
    IDLE = "IDLE"
    WAITING_FOR_SERVE = "WAITING_FOR_SERVE"
    SERVE_IN_PROGRESS = "SERVE_IN_PROGRESS"
    RALLY = "RALLY"
    POINT_ENDED = "POINT_ENDED"

@dataclass
class RallyFrameState:
    frame_idx: int
    is_active: bool
    reason: str
    state: RallyState


@dataclass
class RallySegment:
    """
    Continuous active rally frame range.
    """

    start_frame: int
    end_frame: int
    duration_frames: int


class RallySegmenter:
    """
    Detect active rally regions from ball tracking.

    First version:
    - active if ball is tracked and moving
    - inactive if ball is missing/stopped for several frames
    """

    def __init__(
        self,
        min_ball_speed_px: float = 4.0,
        min_active_frames: int = 8,
        inactive_gap_frames: int = 20,
        active_confirm_frames: int = 5,
        inactive_confirm_frames: int = 12,
    ) -> None:
        self.min_ball_speed_px = min_ball_speed_px
        self.min_active_frames = min_active_frames
        self.inactive_gap_frames = inactive_gap_frames

        self.active_confirm_frames = active_confirm_frames
        self.inactive_confirm_frames = inactive_confirm_frames

        self.active_candidate_count = 0
        self.inactive_candidate_count = 0
        self.is_currently_active = False

        self.previous_ball: TrackedBall | None = None

        self.active_start_frame: int | None = None
        self.last_active_frame: int | None = None

        self.frame_states: list[RallyFrameState] = []
        self.segments: list[RallySegment] = []

        self.state = RallyState.IDLE
        self.point_ended_count = 0

    def update(
        self,
        frame_idx: int,
        tracked_ball: TrackedBall | None,
    ) -> RallyFrameState:
        """
        Update rally state using current tracked ball.
        """

        is_active_candidate = False
        reason = "no_ball"

        # --------------------------------------------------------
        # Raw activity candidate
        #
        # This is the immediate per-frame decision before smoothing.
        # It may flicker because tracking can briefly miss the ball.
        # --------------------------------------------------------

        if tracked_ball is not None:
            speed = self._estimate_ball_speed(tracked_ball)

            if speed >= self.min_ball_speed_px:
                is_active_candidate = True
                reason = "ball_moving"
            else:
                reason = "ball_too_slow"

        # --------------------------------------------------------
        # Activity smoothing / hysteresis
        #
        # ACTIVE is entered only after several active candidates.
        # INACTIVE is entered only after several inactive candidates.
        # --------------------------------------------------------

        if is_active_candidate:
            self.active_candidate_count += 1
            self.inactive_candidate_count = 0
        else:
            self.inactive_candidate_count += 1
            self.active_candidate_count = 0

        if not self.is_currently_active:
            if self.active_candidate_count >= self.active_confirm_frames:
                self.is_currently_active = True

        if self.is_currently_active:
            if self.inactive_candidate_count >= self.inactive_confirm_frames:
                self.is_currently_active = False

        # --------------------------------------------------------
        # Explicit rally state machine
        #
        # First version:
        # IDLE -> RALLY -> POINT_ENDED -> IDLE
        #
        # Serve-specific states are placeholders for later.
        # --------------------------------------------------------

        if self.state == RallyState.IDLE:
            if self.is_currently_active:
                self.state = RallyState.RALLY

        elif self.state == RallyState.RALLY:
            if not self.is_currently_active:
                self.state = RallyState.POINT_ENDED
                self.point_ended_count = 0

        elif self.state == RallyState.POINT_ENDED:
            self.point_ended_count += 1

            if self.is_currently_active:
                self.state = RallyState.RALLY
            elif self.point_ended_count >= self.inactive_gap_frames:
                self.state = RallyState.IDLE

        # --------------------------------------------------------
        # Segment update uses the smoothed state, not the raw state.
        # --------------------------------------------------------

        if self.is_currently_active:
            self._mark_active(frame_idx)
            state = RallyFrameState(
                frame_idx=frame_idx,
                is_active=True,
                reason=reason,
                state=self.state,
            )
        else:
            self._maybe_close_segment(frame_idx)
            state = RallyFrameState(
                frame_idx=frame_idx,
                is_active=False,
                reason=reason,
                state=self.state,
            )

        self.previous_ball = tracked_ball
        self.frame_states.append(state)

        return state
    
    def _estimate_ball_speed(
        self,
        tracked_ball: TrackedBall,
    ) -> float:
        """
        Estimate image-space ball speed from consecutive tracked positions.
        """

        if self.previous_ball is None:
            return 0.0

        return float(
            np.hypot(
                tracked_ball.x - self.previous_ball.x,
                tracked_ball.y - self.previous_ball.y,
            )
        )

    def _mark_active(
        self,
        frame_idx: int,
    ) -> None:
        """
        Start or extend current active rally region.
        """

        if self.active_start_frame is None:
            self.active_start_frame = frame_idx

        self.last_active_frame = frame_idx

    def _maybe_close_segment(
        self,
        frame_idx: int,
    ) -> None:
        """
        Close current segment after enough inactive frames.
        """

        if self.active_start_frame is None:
            return

        if self.last_active_frame is None:
            return

        inactive_frames = frame_idx - self.last_active_frame

        if inactive_frames < self.inactive_gap_frames:
            return

        duration = self.last_active_frame - self.active_start_frame + 1

        if duration >= self.min_active_frames:
            self.segments.append(
                RallySegment(
                    start_frame=self.active_start_frame,
                    end_frame=self.last_active_frame,
                    duration_frames=duration,
                )
            )

        self.active_start_frame = None
        self.last_active_frame = None

    def finalize(self) -> list[RallySegment]:
        """
        Close final active segment at end of video.
        """

        if (
            self.active_start_frame is not None
            and self.last_active_frame is not None
        ):
            duration = self.last_active_frame - self.active_start_frame + 1

            if duration >= self.min_active_frames:
                self.segments.append(
                    RallySegment(
                        start_frame=self.active_start_frame,
                        end_frame=self.last_active_frame,
                        duration_frames=duration,
                    )
                )

            self.active_start_frame = None
            self.last_active_frame = None

        return self.segments


def draw_rally_state(
    frame: np.ndarray,
    state: RallyFrameState,
) -> np.ndarray:
    """
    Draw current rally activity state.
    """

    output = frame.copy()

    if state.is_active:
        text = "RALLY ACTIVE"
        color = (0, 255, 0)
    else:
        text = f"INACTIVE: {state.reason}"
        color = (120, 120, 120)

    cv2.putText(
        output,
        text,
        (frame.shape[1] - 360, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )

    return output