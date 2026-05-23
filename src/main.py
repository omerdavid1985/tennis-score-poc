from pathlib import Path

import cv2
from tqdm import tqdm

# Court calibration / visualization
from geometry.court_detector import (
    draw_court_lines,
    load_court_calibration,
    save_court_calibration,
    select_court_corners,
)

# Court geometry transforms
from geometry.homography import (
    compute_image_to_court_homography,
    image_point_to_court,
)

# Visualization helpers
from visualization.draw import draw_mini_top_down_court

# Ball detection
from detection.yolo_ball_detector import (
    YoloBallDetector,
    draw_yolo_ball_detections,
)

from tracking.ball_tracker import BallTracker, draw_tracked_ball

import time

from detection.detection_io import (
    load_ball_detections_csv,
    save_ball_detections_csv,
)

from detection.motion_filter import MotionFilter, draw_motion_mask_preview

from tracking.trajectory_io import save_tracked_trajectory_csv

from events.ball_event_detector import (
    BallEventDetector,
    draw_ball_events,
)

from events.event_io import save_ball_events_csv

from visualization.event_visualizer import (
    draw_ball_events_on_mini_court,
)

from tracking.tracking_debug_io import save_tracking_debug_csv

from detection.player_detector import (
    PlayerDetector,
    draw_player_detections,
)

from detection.player_detection_cache import (
    load_player_detections_csv,
    save_player_detections_csv,
)

# -----------------------------------------------------------------------------
# Input / output paths
# -----------------------------------------------------------------------------

INPUT_VIDEO = Path("data/input/sample_match.mp4")
OUTPUT_VIDEO = Path("data/output/annotated_sample_match.mp4")
BALL_DETECTIONS_FILE = Path("data/output/ball_detections.csv")
TRACKED_TRAJECTORY_FILE = Path("data/output/tracked_trajectory.csv")
BALL_EVENTS_FILE = Path("data/output/ball_events.csv")
TRACKING_DEBUG_FILE = Path("data/output/tracking_debug.csv")
PLAYER_DETECTIONS_FILE = Path("data/output/player_detections.csv")

# Saved calibration file so the user does not need to recalibrate every run
COURT_CALIBRATION_FILE = Path("data/output/court_calibration.json")


def main() -> None:
    """
    Main video-processing pipeline.

    Current pipeline:
        video input
        -> court calibration
        -> homography setup
        -> court overlay
        -> mini top-down court
        -> ball detection
        -> output annotated video
    """

    # -------------------------------------------------------------------------
    # Validate input
    # -------------------------------------------------------------------------

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")

    # Ensure output directory exists
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Open input video
    # -------------------------------------------------------------------------

    cap = cv2.VideoCapture(str(INPUT_VIDEO))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {INPUT_VIDEO}")

    # Read video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # -------------------------------------------------------------------------
    # Read first frame for calibration
    # -------------------------------------------------------------------------

    ret, first_frame = cap.read()

    if not ret:
        raise RuntimeError("Failed to read first frame")

    # -------------------------------------------------------------------------
    # Court calibration
    # -------------------------------------------------------------------------
    #
    # If calibration already exists:
    #     load it from disk
    #
    # Otherwise:
    #     ask the user to click court corners
    #     save calibration for future runs
    #
    # Corner click order:
    #     1. near-left
    #     2. near-right
    #     3. far-right
    #     4. far-left
    #
    # -------------------------------------------------------------------------

    if COURT_CALIBRATION_FILE.exists():
        corners = load_court_calibration(COURT_CALIBRATION_FILE)

        print(f"Loaded calibration: {COURT_CALIBRATION_FILE}")

    else:
        corners = select_court_corners(first_frame)

        save_court_calibration(corners, COURT_CALIBRATION_FILE)

        print(f"Saved calibration: {COURT_CALIBRATION_FILE}")

    # -------------------------------------------------------------------------
    # Homography setup
    # -------------------------------------------------------------------------
    #
    # Homography allows conversion:
    #
    #     image pixels <-> real court coordinates
    #
    # This is the foundation for:
    #     - bounce detection
    #     - in/out estimation
    #     - top-down court visualization
    #     - player positioning
    #
    # -------------------------------------------------------------------------

    image_to_court_h = compute_image_to_court_homography(corners)

    # Example debug conversion:
    # map image center into court coordinates
    center_x = width / 2
    center_y = height / 2

    court_x, court_y = image_point_to_court(
        (center_x, center_y),
        image_to_court_h,
    )

    print(
        f"Image center maps to court: "
        f"x={court_x:.2f}m, y={court_y:.2f}m"
    )

    # -------------------------------------------------------------------------
    # Initialize detectors
    # -------------------------------------------------------------------------

    ball_detector = YoloBallDetector(
    model_path="models/ball/tennis_ball_yolov8.pt",
    confidence_threshold=0.10,
    )

    motion_filter = MotionFilter()

    player_detector = PlayerDetector()

    ball_tracker = BallTracker(trajectory_length=120)

    ball_event_detector = BallEventDetector()
    ball_events = []

    cached_detections = load_ball_detections_csv(BALL_DETECTIONS_FILE)

    if cached_detections:
        print(f"Loaded cached ball detections: {BALL_DETECTIONS_FILE}")
    else:
        print("No cached ball detections found. YOLO will run.")

    if PLAYER_DETECTIONS_FILE.exists():
        print(f"Loading cached player detections: {PLAYER_DETECTIONS_FILE}")
        player_detections_by_frame = load_player_detections_csv(
            PLAYER_DETECTIONS_FILE
        )
        use_cached_player_detections = True
    else:
        print("No cached player detections found. Running YOLO person detector.")
        player_detections_by_frame = {}
        use_cached_player_detections = False

    # -------------------------------------------------------------------------
    # Reset video back to frame 0
    # -------------------------------------------------------------------------
    #
    # We already consumed the first frame during calibration.
    #
    # -------------------------------------------------------------------------

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # -------------------------------------------------------------------------
    # Output video writer
    # -------------------------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height),
    )

    start_time = time.time()
    processed_frames = 0
    detections_by_frame = {}
    trajectory_by_frame = {}
    tracking_debug_rows = []

    # -------------------------------------------------------------------------
    # Main frame-processing loop
    # -------------------------------------------------------------------------

    for frame_idx in tqdm(
        range(frame_count),
        desc="Processing video",
    ):
        ret, frame = cap.read()

        if not ret:
            break

        # ---------------------------------------------------------------------
        # Frame timing
        # ---------------------------------------------------------------------

        time_sec = frame_idx / fps if fps > 0 else 0.0

        # ---------------------------------------------------------------------
        # Court overlay
        # ---------------------------------------------------------------------

        # Run detection on the clean camera frame before drawing overlays.
        # This avoids detecting our own court lines, mini-map, or debug text.
        # Build motion mask from the clean camera frame.
        motion_mask = motion_filter.build_motion_mask(frame)

        # Load or run YOLO detections.
        if frame_idx in cached_detections:
            detections = cached_detections[frame_idx]
        else:
            detections = ball_detector.detect(frame)
            detections_by_frame[frame_idx] = detections

        # Keep only YOLO detections that overlap with moving pixels.
        motion_filtered_detections = motion_filter.filter_detections(
            detections,
            motion_mask,
        )

        # Track only motion-filtered detections.
        tracked_ball = ball_tracker.update(motion_filtered_detections)

        tracking_debug_rows.append(
            {
                "frame_idx": frame_idx,
                "num_ball_detections": len(detections),
                "num_motion_filtered_detections": len(motion_filtered_detections),
                "tracked_ball": tracked_ball,
            }
        )

        ball_event = ball_event_detector.update(
            frame_idx,
            tracked_ball,
            image_to_court_h,
        )

        if use_cached_player_detections:
            player_detections = player_detections_by_frame.get(frame_idx, [])
        else:
            player_detections = player_detector.detect(frame)
            player_detections_by_frame[frame_idx] = player_detections

        if ball_event is not None:
            ball_events.append(ball_event)
        
        trajectory_by_frame[frame_idx] = tracked_ball

        # Draw visualization overlays only after detection is complete.
        frame = draw_court_lines(frame, corners)
        frame = draw_mini_top_down_court(frame)

        frame = draw_ball_events_on_mini_court(
            frame,
            ball_events,
        )    
        
        # Raw YOLO detections can be noisy, so draw only filtered detections for now.
        frame = draw_yolo_ball_detections(frame, motion_filtered_detections)

        frame = draw_tracked_ball(
            frame,
            tracked_ball,
            ball_tracker.trajectory,
        )

        frame = draw_ball_events(
            frame,
            ball_events,
        )

        # Optional debug preview.
        frame = draw_motion_mask_preview(frame, motion_mask)

        frame = draw_player_detections(frame, player_detections)
        
        # ---------------------------------------------------------------------
        # Debug information overlay
        # ---------------------------------------------------------------------

        cv2.putText(
            frame,
            f"Frame: {frame_idx} | Time: {time_sec:.2f}s",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # ---------------------------------------------------------------------
        # Write annotated frame
        # ---------------------------------------------------------------------

        writer.write(frame)

        processed_frames += 1

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    cap.release()
    writer.release()

    if detections_by_frame:
        save_ball_detections_csv(BALL_DETECTIONS_FILE, detections_by_frame)
        print(f"Saved ball detections: {BALL_DETECTIONS_FILE}")

    save_tracked_trajectory_csv(
        TRACKED_TRAJECTORY_FILE,
        trajectory_by_frame,
    )

    save_ball_events_csv(
        BALL_EVENTS_FILE,
        ball_events,
    )

    save_tracking_debug_csv(
        TRACKING_DEBUG_FILE,
        tracking_debug_rows,
    )

    if not use_cached_player_detections:
        save_player_detections_csv(
            PLAYER_DETECTIONS_FILE,
            player_detections_by_frame,
        )

    print(f"Saved player detections: {PLAYER_DETECTIONS_FILE}")

    print(f"Saved tracking debug: {TRACKING_DEBUG_FILE}")

    print(f"Saved ball events: {BALL_EVENTS_FILE}")

    print(f"Saved tracked trajectory: {TRACKED_TRAJECTORY_FILE}")

    elapsed_time = time.time() - start_time
    processing_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0.0

    print(f"Processed frames: {processed_frames}")
    print(f"Processing time: {elapsed_time:.2f}s")
    print(f"Processing FPS: {processing_fps:.2f}")
    print(f"Done: {OUTPUT_VIDEO}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()