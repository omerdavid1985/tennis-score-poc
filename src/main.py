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

# -----------------------------------------------------------------------------
# Input / output paths
# -----------------------------------------------------------------------------

INPUT_VIDEO = Path("data/input/sample_match.mp4")
OUTPUT_VIDEO = Path("data/output/annotated_sample_match.mp4")

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
    ball_tracker = BallTracker()

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
        detections = ball_detector.detect(frame)

        # Draw visualization overlays only after detection is complete.
        frame = draw_court_lines(frame, corners)
        frame = draw_mini_top_down_court(frame)
        tracked_ball = ball_tracker.update(detections)

        frame = draw_yolo_ball_detections(frame, detections)
        frame = draw_tracked_ball(frame, tracked_ball)
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