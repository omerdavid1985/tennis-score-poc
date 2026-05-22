from pathlib import Path

import cv2
from tqdm import tqdm

from geometry.court_detector import (
    draw_court_lines,
    load_court_calibration,
    save_court_calibration,
    select_court_corners,
)

from geometry.homography import compute_image_to_court_homography, image_point_to_court

from visualization.draw import draw_mini_top_down_court

INPUT_VIDEO = Path("data/input/sample_match.mp4")
OUTPUT_VIDEO = Path("data/output/annotated_sample_match.mp4")
COURT_CALIBRATION_FILE = Path("data/output/court_calibration.json")


def main() -> None:
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame")

    if COURT_CALIBRATION_FILE.exists():
        corners = load_court_calibration(COURT_CALIBRATION_FILE)
        print(f"Loaded calibration: {COURT_CALIBRATION_FILE}")
    else:
        corners = select_court_corners(first_frame)
        save_court_calibration(corners, COURT_CALIBRATION_FILE)
        print(f"Saved calibration: {COURT_CALIBRATION_FILE}")

    image_to_court_h = compute_image_to_court_homography(corners)

    center_x = width / 2
    center_y = height / 2
    court_x, court_y = image_point_to_court((center_x, center_y), image_to_court_h)

    print(f"Image center maps to court: x={court_x:.2f}m, y={court_y:.2f}m")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height),
    )

    for frame_idx in tqdm(range(frame_count), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_idx / fps if fps > 0 else 0.0

        frame = draw_court_lines(frame, corners)

        frame = draw_mini_top_down_court(frame) 

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

        writer.write(frame)

    cap.release()
    writer.release()

    print(f"Done: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()