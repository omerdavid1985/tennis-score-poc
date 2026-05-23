# Tennis AI Scoring POC — Task Tracker

## Project Status

- [v] Milestone 0 — Project Bootstrap
- [v] Milestone 1 — Manual Court Calibration
- [v] Milestone 2 — Ball Detection Baseline
- [v] Milestone 3 — Ball Tracking
- [ ] Milestone 4 — Player Detection
- [ ] Milestone 5 — Bounce and Hit Candidates
- [ ] Milestone 6 — Rally Segmentation
- [ ] Milestone 7 — First Scoring POC

---

# Milestone 0 — Project Bootstrap

## Environment Setup

- [v] Install Python 3.10 or 3.11
- [v] Install VS Code extensions
  - [v] Python
  - [v] Pylance
  - [v] Jupyter
  - [v] Black Formatter
  - [v] isort
  - [v] GitLens
  - [v] TODO Highlight

- [v] Create repository folder
- [v] Open project in VS Code
- [v] Create `.venv`
- [v] Activate virtual environment
- [v] Upgrade pip
- [v] Create `requirements.txt`
- [v] Install dependencies

---

## Project Structure

- [v] Create `data/input`
- [v] Create `data/output`
- [v] Create `src`
- [v] Create `src/video`
- [v] Create `src/detection`
- [v] Create `src/tracking`
- [v] Create `src/geometry`
- [v] Create `src/events`
- [v] Create `src/scoring`
- [v] Create `src/visualization`
- [v] Create `tests`

---

## Video Bootstrap

- [v] Add sample tennis video
- [v] Create `src/main.py`
- [v] Open video with OpenCV
- [v] Read frames successfully
- [v] Extract FPS/frame count
- [v] Create output video writer
- [v] Draw frame number
- [v] Draw timestamp
- [v] Write annotated frames
- [v] Save output video
- [v] Verify output video plays correctly

---

## Milestone 0 Success Criteria

- [v] `annotated_sample_match.mp4` is generated successfully
- [v] Output video contains frame number and timestamp overlay
- [v] Video processing runs end-to-end without crashing

---

# Milestone 1 — Manual Court Calibration

## Court Geometry

- [v] Create court calibration module
- [v] Add manual point selection
- [v] Select baseline corners
- [v] Save calibration JSON
- [v] Load calibration JSON
- [v] Draw court overlay
- [v] Validate overlay alignment
- [v] Draw mini top-down court
---

## Git Setup
- [v] Initialize Git repository
- [v] Create `.gitignore`
- [v] Create first Git commit
---

## Optional Enhancements

- [v] Add calibration visualization
- [ ] Add recalibration shortcut

---

## Milestone 1 Success Criteria

- [v] Court overlay aligns correctly for full video
- [v] Homography transforms image points into court coordinates

---

# Milestone 2 — Ball Detection Baseline

## Detection Infrastructure

- [v] Create ball detector interface
- [v] Add YOLO experiment
- [ ] Add TrackNet experiment
- [v] Load pretrained model
- [v] Run inference on single frame
- [v] Run inference on full video
- [v] Draw ball detections
- [v] Store detections to CSV
- [v] Integrate tennis-ball-specific YOLO model
- [x] Cache AI detections to disk
- [x] Load cached detections
- [x] Add motion-based detection filtering
- [x] Filter static false positives
- [x] Add trajectory history visualization
- [x] Measure processing FPS
---

## Detection Quality

- [v] Evaluate missed detections
- [v] Evaluate false positives
- [v] Add confidence filtering
- [v] Restrict detections to court region

---

## Milestone 2 Success Criteria

- [v] Ball visible in meaningful percentage of rally frames
- [v] Detection CSV exported successfully

---

# Milestone 3 — Ball Tracking

## Tracking Logic

- [v] Create tracking module
- [v] Add Kalman filter
- [v] Predict next ball position
- [v] Match detections to prediction
- [v] Reject impossible jumps
- [v] Handle temporary missed detections
- [v] Draw trajectory line
- [v] Export tracked trajectory CSV

---

## Tracking Quality

- [v] Verify smooth trajectory
- [v] Tune tracking thresholds
- [ ] Tune missing-frame recovery

---

## Milestone 3 Success Criteria

- [v] Ball track survives short detection failures
- [v] Trajectory smoother than raw detections

---

# Milestone 4 — Player Detection

## Player Detection

- [ ] Add YOLO person detector
- [ ] Detect players on court
- [ ] Filter spectators/background
- [ ] Identify near-side player
- [ ] Identify far-side player
- [ ] Draw player boxes
- [ ] Export player tracks

---

## Tracking

- [ ] Add simple player tracker
- [ ] Maintain player IDs between frames

---

## Milestone 4 Success Criteria

- [ ] Near/far players identified consistently

---

# Milestone 5 — Bounce and Hit Candidates

## Bounce Detection

- [ ] Smooth ball trajectory
- [ ] Detect direction changes
- [ ] Detect bounce candidates
- [ ] Filter impossible bounce locations
- [ ] Draw bounce markers

---

## Hit Detection

- [ ] Detect trajectory changes
- [ ] Estimate hitter proximity
- [ ] Detect hit candidates
- [ ] Draw hit markers

---

## Event Timeline

- [ ] Create event CSV exporter
- [ ] Create event JSON exporter
- [ ] Store bounce events
- [ ] Store hit events

---

## Milestone 5 Success Criteria

- [ ] Candidate events look reasonable on annotated video

---

# Milestone 6 — Rally Segmentation

## State Machine

- [ ] Create rally state machine
- [ ] Add IDLE state
- [ ] Add WAITING_FOR_SERVE state
- [ ] Add SERVE_IN_PROGRESS state
- [ ] Add RALLY state
- [ ] Add POINT_ENDED state

---

## Rally Detection

- [ ] Detect rally start candidates
- [ ] Detect rally end candidates
- [ ] Add confidence scoring
- [ ] Add manual override support

---

## Milestone 6 Success Criteria

- [ ] Video split into rally segments successfully

---

# Milestone 7 — First Scoring POC

## Tennis Scoring Engine

- [ ] Create score model
- [ ] Implement 0/15/30/40 logic
- [ ] Implement deuce
- [ ] Implement advantage
- [ ] Implement games
- [ ] Implement sets
- [ ] Add unit tests

---

## AI-Assisted Scoring

- [ ] Connect point-end events
- [ ] Suggest point winner
- [ ] Update score automatically
- [ ] Export score timeline JSON
- [ ] Draw score overlay on video

---

## Milestone 7 Success Criteria

- [ ] Score updates correctly from validated point winners
- [ ] AI point suggestions visible with confidence

---

# Future / Nice To Have

## Broadcast Features

- [ ] Live camera input
- [ ] Streaming support
- [ ] Real-time overlays
- [ ] WebRTC experimentation
- [ ] RTMP/HLS experimentation

---

## AI Improvements

- [ ] Automatic court calibration
- [ ] Better ball tracking model
- [ ] Bounce classification model
- [ ] Shot classification
- [ ] Serve speed estimation
- [ ] In/out estimation

---

## App Features

- [ ] Mobile app
- [ ] Web dashboard
- [ ] Manual score correction
- [ ] Match statistics
- [ ] Clip generation
- [ ] Cloud processing

