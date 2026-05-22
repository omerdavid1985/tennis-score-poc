# Tennis AI Scoring POC — Task Tracker

## Project Status

- [ ] Milestone 0 — Project Bootstrap
- [ ] Milestone 1 — Manual Court Calibration
- [ ] Milestone 2 — Ball Detection Baseline
- [ ] Milestone 3 — Ball Tracking
- [ ] Milestone 4 — Player Detection
- [ ] Milestone 5 — Bounce and Hit Candidates
- [ ] Milestone 6 — Rally Segmentation
- [ ] Milestone 7 — First Scoring POC

---

# Milestone 0 — Project Bootstrap

## Environment Setup

- [ ] Install Python 3.10 or 3.11
- [ ] Install VS Code extensions
  - [ ] Python
  - [ ] Pylance
  - [ ] Jupyter
  - [ ] Black Formatter
  - [ ] isort
  - [ ] GitLens
  - [ ] TODO Highlight

- [ ] Create repository folder
- [ ] Open project in VS Code
- [ ] Create `.venv`
- [ ] Activate virtual environment
- [ ] Upgrade pip
- [ ] Create `requirements.txt`
- [ ] Install dependencies

---

## Project Structure

- [ ] Create `data/input`
- [ ] Create `data/output`
- [ ] Create `src`
- [ ] Create `src/video`
- [ ] Create `src/detection`
- [ ] Create `src/tracking`
- [ ] Create `src/geometry`
- [ ] Create `src/events`
- [ ] Create `src/scoring`
- [ ] Create `src/visualization`
- [ ] Create `tests`

---

## Video Bootstrap

- [ ] Add sample tennis video
- [ ] Create `src/main.py`
- [ ] Open video with OpenCV
- [ ] Read frames successfully
- [ ] Extract FPS/frame count
- [ ] Create output video writer
- [ ] Draw frame number
- [ ] Draw timestamp
- [ ] Write annotated frames
- [ ] Save output video
- [ ] Verify output video plays correctly

---

## Milestone 0 Success Criteria

- [ ] `annotated_sample_match.mp4` is generated successfully
- [ ] Output video contains frame number and timestamp overlay
- [ ] Video processing runs end-to-end without crashing

---

# Milestone 1 — Manual Court Calibration

## Court Geometry

- [ ] Create court calibration module
- [ ] Define court dimensions
- [ ] Add manual point selection
- [ ] Select baseline corners
- [ ] Select service box corners
- [ ] Save calibration JSON
- [ ] Load calibration JSON
- [ ] Compute homography matrix
- [ ] Draw court overlay
- [ ] Validate overlay alignment

---

## Optional Enhancements

- [ ] Draw mini top-down court
- [ ] Add calibration visualization
- [ ] Add recalibration shortcut

---

## Milestone 1 Success Criteria

- [ ] Court overlay aligns correctly for full video
- [ ] Homography transforms image points into court coordinates

---

# Milestone 2 — Ball Detection Baseline

## Detection Infrastructure

- [ ] Create ball detector interface
- [ ] Add YOLO experiment
- [ ] Add TrackNet experiment
- [ ] Load pretrained model
- [ ] Run inference on single frame
- [ ] Run inference on full video
- [ ] Draw ball detections
- [ ] Store detections to CSV

---

## Detection Quality

- [ ] Evaluate missed detections
- [ ] Evaluate false positives
- [ ] Add confidence filtering
- [ ] Restrict detections to court region

---

## Milestone 2 Success Criteria

- [ ] Ball visible in meaningful percentage of rally frames
- [ ] Detection CSV exported successfully

---

# Milestone 3 — Ball Tracking

## Tracking Logic

- [ ] Create tracking module
- [ ] Add Kalman filter
- [ ] Predict next ball position
- [ ] Match detections to prediction
- [ ] Reject impossible jumps
- [ ] Handle temporary missed detections
- [ ] Draw trajectory line
- [ ] Export tracked trajectory CSV

---

## Tracking Quality

- [ ] Verify smooth trajectory
- [ ] Tune tracking thresholds
- [ ] Tune missing-frame recovery

---

## Milestone 3 Success Criteria

- [ ] Ball track survives short detection failures
- [ ] Trajectory smoother than raw detections

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

