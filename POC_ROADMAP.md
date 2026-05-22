# Tennis AI Scoring POC Roadmap

## 1. Goal

Build a small, no-GUI proof of concept that analyzes a recorded tennis game video taken from the fence behind the court and produces:

1. An annotated output video.
2. A CSV/JSON timeline of detected events.
3. A first simple estimate of rally/point state.
4. Eventually, a proposed score update.

The POC should prove whether automatic or semi-automatic scoring is feasible from the expected camera angle.

This is not the full broadcast app yet. The POC focuses only on computer vision and scoring logic.

---

## 2. Starting Assumptions

### Camera

- Static camera.
- Mounted behind one baseline/fence.
- Full court visible as much as possible.
- Preferably 1080p or higher.
- Preferably 30 FPS or 60 FPS.
- No zoom/pan during the point.
- Singles first. Doubles later.

### Game Format

Start with a simple singles match.

Do not try to support every tennis rule immediately. First goal is to detect:

- ball trajectory
- court geometry
- players
- bounce candidates
- rally start/end candidates

Only after that should we infer point winner and score.

---

## 3. Why Python for the POC

Even if the final product is not Python, Python is the fastest way to validate the AI/scoring part.

Python gives easy access to:

- OpenCV video processing
- PyTorch models
- YOLO object detection
- existing tennis ball/court tracking projects
- quick visualization and debugging

The final app can later use a Python AI service behind a mobile/web app.

---

## 4. Recommended Repository Structure

```text
tennis-score-poc/
  README.md
  ROADMAP.md
  requirements.txt
  .gitignore

  data/
    input/
      sample_match.mp4
    output/
      annotated_sample_match.mp4
      events.csv
      events.json

  models/
    ball/
    court/
    players/

  src/
    main.py
    config.py

    video/
      video_reader.py
      video_writer.py

    detection/
      ball_detector.py
      player_detector.py
      court_detector.py

    tracking/
      ball_tracker.py
      player_tracker.py

    geometry/
      court_model.py
      homography.py

    events/
      bounce_detector.py
      hit_detector.py
      rally_state.py

    scoring/
      tennis_score.py
      scoring_engine.py

    visualization/
      draw.py

  notebooks/
    experiments.ipynb

  tests/
    test_tennis_score.py
    test_rally_state.py
```

---

## 5. VS Code Setup

### Required VS Code Extensions

Install these extensions:

1. **Python** by Microsoft  
   Python language support, debugging, virtual environments.

2. **Pylance** by Microsoft  
   Better IntelliSense and type checking.

3. **Jupyter** by Microsoft  
   Useful for quick frame/video experiments.

4. **Black Formatter**  
   Automatic code formatting.

5. **isort**  
   Import sorting.

6. **GitLens**  
   Helpful Git history and code tracking.

7. **TODO Highlight**  
   Highlights TODO/FIXME comments.

Optional but useful:

8. **Python Debugger** by Microsoft  
   Better debugging support.

9. **Even Better TOML**  
   Useful if later using `pyproject.toml`.

10. **Markdown All in One**  
   Comfortable editing of this roadmap and docs.

---

## 6. Python Environment

Recommended Python version:

```text
Python 3.10 or 3.11
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Activate on Linux/macOS/Git Bash:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 7. Initial `requirements.txt`

Start with:

```txt
opencv-python
numpy
pandas
matplotlib
scipy
tqdm
ultralytics
torch
torchvision
filterpy
scikit-learn
pytest
black
isort
```

Optional later:

```txt
supervision
lapx
onnxruntime
moviepy
fastapi
uvicorn
```

Notes:

- `opencv-python` is used for video I/O, frame processing, drawing, and image geometry.
- `ultralytics` is used for YOLO experiments.
- `torch` / `torchvision` are needed for many AI models.
- `filterpy` can be used for Kalman filtering.
- `scipy` helps with signal processing and trajectory smoothing.
- `pandas` is convenient for event CSV output.

---

## 8. Candidate AI Models

### 8.1 Ball Detection

Ball detection is the hardest part.

Candidate approaches:

#### Option A: TrackNet-style model

TrackNet is designed for small, fast tennis ball tracking. It uses consecutive frames and produces a heatmap for the ball location.

Pros:

- Designed specifically for tennis ball tracking.
- Better than generic object detection when the ball is tiny or blurry.
- Uses motion/history, not just a single frame.

Cons:

- More setup work.
- May require model conversion or retraining.
- Existing repos may need code cleanup.

Use this if YOLO ball detection is unstable.

#### Option B: YOLO custom tennis ball detector

Use YOLO through Ultralytics.

Pros:

- Easy setup.
- Easy inference.
- Easy to train/fine-tune later.
- Good tooling and documentation.

Cons:

- Small tennis balls are difficult for generic YOLO.
- May miss ball during fast motion or blur.
- False positives possible.

Use this as the first practical experiment because it is simple.

#### Option C: Hybrid detection

Use YOLO/TrackNet candidates, then improve with tracking:

- trajectory continuity
- Kalman filter
- velocity constraints
- court-region filtering
- reject impossible jumps

This is probably the best long-term POC direction.

---

### 8.2 Court Detection

We need court geometry to understand whether the ball is in/out and where players are.

Candidate approaches:

#### Option A: Court keypoint detector

Detect 14 court keypoints, then estimate homography from image coordinates to court coordinates.

Pros:

- Good foundation for scoring.
- Converts image positions into real court positions.
- Enables in/out logic later.

Cons:

- Fence camera angle may differ from broadcast examples.
- Occlusions and partial court visibility may hurt detection.

#### Option B: Manual court calibration for POC

For the first POC, manually click/select court keypoints once at the start of the video.

Pros:

- Much faster.
- Removes court-detection uncertainty.
- Lets us focus on ball tracking and scoring logic.

Cons:

- Not automatic.
- Needs manual setup per video.

Recommended for first milestone:

```text
Use manual court calibration first.
Add automatic court detection later.
```

---

### 8.3 Player Detection

Use YOLO person detection.

We only need rough player positions at first.

For the first version:

- detect people with YOLO
- filter people inside/near the court
- assign near-side player and far-side player by court location
- track them with simple nearest-neighbor or ByteTrack later

Player detection is less critical than ball/court detection for early scoring, but useful for:

- assigning hitter
- identifying server side
- estimating who won the point

---

## 9. Core Algorithms

### 9.1 Video Frame Pipeline

For each frame:

```text
read frame
↓
detect court / use calibrated court
↓
detect ball candidates
↓
track ball over time
↓
detect players
↓
map image points to court coordinates
↓
detect bounce/hit/rally events
↓
update rally state
↓
draw annotations
↓
write output frame
```

---

### 9.2 Court Homography

Once court keypoints are known, compute a homography:

```text
image pixel coordinates → real/top-down court coordinates
```

This enables:

- checking if ball bounce is inside the court
- locating players on court
- normalizing different camera angles
- drawing a mini-court later

For the first POC, the court model can use approximate tennis court dimensions.

Singles court:

```text
length: 23.77 m
singles width: 8.23 m
doubles width: 10.97 m
net at center
service boxes included
```

---

### 9.3 Ball Tracking

Detection alone is not enough. We need temporal tracking.

Recommended first algorithm:

1. Run ball detector per frame.
2. Keep candidate detections.
3. Use Kalman filter to predict next ball position.
4. Choose the detection closest to prediction.
5. Reject detections with impossible speed/jump.
6. Interpolate short missing segments.

Useful signals:

- ball position
- velocity
- acceleration
- confidence
- missing-frame count

---

### 9.4 Bounce Detection

A bounce often appears as a sharp change in vertical image motion or court-plane trajectory.

First simple approach:

1. Smooth ball trajectory.
2. Detect local direction changes.
3. Look for sudden velocity/acceleration change.
4. Keep only candidates near the court plane.
5. Reject candidates outside valid court area.

Important:

From a single camera, bounce detection is approximate. It will probably need tuning and manual validation.

---

### 9.5 Hit Detection

A hit can be estimated from:

- ball trajectory change
- ball speed change
- ball proximity to a player
- timing after bounce
- direction reversal

First simple approach:

```text
if ball direction changes strongly
and ball is near one player
and ball speed changes
then mark possible hit
```

---

### 9.6 Rally State Machine

Implement this early, even before AI is perfect.

Example states:

```text
IDLE
WAITING_FOR_SERVE
SERVE_IN_PROGRESS
RALLY
POINT_ENDED
UNKNOWN
```

Example transitions:

```text
IDLE → WAITING_FOR_SERVE
WAITING_FOR_SERVE → SERVE_IN_PROGRESS
SERVE_IN_PROGRESS → RALLY
RALLY → POINT_ENDED
POINT_ENDED → WAITING_FOR_SERVE
```

At first, transitions can be manual/semi-automatic.

For example:

```text
press key / config timestamp to mark point start
AI suggests point end
human validates result
```

This keeps the POC realistic.

---

### 9.7 Scoring Engine

Separate tennis scoring from computer vision.

The scoring engine should receive simple events:

```json
{
  "event": "point_won",
  "player": "near"
}
```

Then update the score:

```text
0, 15, 30, 40, game
deuce
advantage
set score
match score
```

This part is deterministic and should have unit tests.

Start with:

- normal game
- deuce/advantage
- set score

Add later:

- tie-breaks
- super tie-breaks
- no-ad
- doubles
- custom club formats

---

## 10. Milestones

### Milestone 0 — Project Bootstrap

Goal:

```text
Run a Python script from VS Code that opens a video and writes an annotated copy.
```

Deliverables:

- repo structure
- virtual environment
- requirements installed
- `src/main.py`
- read video
- draw frame number
- write output video

Success criteria:

```text
data/output/annotated_sample_match.mp4 is created.
```

---

### Milestone 1 — Manual Court Calibration

Goal:

```text
Manually define court keypoints and draw court overlay.
```

Deliverables:

- select/store court points
- save calibration JSON
- draw court lines
- compute homography
- draw mini top-down court if useful

Success criteria:

```text
Court overlay stays aligned with the real court throughout the static video.
```

---

### Milestone 2 — Ball Detection Baseline

Goal:

```text
Detect the tennis ball in as many frames as possible.
```

Deliverables:

- YOLO or TrackNet detector wrapper
- detection confidence
- draw ball marker
- save ball detections to CSV

Success criteria:

```text
Ball is visible/marked in a meaningful percentage of rally frames.
```

Do not expect perfection yet.

---

### Milestone 3 — Ball Tracking

Goal:

```text
Convert noisy detections into a continuous ball track.
```

Deliverables:

- Kalman filter or simple tracker
- missing-frame handling
- impossible-jump rejection
- trajectory drawing

Success criteria:

```text
Trajectory is smoother than raw detections and survives short missed detections.
```

---

### Milestone 4 — Player Detection

Goal:

```text
Detect and track near/far players.
```

Deliverables:

- YOLO person detection
- court-region filtering
- near/far player assignment
- player center/feet estimate

Success criteria:

```text
Near and far players are consistently identified in most frames.
```

---

### Milestone 5 — Bounce and Hit Candidates

Goal:

```text
Detect candidate bounces and hits.
```

Deliverables:

- bounce candidate detector
- hit candidate detector
- event timeline CSV/JSON
- annotation on output video

Success criteria:

```text
The output timeline contains reasonable candidate bounce/hit events.
```

---

### Milestone 6 — Rally Segmentation

Goal:

```text
Detect or assist point start/end.
```

Deliverables:

- rally state machine
- point start/end candidates
- event confidence
- ability to manually override through config

Success criteria:

```text
The system can split the video into rally/point segments with acceptable manual correction.
```

---

### Milestone 7 — First Scoring POC

Goal:

```text
Suggest point winner and update score.
```

Deliverables:

- scoring engine
- point_won event
- scoreboard state
- CSV/JSON score timeline
- unit tests for tennis scoring

Success criteria:

```text
Given validated point winners, score is always correct.
AI-suggested point winners are shown separately with confidence.
```

---

## 11. Output Files

The app should produce:

```text
data/output/annotated_sample_match.mp4
data/output/ball_detections.csv
data/output/ball_track.csv
data/output/player_tracks.csv
data/output/events.csv
data/output/events.json
data/output/score_timeline.json
```

Example `events.csv`:

```csv
frame,time_sec,event,player,x_img,y_img,x_court,y_court,confidence
120,4.00,bounce,unknown,912,433,3.1,9.4,0.72
145,4.83,hit,far,870,310,2.7,11.8,0.65
210,7.00,point_end,near,,,,,0.58
```

---

## 12. Suggested First Implementation Order

Do this in order:

1. Create repo and Python environment.
2. Read/write video.
3. Draw frame number and timestamp.
4. Add manual court calibration.
5. Draw calibrated court overlay.
6. Add YOLO person detection.
7. Add ball detector experiment.
8. Add ball tracker.
9. Export CSV.
10. Add simple event candidates.
11. Add tennis score engine.
12. Connect event timeline to scoring.

Do not start with scoring AI directly.

---

## 13. Practical Risks

### Ball is too small or blurry

Mitigation:

- record in 1080p/4K
- use 60 FPS if possible
- keep the full court visible but not too far away
- use TrackNet-style model if YOLO fails

### Fence blocks view

Mitigation:

- test from different fence heights
- place camera close to fence opening if available
- use trajectory filtering to handle short occlusions

### Court lines are hard to detect

Mitigation:

- manual calibration first
- automatic court detection later

### Lighting changes

Mitigation:

- start with daylight videos
- later test evening/night videos separately

### Full automatic scoring may be unreliable

Mitigation:

- build AI-assisted scoring first
- human confirms uncertain point winner
- use manual override in the app later

---

## 14. Recommended AI/Computer Vision Strategy

For the first useful POC:

```text
Manual court calibration
+ YOLO person detection
+ YOLO/TrackNet ball detection
+ Kalman ball tracking
+ rule-based event detection
+ deterministic tennis scoring
```

Avoid end-to-end AI scoring at the beginning.

The system should expose intermediate outputs so we can debug:

- ball detections
- ball track
- court transform
- player tracks
- bounce candidates
- hit candidates
- point-end candidates
- score timeline

---

## 15. Future App Direction

Once the scoring POC works:

### Phase 1 — Local POC

```text
Recorded video → annotated video + score timeline
```

### Phase 2 — Assisted Scoring App

```text
Live stream + manual score buttons + AI suggestions
```

### Phase 3 — Broadcast App

```text
Mobile camera → cloud/live stream → viewers
```

### Phase 4 — AI Scoring Service

```text
Video frames → AI backend → real-time score suggestions
```

Possible architecture:

```text
Mobile app / web app
↓
streaming layer: WebRTC or RTMP/HLS
↓
Python AI service
↓
event/scoring API
↓
score overlay + viewer broadcast
```

---

## 16. Definition of Done for the POC

The POC is successful if we can take one recorded fence-camera tennis video and produce:

1. Annotated video with:
   - court overlay
   - ball marker/trajectory
   - player boxes
   - candidate bounce/hit/point-end events

2. Event files:
   - ball detections
   - ball track
   - event timeline
   - score timeline

3. A clear answer to:

```text
Is the camera angle good enough for AI-assisted scoring?
What parts can be automatic?
What parts need manual confirmation?
```

---

## 17. Immediate Next Task

Create Milestone 0:

```text
src/main.py
```

It should:

1. Load `data/input/sample_match.mp4`.
2. Read frames using OpenCV.
3. Draw frame number and timestamp.
4. Write `data/output/annotated_sample_match.mp4`.

After that, continue to manual court calibration.
