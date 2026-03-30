# BoxingTracker

A real-time boxing analysis system that uses a single camera and pose estimation to track fighters, detect and classify punches, estimate landing accuracy, and map fighter positions on the ring — all visualised live in a web dashboard.

---

## How It Works

### Overview

The system runs two processes in parallel:

1. **Webcam pipeline** — captures video, runs YOLO pose estimation on every frame, and feeds the results through the punch detection and ring position tracking modules. Events (punches, round markers) are written to disk as JSON.
2. **Dash dashboard** — reads those JSON files every 300ms and renders live counters, charts, heatmaps, and an event log in the browser.

### Punch Detection

Punch detection works in two stages:

**Stage 1 — Proposal (geometry-based).** Each frame, wrist speed, elbow extension, and wrist-to-shoulder displacement are tracked per fighter. When these biomechanical signals cross thresholds, a punch window is opened. The window closes when the wrist decelerates or a maximum duration is reached.

**Stage 2 — Classification (TCN).** When a punch window closes, the 25-frame keypoint sequence around the peak wrist speed is fed into a Temporal Convolutional Network (TCN) that classifies it into one of six punch types: `jab`, `cross`, `lead_hook`, `rear_hook`, `lead_uppercut`, `rear_uppercut`.

### Landing Detection

For each classified punch, the system checks whether it landed by combining two signals:
- **Proximity** — the attacker's wrist must be within striking range of the defender's head or torso at peak wrist speed.
- **Reaction** — the defender's head or torso must displace by a meaningful amount in the window around the punch.

A punch is marked as landed if the wrist is very close (contact certain), or if it is within striking range AND the defender reacted.

### Ring Position Tracking

The user calibrates the ring once by clicking its four corners on a live frame. This computes a perspective homography that maps any image pixel to a flat, top-down ring coordinate system. Each frame, each fighter's ground contact point (estimated from ankle/hip keypoints) is projected through this homography and accumulated into a 5×5 heatmap grid showing where in the ring each fighter has spent time.

### Dashboard

The Dash dashboard displays:
- **Counters** — total punches, landed, missed
- **Punch type chart** — stacked bar chart showing landed vs missed per punch type
- **Ring heatmaps** — per-fighter 5×5 position grids, colour-mapped (black → red → white)
- **Event log** — full chronological log of all punches and round markers

---

## Project Structure

```
BoxingTracker/
├── run.py                              # Entry point — starts both processes
├── app/
│   └── dash/
│       └── dashboard.py               # Live Dash dashboard
├── logic/
│   └── models/
│       ├── ring/
│       │   ├── calibration.py         # Homography computation and ring calibration UI
│       │   └── position.py            # Ground point estimation and heatmap accumulation
│       └── visual/
│           ├── main.py                # Camera selection and pipeline entry point
│           ├── device.py              # GPU/CPU model selection
│           ├── posing/
│           │   └── coco_poses.py      # COCO-17 keypoint index mapping
│           ├── punches/
│           │   ├── proposal.py        # Geometry-based punch window detection
│           │   ├── features.py        # Pose feature extraction for the TCN
│           │   ├── TCN_model.py       # TCN architecture
│           │   ├── TCN_train.py       # Training script
│           │   ├── checkpoints/       # Saved model weights (tcn_best.pt, tcn_last.pt)
│           │   └── data/
│           │       ├── labels/        # Annotated punch labels (.xlsx, one per video)
│           │       └── skeleton/      # Extracted skeleton clips (.npy, one per video)
│           ├── tracking/
│           │   └── botsort_boxing.yaml # Tracker configuration
│           └── yolo/
│               └── yoloposewcam.py    # Main webcam processing loop
├── tools/
│   ├── extract_skeletons.py           # Extract YOLO skeletons from a video file
│   ├── annotate_punches.py            # Interactive punch labelling tool
│   ├── extract_all.bat                # Batch skeleton extraction
│   └── annotate_all.bat               # Batch annotation launcher
└── artifacts/                         # Runtime output (gitignored)
    ├── state.json                     # Live dashboard state
    ├── events.json                    # Full punch/round event log
    └── ring_calibration.json          # Saved ring homography
```

---

## Installation

**Requirements:** Python 3.10+, a YOLO pose model weight file (e.g. `yolo11l-pose.pt`).

```bash
# Clone the repo
git clone https://github.com/Daryox/BoxingTracker.git
cd BoxingTracker

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux

# Install dependencies
pip install ultralytics opencv-python dash plotly numpy pandas openpyxl torch
```

Download a YOLO pose model weight and place it in the project root:
```
yolo11l-pose.pt   ← recommended (best accuracy)
yolo11s-pose.pt   ← faster, lower accuracy
```

---

## Running the App

```bash
python run.py
```

This starts the webcam pipeline and the dashboard simultaneously. Open **http://127.0.0.1:8050** in a browser to see the live dashboard.

### Camera Controls (OpenCV window)

| Key | Action |
|-----|--------|
| `c` | Calibrate ring — click TL, TR, BR, BL corners then press `s` |
| `l` | Reload saved ring calibration |
| `h` | Toggle heatmap inset overlay |
| `[` | Start round |
| `]` | End round |
| `r` | Reset session (clears all counters and heatmaps) |
| `q` | Quit |

**Ring calibration is required before any position tracking works.** Do it once at the start of each session unless a saved calibration already exists for your camera angle.

---

## Camera Setup Tips

For best results:

- **Use a single static camera.** The homography is computed once at calibration time. If the camera moves, all ring position data becomes invalid and you must recalibrate.
- **Mount the camera at an elevated angle** — ideally at ringside, slightly above head height. Avoid mounting directly overhead (top-down) as pose estimation degrades significantly.
- **Keep the camera stable.** Use a tripod. Hand-held or mounted-on-ropes cameras will produce noisy position data.
- **Avoid crowded frames.** The system tracks a maximum of two fighters. Spectators, coaches, or referees in frame will be picked up by the tracker and create spurious punch detections. Clear the ring area of bystanders where possible.
- **Ensure good lighting.** YOLO pose estimation degrades in low light. Avoid backlighting (camera facing a bright window). Consistent, even lighting produces the most reliable keypoints.
- **fighters should wear contrasting colours** where possible — this helps the tracker maintain stable IDs during clinches.
- **Minimum resolution: 720p.** The pipeline runs at 1280×720. Lower resolutions will reduce keypoint accuracy.

---

## Training the Punch Classifier

### Adding New Training Data

1. **Record a video** of punching drills or sparring.
2. **Extract skeletons** from the video:
   ```bash
   python tools/extract_skeletons.py path/to/video.mp4 --id V18
   ```
   This saves a raw skeleton array to `logic/models/visual/punches/data/raw/V18_raw.npy`.
3. **Annotate punches** using the interactive tool:
   ```bash
   python tools/annotate_punches.py path/to/video.mp4 --id V18
   ```
   Use the keyboard controls to scrub the video, mark punch start/end frames, and assign punch types. The tool saves:
   - `logic/models/visual/punches/data/skeleton/V18.npy` — clip keypoint sequences
   - `logic/models/visual/punches/data/labels/V18.xlsx` — punch labels

### Annotator Controls

| Key | Action |
|-----|--------|
| `SPACE` | Play / pause |
| `LEFT` / `RIGHT` | Step one frame |
| `,` / `.` | Jump 5 frames |
| `TAB` | Toggle active fighter (sparring mode) |
| `S` | Set punch start at current frame |
| `E` | Set punch end and assign type (prompts 1–6) |

Punch types: `1`=jab, `2`=cross, `3`=lead_hook, `4`=rear_hook, `5`=lead_uppercut, `6`=rear_uppercut

### Retraining

Once new data is annotated, retrain the TCN from the project root:

```bash
python -m logic.models.visual.punches.TCN_train
```

The best checkpoint is saved to `logic/models/visual/punches/checkpoints/tcn_best.pt` and is automatically used by the pipeline on the next run.

### Current Class Distribution

| Punch Type | Samples | % |
|---|---|---|
| jab | 1538 | 28.7% |
| cross | 1487 | 27.8% |
| lead_hook | 1011 | 18.9% |
| rear_uppercut | 518 | 9.7% |
| lead_uppercut | 472 | 8.8% |
| rear_hook | 328 | 6.1% |

`rear_hook` is the least represented class. Prioritise recording and annotating rear hook sequences when adding new data.

---

## Known Limitations

- **Depth ambiguity** — the system works in 2D. A near-miss thrown straight at the camera looks identical in the image to a punch that landed. Proximity-based landing detection is therefore an approximation.
- **Single camera** — no stereo depth information. Ring position accuracy degrades near the edges of the ring where perspective distortion is highest, even after homography correction.
- **Class imbalance** — `rear_hook` has significantly fewer training samples than `jab`/`cross`. The classifier may under-perform on this class.
- **Re-identification** — during prolonged clinches or when fighters leave the frame, the tracker may assign new IDs. The slot mapper absorbs most of these but edge cases can cause heatmap or counter drift.
- **No glove detection** — punch detection relies entirely on wrist keypoints from pose estimation, not on detecting the glove itself. Loose or baggy clothing can affect keypoint confidence.
