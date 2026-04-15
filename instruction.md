# instruction.md — Build Guide for AI Classroom Intelligence System (Backend Only)

## 0. Objective

Build a fully working backend pipeline that processes classroom video using multiple YOLO models and converts detections into structured behavioral insights.

⚠️ This phase focuses ONLY on workflow and logic.
❌ No frontend
❌ No web integration

---

## 1. Tech Stack

* Python 3.10+
* PyTorch
* Ultralytics YOLO (for loading `.pt`)
* OpenCV (video processing)
* NumPy
* collections.deque (temporal buffer)

Optional:

* SORT / DeepSORT (tracking)

---

## 2. Project Structure

Create this exact structure:

```bash
project/
│
├── models/
│   ├── talk.pt
│   ├── handraise.pt
│   ├── stand.pt
│   ├── teacher.pt
│   └── ...
│
├── src/
│   ├── main.py
│   ├── config.py
│   ├── detector.py
│   ├── tracker.py
│   ├── temporal.py
│   ├── event_engine.py
│   ├── behavior_engine.py
│   ├── metrics.py
│   └── utils.py
│
├── data/
│   └── sample_video.mp4
│
└── outputs/
    └── logs.json
```

---

## 3. Step-by-Step Implementation

---

### STEP 1 — Config Setup

Create `config.py`

Define:

* model paths
* confidence thresholds
* temporal window size
* event thresholds

Example:

```python
MODEL_PATHS = {
    "talk": "models/talk.pt",
    "handraise": "models/handraise.pt",
    "stand": "models/stand.pt"
}

CONF_THRESHOLD = 0.5
TEMPORAL_WINDOW = 10
EVENT_THRESHOLD = 6
```

---

### STEP 2 — Detection Module

File: `detector.py`

Responsibilities:

* load all YOLO models once
* run inference per frame
* return structured detections

Output format:

```python
[
  {"class": "talk", "conf": 0.8, "bbox": [...]},
  {"class": "handraise", "conf": 0.9, "bbox": [...]}
]
```

⚠️ Do NOT mix logic here. Only detection.

---

### STEP 3 — Tracking Module

File: `tracker.py`

Responsibilities:

* assign IDs to detected persons
* maintain identity across frames

Output:

```python
[
  {"id": 1, "class": "talk"},
  {"id": 2, "class": "handraise"}
]
```

⚠️ If you skip tracking, your system becomes unreliable.

---

### STEP 4 — Temporal Module

File: `temporal.py`

Use:

```python
from collections import deque
```

Responsibilities:

* store last N frames per ID
* smooth predictions

Logic:

```python
if last_10_frames.count("talk") > threshold:
    return "talk"
```

⚠️ This removes flickering.

---

### STEP 5 — Event Engine

File: `event_engine.py`

Convert smoothed detections → events

Examples:

```python
if handraise_stable:
    event = "student_answering"

if talk_without_teacher:
    event = "side_conversation"

if standing_long:
    event = "disruption"
```

Output:

```python
["student_answering", "disruption"]
```

---

### STEP 6 — Behavior Engine

File: `behavior_engine.py`

Combine events → classroom state

Examples:

```python
if many_handraise and teacher_present:
    state = "interactive"

elif high_talking:
    state = "distracted"

elif teacher_at_board:
    state = "lecture_mode"
```

Output:

```python
{
  "class_state": "interactive"
}
```

---

### STEP 7 — Metrics Module

File: `metrics.py`

Compute:

* engagement_score
* participation_rate
* disruption_index

Example:

```python
engagement = (handraise + writing) - talking
```

Return normalized values (0–1)

---

### STEP 8 — Main Pipeline

File: `main.py`

Responsibilities:

* load video
* loop through frames
* call all modules in order

Pipeline:

```python
frame → detector → tracker → temporal → events → behavior → metrics
```

Store output:

```python
{
  "timestamp": t,
  "state": "...",
  "metrics": {...}
}
```

Save to:
`outputs/logs.json`

---

## 4. Execution Flow

1. Load all models
2. Open video stream
3. For each frame:

   * run detection
   * track objects
   * update temporal buffer
   * extract events
   * infer behavior
   * compute metrics
4. Save results

---

## 5. Testing Strategy

Start simple:

* run on short video (10–20 sec)
* print outputs

Then:

* verify temporal smoothing
* verify event triggers
* verify state transitions

---

## 6. Rules You Must Follow

* Keep modules independent
* Do NOT mix detection with logic
* Do NOT skip temporal smoothing
* Do NOT hardcode everything in main.py
* Always return structured outputs

---

## 7. Minimum Working Output

At the end, system must produce:

```json
{
  "timestamp": "...",
  "class_state": "interactive",
  "engagement_score": 0.7,
  "events": ["handraise", "teacher_present"]
}
```

If you don’t reach this → your pipeline is incomplete.

---

## 8. Future (Not Now)

* UI dashboard
* real-time alerts
* database storage
* deployment

Ignore these for now.

---

## 9. Final Note

This system is NOT about running YOLO.

It is about:

* stabilizing predictions
* interpreting behavior
* generating insights

If you treat it like detection → weak project
If you build full pipeline → strong system
