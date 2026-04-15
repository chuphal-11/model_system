# Neural Nexus — AI Classroom Intelligence System

## Goal

Build a complete backend pipeline that transforms classroom video into structured behavioral insights using 8 custom YOLOv5 models. The pipeline follows the architecture defined in `research.md`:

**Video → Frames → YOLO Detection → Tracking → Temporal Smoothing → Events → Behavior Inference → Metrics → JSON Output**

---

## Key Discovery: Model Details

The 8 custom `.pt` models are **YOLOv5-based** with a custom `MP` (Max Pooling) layer. They cannot be loaded with stock `ultralytics` or standard `yolov5` — we need to patch the loading to inject the `MP` class.

| Model File | Detected Classes |
|---|---|
| `2_BlackBoard_Sreen_Teacher.pt` | `screen`, `blackBoard`, `teacher` |
| `3_Discuss.pt` | `discuss` |
| `4_Handrise_Read_write.pt` | `hand-raising`, `read`, `write` |
| `5_Stand.pt` | `stand` |
| `6_Talk.pt` | `talk` |
| `7_Talk_Teacher_Behavior.pt` | `talk`, `guide`, `answer`, `On-stage interaction`, `blackboard-writing` |
| `8_Teacher.pt` | `teacher` |
| `9_Teacher_Behavior.pt` | `guide`, `answer`, `On-stage interaction`, `blackboard-writing` |

Additionally, `yolov8n.pt` (COCO pretrained) is available for person-level detection/counting.

---

## User Review Required

> [!IMPORTANT]
> The custom models require a patched YOLOv5 loading mechanism (injecting the missing `MP` class). This approach monkey-patches `torch.load` to intercept the missing module. If this fails at runtime, we may need the original training codebase.

> [!WARNING]
> Since the models use a custom architecture, there's no guarantee they'll load correctly without the original training code. The system is designed with a **graceful degradation** approach — if a model fails to load, it's skipped and the pipeline continues with remaining models.

---

## Proposed Changes

### Project Structure

```
NEURAL NEXUS/
├── models/                          # Existing model files
├── research.md                      # Existing research doc
├── instruction.md                   # Existing
├── config.py                        # [NEW] Configuration & constants
├── main.py                          # [NEW] CLI entry point
├── pipeline/                        # [NEW] Core pipeline package
│   ├── __init__.py
│   ├── frame_extractor.py           # Video → frames
│   ├── detector.py                  # YOLO model loading & inference
│   ├── tracker.py                   # SORT-based object tracking
│   ├── temporal_smoother.py         # Sliding window smoothing
│   ├── event_engine.py              # Detection → semantic events
│   ├── behavior_engine.py           # Events → classroom state
│   └── metrics.py                   # Quantified metrics computation
├── utils/                           # [NEW] Utilities
│   ├── __init__.py
│   ├── model_loader.py              # Patched YOLOv5 model loading
│   └── visualization.py             # Optional annotated frame output
├── output/                          # [NEW] Output directory (auto-created)
└── requirements.txt                 # [NEW] Dependencies
```

---

### Configuration (`config.py`)

#### [NEW] [config.py](file:///home/shiro/Downloads/NEURAL%20NEXUS/config.py)

Central configuration file containing:
- Model paths and their class name mappings
- Temporal smoothing parameters (window size = 15 frames, threshold = 0.6)
- Event extraction rules and thresholds
- Behavior inference rule matrix
- Metrics computation weights
- Output format settings

---

### Utilities

#### [NEW] [model_loader.py](file:///home/shiro/Downloads/NEURAL%20NEXUS/utils/model_loader.py)

Handles loading the custom YOLOv5 models by:
1. Cloning the YOLOv5 repository (if not cached) or using installed `yolov5` package
2. Injecting the missing `MP` class into `models.common`
3. Loading each `.pt` file with `torch.load` using the patched module namespace
4. Extracting model metadata (class names, anchors, stride)
5. Returning a callable wrapper for inference

Falls back to `ultralytics` YOLO for `yolov8n.pt`.

---

### Pipeline Modules

#### [NEW] [frame_extractor.py](file:///home/shiro/Downloads/NEURAL%20NEXUS/pipeline/frame_extractor.py)

- Opens video file or camera stream via OpenCV
- Yields `(frame_number, timestamp, frame_image)` tuples
- Supports configurable FPS sampling (skip frames for efficiency)
- Handles both file and webcam input

#### [NEW] [detector.py](file:///home/shiro/Downloads/NEURAL%20NEXUS/pipeline/detector.py)

- Manages all 8 YOLO models + the person detector
- Runs each model on every frame (or a subset for performance)
- Merges all detections into a unified format:
  ```python
  Detection(bbox, class_name, confidence, source_model)
  ```
- Applies confidence thresholds per model
- Uses NMS to handle overlapping detections from different models

#### [NEW] [tracker.py](file:///home/shiro/Downloads/NEURAL%20NEXUS/pipeline/tracker.py)

- Implements SORT (Simple Online and Realtime Tracking) algorithm
- Assigns persistent IDs to detected persons using IoU-based matching
- Uses Kalman filtering for motion prediction
- Maps activity detections to tracked person IDs via bounding box overlap
- Output: `TrackedEntity(id, bbox, activities: dict[str, float])`

#### [NEW] [temporal_smoother.py](file:///home/shiro/Downloads/NEURAL%20NEXUS/pipeline/temporal_smoother.py)

- Maintains a sliding window buffer per tracked entity (default: 15 frames)
- For each entity, stores the detection history for each activity class
- Applies majority voting with configurable threshold (default: 60%)
- Produces stable activity labels, filtering out single-frame noise
- Output: `SmoothedState(entity_id, confirmed_activities, stability_scores)`

#### [NEW] [event_engine.py](file:///home/shiro/Downloads/NEURAL%20NEXUS/pipeline/event_engine.py)

Converts smoothed detections into semantic events using rule-based logic:

| Pattern | Event |
|---|---|
| Continuous hand-raising (>3s) | `student_wants_to_answer` |
| Talking without teacher engagement | `side_conversation` |
| Sustained standing (>5s) | `potential_disruption` |
| Read/write detected | `active_learning` |
| Discuss detected | `group_discussion` |
| Teacher + blackboard | `lecture_in_progress` |
| Teacher guide/answer | `teacher_student_interaction` |

#### [NEW] [behavior_engine.py](file:///home/shiro/Downloads/NEURAL%20NEXUS/pipeline/behavior_engine.py)

Combines individual events into classroom-level behavioral states:

| Conditions | Classroom State |
|---|---|
| High hand-raises + active teacher | `interactive` |
| High talking + low participation | `distracted` |
| Teacher at board + low student activity | `lecture_mode` |
| No teacher + high activity | `uncontrolled` |
| High read/write + low disruption | `focused_learning` |
| Group discussion detected | `collaborative` |

Also tracks:
- Student behavior signals: participation, distraction, inactivity
- Teacher behavior signals: teaching, engagement, absence

#### [NEW] [metrics.py](file:///home/shiro/Downloads/NEURAL%20NEXUS/pipeline/metrics.py)

Computes quantified metrics over configurable time windows:

- **Engagement Score** (0–1): weighted combination of hand-raises, writing, reading activity
- **Participation Rate** (0–1): active students / total detected students
- **Disruption Index** (0–1): based on standing, random talking, movement
- **Teacher Interaction Ratio** (0–1): time interacting vs lecturing

---

### Entry Point

#### [NEW] [main.py](file:///home/shiro/Downloads/NEURAL%20NEXUS/main.py)

CLI interface supporting:
```bash
# Process a video file
python main.py --input video.mp4 --output output/results.json

# Process from webcam
python main.py --input 0 --output output/results.json

# With visualization (annotated frames saved)
python main.py --input video.mp4 --output output/results.json --visualize
```

Orchestrates the full pipeline: frame extraction → detection → tracking → smoothing → events → behavior → metrics → JSON output.

Output JSON format:
```json
{
  "video": "classroom.mp4",
  "total_frames": 1500,
  "fps": 30,
  "duration_seconds": 50.0,
  "timeline": [
    {
      "timestamp": "00:00:10",
      "frame": 300,
      "class_state": "interactive",
      "engagement_score": 0.72,
      "participation_rate": 0.58,
      "disruption_index": 0.21,
      "teacher_interaction_ratio": 0.65,
      "active_events": ["hand_raise", "teacher_interaction"],
      "tracked_entities": 12,
      "student_summary": {
        "participating": 7,
        "distracted": 2,
        "inactive": 3
      }
    }
  ],
  "aggregate": {
    "avg_engagement": 0.68,
    "avg_participation": 0.55,
    "avg_disruption": 0.18,
    "dominant_state": "interactive",
    "state_distribution": {
      "interactive": 0.4,
      "lecture_mode": 0.35,
      "focused_learning": 0.2,
      "distracted": 0.05
    }
  }
}
```

---

### Utilities

#### [NEW] [visualization.py](file:///home/shiro/Downloads/NEURAL%20NEXUS/utils/visualization.py)

Optional module for debugging:
- Draws bounding boxes with track IDs and activity labels on frames
- Color-coded by activity type
- Saves annotated frames or video

---

## Open Questions

> [!IMPORTANT]
> **Model Loading**: The custom models have a non-standard `MP` layer. If the monkey-patching approach fails, would you be able to provide the original training repository or the `MP` class definition?

> [!NOTE]
> **Performance vs Accuracy**: Running all 8 models on every frame will be slow. Should we prioritize accuracy (run all models every frame) or speed (run models on alternating frames or subsample)?

---

## Verification Plan

### Automated Tests
1. Run `python main.py --help` to verify CLI works
2. Test with a sample video: `python main.py --input sample.mp4 --output output/test.json`
3. Verify JSON output matches expected schema
4. Test graceful degradation when models fail to load

### Manual Verification
- Inspect JSON output for reasonable metric values
- If `--visualize` is enabled, check annotated frames for correct bounding boxes and labels
