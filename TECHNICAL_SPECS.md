# Technical Specifications: AI Pipeline Models & Architecture

This document breaks down the deep technical methodology, parameters, and structural approach that the Neural Nexus AI system relies on to reliably assess classroom intelligence. 

## 1. Multi-Model Detection Inference Strategy

Rather than forcing a single YOLO network to learn overlapping and contradictory generic behavioral classes, the architecture isolates concerns across multiple finely tuned YOLOv7 expert heads alongside a centralized YOLOv8 person localizer.

### Why 9 Models?
Micro-actions in a classroom context (such as distinguishing *writing* versus *reading*) introduce high loss when combined with macro-scale layout detection (like *blackboard* or *screen* identification) on a single unified head. By utilizing discrete specialized models, we prevent network forgetting and drastically reduce False Positive detection merging. 

### The Expert Heads
All custom models are loaded seamlessly into GPU VRAM (via CUDA) utilizing customized namespace manipulation to dynamically recreate registered buffer grids. 

| Model Alias | Base Arch | Classes Extracted | Threshold | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `Person Detector` | YOLOv8n | `person` | `0.30` | Baseline localization and tracking anchor. |
| `Model 2` | YOLOv7 | `screen`, `blackBoard`, `teacher` | `0.25` | Environment structure extraction. |
| `Model 3` | YOLOv7 | `discuss` | `0.25` | Peer-to-peer micro-behavior. |
| `Model 4` | YOLOv7 | `hand-raising`, `read`, `write` | `0.25` | Student focal-driven activities. |
| `Model 5` | YOLOv7 | `stand` | `0.25` | Spatial/kinematic mapping. |
| `Model 6` | YOLOv7 | `talk` | `0.25` | Vocal metric inference mapping. |
| `Model 7` | YOLOv7 | *Teacher action variants* | `0.25` | Holistic combination tracking. |
| `Model 8` | YOLOv7 | `teacher` | `0.25` | Verification of authority figure instance. |
| `Model 9` | YOLOv7 | `guide`, `answer`, `on-stage` | `0.25` | Detailed instructional modalities. |

---

## 2. Tracking Approach: DeepSORT Implementation

Traditional IOUs (Intersection over Union) fail dramatically in classrooms due to occlusions (students sitting behind students) and identity-swapping when students cross paths. 

The system implements computationally decoupled **DeepSORT** (Simple Online and Realtime Tracking with a Deep Association Metric) via the `deep_sort_realtime` repository:

### Spatial vs Temporal Feature Extraction
1. **Kinematic Tracker Matrix**: Solved via Kalman Filtering over bbox coordinates `[x, y, a, h]`.
2. **Appearance Matrix (Embedder)**: To distinguish identities, tightly cropped bounding boxes are pushed through a `MobileNetV2` CNN. The generated 128-dimensional embedding acts as an identity fingerprint. 
    * *Optimization Strategy:* To bypass CUDA `bad allocation` VRAM exhaustion caused by the 9 YOLO tensors simultaneously caching memory, the MobileNetV2 embedder is strictly delegated to the **CPU**.

### Tuning Specifications (`SORTTracker`)
* `max_age=30`: An entity can be occluded/lost for up to 30 frames (1 full second at 30fps) before being deleted.
* `min_hits=3`: Track tentatively ignores noise; requires 3 consecutive frames before an identity is officially validated.
* `max_cosine_distance=0.2`: Extremely stringent visual-matching metric; enforces minimal visual drift across identities. 
* `nn_budget=100`: The DeepSORT matrix keeps up to 100 historical descriptor embeddings per track to compare against frame variants. 

---

## 3. Signal Denoising: The Temporal Smoother

Frame-by-frame object detection yields high volatility (Precision/Recall stuttering). A student lowering their hand momentarily should not slice a `hand-raising` event into multiple erratic chunks. 

**The Algorithm (`TemporalSmoother`)**:
The smoother allocates a sliding queue for each uniquely tracked `ID`. It operates as a high-pass frequency filter.

* **Parameter `window_size = 30`**: The filter looks uniformly across the last 30 frames.
* **Parameter `threshold = 0.60`**: For an activity to graduate to a mathematically `CONFIRMED` state during inference, the YOLO models must have detected that activity in **at least 60% of the frames within the sliding window**. 

This completely eliminates split-second False Positives in background static. 

---

## 4. Behavior Engine & Macro Metric Math

Once vectors move from raw bounding-boxes -> into tracked identities -> into smoothed confirmed sequences, they pass to the `BehaviorEngine` and `MetricsComputer` for deductive logic mapping.

### Scoring Indices formula sets
* **Engagement Score**: Evaluates positive traits (`read`, `write`, `hand-raising`, `discuss`) and divides by the total tracked students. A logarithmic penalty subtracts from this score based on continuous negative triggers (`stand`, unprompted `talk`).
* **Teacher Interaction Ratio**: Measures intersection: The ratio of time the teacher is exhibiting `guide` or `answer` proportional to the exact synchronized time students are executing `hand-raising`.
* **State Classification Matrix**:
  * If `Disruption Metric > 0.4` -> Classroom State switches to `DISRUPTED`.
  * If Group-Discuss exceeds threshold -> `COLLABORATIVE_LEARNING`.
  * If Teacher `on-stage` is active with low student physical anomalies -> `LECTURE_MODE`.

This deterministic hierarchy ensures qualitative classroom reports are completely shielded from hallucination, relying strictly on visual vector consensus. 
