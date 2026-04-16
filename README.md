<div align="center">
  <h1>Neural Nexus</h1>
  <p><b>AI-Powered Classroom Intelligence System</b></p>
  <p><i>Real-time Behavioral Analysis, Engagement Tracking, and Live Camera Streaming Powered by 9 GPU-Accelerated YOLO Models & DeepSORT</i></p>
</div>

---

## Overview

**Neural Nexus** is a production-grade inference system designed to monitor and analyze physical classroom environments. By utilizing a robust sequential computer vision pipeline, it transforms raw classroom video or live webcam feeds into structured, actionable intelligence including student engagement scores, disruptive event tracking, and overall classroom state assessment.

Built on **FastAPI (Backend)** and **Next.js (Frontend UI)**, Neural Nexus delivers real-time analytics to a web interface featuring a modern, NexArch-inspired aesthetic. 

---

## System Architecture

The overarching system leverages a decoupled architecture. The frontend handles interactive file drops, status polling, and live video renders, while the FastAPI backend handles REST routing, WebSocket streaming, and asynchronous AI model processing.

```mermaid
graph TD
    subgraph Frontend [Next.js Web App]
        UI1[Dashboard UI]
        UI2[Upload Page]
        UI3[Live Camera Dashboard]
    end

    subgraph API [FastAPI Backend]
        R1[REST: /api/upload]
        R2[REST: /api/status]
        R3[WS: /ws/camera]
        JM[Background Job Manager]
    end

    subgraph Pipeline [AI Vision Pipeline]
        ME[Model Engine & GPU Memory]
        TR[DeepSORT Tracker]
        SM[Temporal Smoother]
        EE[Event & Behavior Engine]
    end

    UI2 -- "POST Video File" --> R1
    UI1 -- "Poll Hardware state" --> R2
    UI3 -- "Duplex Telemetry" --> R3

    R1 --> JM
    R3 --> Pipeline
    JM -- "Routes Frames" --> Pipeline

    ME --> TR
    TR --> SM
    SM --> EE
```

---

## The AI Vision Pipeline

The core component of Neural Nexus is the modular AI pipeline, meticulously developed to resolve flickering detections and unstable tracking. The pipeline runs sequentially across every processed frame.

```mermaid
journey
    title Per-Frame Processing Lifecycle
    section 1. Visual Ingestion
      Frame Extraction: 5: Extractor
    section 2. Multi-Model Inference
      Person Detection (YOLOv8): 7: Model
      Activity & Object Detection (YOLOv7 x8): 7: Model
    section 3. Contextual Tracking
      DeepSORT Embedder: 6: Tracker
      IoU Entity Mapping: 5: Tracker
    section 4. Cognitive Analysis
      Temporal Smoothing: 4: Buffer
      Behavior & Metrics Engine: 6: Aggregator
```

### 1. Multi-Model Inference
Because classroom environments contain vast diversity, learning specific micro-activities requires multiple fine-tuned models rather than one monolithic network. We cache all 9 models into GPU VRAM (`cuda:0`) on server startup to bypass the 15-second cold-start latency. 

```mermaid
graph LR
    Frame[Raw Image Frame]
    
    subgraph YOLO Stack
        M0[yolov8n - Person Extractor]
        M1[yolov7 - Blackboard/Screen]
        M2[yolov7 - Discussing]
        M3[yolov7 - Hand-raise/Read/Write]
        M4[yolov7 - Standing]
        M5[yolov7 - Talking]
        M6[yolov7 - Teacher Behavior Stage]
        M7[yolov7 - Teacher Locator]
        M8[yolov7 - Core Guide/Answer]
    end
    
    Frame --> M0
    Frame --> M1
    Frame --> M2
    Frame --> M3
    Frame --> M4
    Frame --> M5
    Frame --> M6
    Frame --> M7
    Frame --> M8
    
    M0 --> Dets[Aggregated Entity Bounding Boxes]
    M1 --> Dets
    M2 --> Dets
    M8 --> Dets
```

### 2. Entity Tracking (DeepSORT)
To measure behavior over time, the system uses **DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric)**. Bounding boxes are filtered, matched with historical tracks using an IoU metric, and re-identified via a CNN Embedder (MobileNetV2 running on CPU to save critical GPU VRAM). 

### 3. Temporal Smoothing
Raw frame detections often "flicker" (e.g., a student raising their hand triggers detection on frame 14, drops on frame 15, and resumes on frame 16). The `TemporalSmoother` maintains a rolling memory window (e.g., 30 frames) to require continuous consensus before solidifying an activity as "confirmed".

```mermaid
stateDiagram-v2
    [*] --> RawDetection: Inference Output
    RawDetection --> TemporalBuffer: Inject
    
    state TemporalBuffer {
        [*] --> CheckHistory
        CheckHistory --> ConfirmedActivity: Seen >= 60% of window
        CheckHistory --> Ignored: Transient noise
    }
    
    ConfirmedActivity --> EventEngine: Pass confirmed state
```

### 4. Event & Behavior Engine
Once activities are stabilized, the `EventEngine` infers the overall classroom state by analyzing ratios. If the system detects `Teacher_Guide` + `Talking` + `Interaction`, it registers "Active Engagement".

```mermaid
flowchart TD
    Smoothed[Smoothed Entity States] --> Ratio[Participation Ratios]
    Smoothed --> Teacher[Teacher Context]
    
    Ratio --> Logic{Decision Matrix}
    Teacher --> Logic
    
    Logic -- High Talk/Disruption --> S1[Disrupted / Chaos]
    Logic -- Group clusters --> S2[Collaborative Learning]
    Logic -- Teacher Talking, Students Listening --> S3[Lecture Mode]
    Logic -- High Q&A mapping --> S4[Interactive Discussion]
    Logic -- Low activity --> S5[Idle / Transition]
```

---

## Telemetry Protocol (WebSocket) 

For live camera mode, ensuring latency remains under 400ms is paramount. Neural Nexus pushes a serialized pipeline state back to the browser.
The Frontend natively overlays metrics utilizing `CSS Grid` and React Hooks, ensuring no visible tearing.

**WebSocket Payload Spec:**
```json
{
  "type": "frame",
  "frame": "base64_encoded_jpeg...",
  "metrics": {
    "engagement_score": 0.84,
    "participation_rate": 0.65,
    "disruption_index": 0.05,
    "teacher_interaction_ratio": 0.72
  },
  "classroom_state": "LECTURE_MODE",
  "state_confidence": 0.88,
  "inference_ms": 112.4
}
```

---

## Environment Configuration & Deployment

### Hardware Requirements
- **OS**: Windows / Linux
- **GPU**: NVIDIA GPU with minimum 4GB VRAM (GTX 1650 or higher). *The system actively allocates ~1.5GB merely to cache all 9 uncompressed models.*
- **RAM**: 16 GB System Memory

### Running Locally

Neural Nexus separates the AI backend (FastAPI) and UI frontend (Next.js).

**1. Start the API Server & Pipeline**
```bash
cd backend
pip install -r ../requirements.txt 
python server.py
```

**2. Start the Frontend Application**
```bash
cd frontend
npm install
npm run dev
```
