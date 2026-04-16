#!/usr/bin/env python3
"""
Neural Nexus — FastAPI Backend
==================================
REST API + WebSocket server for the AI Classroom Intelligence System.

Features:
  - POST /api/upload        → Upload video for background processing
  - GET  /api/jobs/{id}     → Poll job status
  - GET  /api/jobs/{id}/results → Download results JSON
  - GET  /api/jobs/{id}/video   → Download annotated video
  - WS   /ws/camera         → Live camera stream with annotated frames
  - GET  /api/status        → System info (GPU, models)

Usage:
    cd backend && uvicorn server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, PROJECT_ROOT)

import config
from pipeline.frame_extractor import FrameExtractor
from pipeline.detector import MultiModelDetector
from pipeline.tracker import SORTTracker
from pipeline.temporal_smoother import TemporalSmoother
from pipeline.event_engine import EventEngine
from pipeline.behavior_engine import BehaviorEngine
from pipeline.metrics import MetricsComputer
from utils.visualization import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)-20s │ %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logger = logging.getLogger("neural_nexus.server")

UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")
RESULTS_DIR = os.path.join(BACKEND_DIR, "output", "jobs")
TEMPLATES_DIR = os.path.join(BACKEND_DIR, "templates")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class PipelineManager:
    """
    Singleton that loads all YOLO models once at startup and provides
    factory methods for creating per-session pipeline components.
    """

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.detector: Optional[MultiModelDetector] = None
        self.visualizer = Visualizer()
        self._loaded = False

    def load_models(self):
        """Load all models into GPU/CPU memory."""
        if self._loaded:
            return
        logger.info(f"Loading models on device: {self.device}")
        self.detector = MultiModelDetector(device=self.device)
        self.detector.load_models()
        self._loaded = True
        logger.info(f"Pipeline ready — {self.detector.num_models_loaded} models loaded")

    @property
    def is_ready(self):
        return self._loaded and self.detector is not None

    def create_tracker(self):
        return SORTTracker(
            max_age=config.TRACKER_MAX_AGE,
            min_hits=config.TRACKER_MIN_HITS,
            iou_threshold=config.TRACKER_IOU_THRESHOLD,
        )

    def create_smoother(self):
        return TemporalSmoother(
            window_size=config.SMOOTHING_WINDOW_SIZE,
            threshold=config.SMOOTHING_THRESHOLD,
        )

    def create_event_engine(self, fps):
        return EventEngine(fps=fps)

    def create_behavior_engine(self):
        return BehaviorEngine()

    def create_metrics_computer(self, fps):
        return MetricsComputer(fps=fps, window_seconds=10.0)

    def get_system_info(self):
        info = {
            "device": self.device,
            "models_loaded": self.detector.num_models_loaded if self.detector else 0,
            "ready": self.is_ready,
            "gpu_name": None,
            "gpu_memory_total_mb": None,
            "gpu_memory_used_mb": None,
        }
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2
            )
            info["gpu_memory_used_mb"] = round(
                torch.cuda.memory_allocated(0) / 1024**2
            )
        return info


pipeline = PipelineManager()


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class Job:
    def __init__(self, job_id: str, filename: str, input_path: str):
        self.id = job_id
        self.filename = filename
        self.input_path = input_path
        self.status = JobStatus.QUEUED
        self.progress = 0
        self.total_frames = 0
        self.results: Optional[dict] = None
        self.results_path: Optional[str] = None
        self.video_path: Optional[str] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now().isoformat()
        self.completed_at: Optional[str] = None


jobs: dict[str, Job] = {}


def format_timestamp(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def process_video_job(job: Job):
    """Run the full pipeline on an uploaded video (runs in a background thread)."""
    try:
        job.status = JobStatus.PROCESSING
        logger.info(f"Job {job.id}: Processing {job.filename}")

        job_dir = os.path.join(RESULTS_DIR, job.id)
        os.makedirs(job_dir, exist_ok=True)

        tracker = pipeline.create_tracker()
        smoother = pipeline.create_smoother()
        behavior_engine = pipeline.create_behavior_engine()
        visualizer = pipeline.visualizer

        extractor = FrameExtractor(source=job.input_path, sample_rate=1)

        with extractor:
            fps = extractor.fps or 30.0
            job.total_frames = extractor.total_frames

            event_engine = pipeline.create_event_engine(fps=fps)
            metrics_computer = pipeline.create_metrics_computer(fps=fps)

            vis_path = os.path.join(job_dir, "annotated.mp4")
            video_writer = None
            if extractor.width > 0:
                video_writer = Visualizer.create_video_writer(
                    vis_path, fps, extractor.width, extractor.height
                )

            timeline = []
            metrics_interval = max(int(config.METRICS_INTERVAL_SEC * fps), 1)
            frame_count = 0
            total_inference_ms = 0.0
            t_start = time.time()

            for frame_num, timestamp, frame in extractor:
                frame_count += 1
                job.progress = frame_count

                det_result = pipeline.detector.detect(frame)
                total_inference_ms += det_result["inference_time_ms"]
                tracked = tracker.update(
                    det_result["person_detections"],
                    det_result["activity_detections"],
                    frame=frame,
                )
                smoothed = smoother.update(tracked)

                teacher_present = any(
                    "teacher" in e.get("confirmed_activities", {})
                    for e in smoothed
                )
                teacher_engaging = any(
                    any(a in e.get("confirmed_activities", {})
                        for a in ["guide", "answer", "On-stage interaction"])
                    for e in smoothed
                )

                events = event_engine.extract_events(
                    smoothed, timestamp=timestamp,
                    teacher_present=teacher_present,
                    teacher_engaging=teacher_engaging,
                )
                event_summary = event_engine.get_event_summary(events)
                behavior = behavior_engine.infer(smoothed, events, event_summary)
                metrics = metrics_computer.update(smoothed, behavior)

                if frame_count % metrics_interval == 0:
                    timeline.append({
                        "timestamp": format_timestamp(timestamp),
                        "frame": frame_num,
                        "seconds": round(timestamp, 2),
                        "classroom_state": behavior["classroom_state"],
                        "state_confidence": behavior["state_confidence"],
                        "engagement_score": metrics["engagement_score"],
                        "participation_rate": metrics["participation_rate"],
                        "disruption_index": metrics["disruption_index"],
                        "teacher_interaction_ratio": metrics["teacher_interaction_ratio"],
                        "active_events": event_summary.get("active_events", []),
                        "tracked_entities": len(tracked),
                        "student_summary": behavior["student_signals"],
                        "teacher_signals": behavior["teacher_signals"],
                    })

                if video_writer:
                    vis_frame = visualizer.annotate_frame(
                        frame,
                        smoothed_entities=smoothed,
                        behavior_result=behavior,
                        metrics=metrics,
                        events=events,
                        raw_detections=det_result["activity_detections"],
                    )
                    video_writer.write(vis_frame)

            if video_writer:
                video_writer.release()

        elapsed = time.time() - t_start
        aggregate = metrics_computer.get_aggregate_metrics()

        results = {
            "job_id": job.id,
            "video": job.filename,
            "total_frames_processed": frame_count,
            "fps": round(fps, 1),
            "duration_seconds": round(extractor.duration_seconds, 2),
            "processing_time_seconds": round(elapsed, 2),
            "avg_inference_ms": round(total_inference_ms / max(frame_count, 1), 1),
            "device": pipeline.device,
            "models_loaded": pipeline.detector.num_models_loaded,
            "timeline": timeline,
            "aggregate": aggregate,
        }

        results_path = os.path.join(job_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        job.results = results
        job.results_path = results_path
        job.video_path = vis_path if os.path.exists(vis_path) else None
        job.status = JobStatus.DONE
        job.completed_at = datetime.now().isoformat()
        logger.info(f"Job {job.id}: Complete — {frame_count} frames in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"Job {job.id}: Failed — {e}", exc_info=True)
        job.status = JobStatus.ERROR
        job.error = str(e)


app = FastAPI(
    title="Neural Nexus — AI Classroom Intelligence",
    version="1.0.0",
)

templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.on_event("startup")
async def startup():
    """Load all models when the server starts."""
    pipeline.load_models()


@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def system_status():
    return pipeline.get_system_info()


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    if not pipeline.is_ready:
        return JSONResponse({"error": "Models not loaded yet"}, status_code=503)

    allowed = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return JSONResponse(
            {"error": f"Invalid file type. Allowed: {', '.join(allowed)}"},
            status_code=400,
        )

    job_id = str(uuid.uuid4())[:8]
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    job = Job(job_id=job_id, filename=file.filename, input_path=input_path)
    jobs[job_id] = job

    thread = threading.Thread(target=process_video_job, args=(job,), daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "queued", "filename": file.filename}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    response = {
        "job_id": job.id,
        "status": job.status.value,
        "filename": job.filename,
        "progress": job.progress,
        "total_frames": job.total_frames,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }
    if job.status == JobStatus.ERROR:
        response["error"] = job.error
    if job.status == JobStatus.DONE and job.results:
        response["aggregate"] = job.results.get("aggregate")
        response["processing_time_seconds"] = job.results.get("processing_time_seconds")
        response["has_video"] = job.video_path is not None

    return response


@app.get("/api/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if job.status != JobStatus.DONE:
        return JSONResponse({"error": "Job not complete"}, status_code=400)
    return job.results


@app.get("/api/jobs/{job_id}/video")
async def get_job_video(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if job.status != JobStatus.DONE or not job.video_path:
        return JSONResponse({"error": "Video not available"}, status_code=400)
    return FileResponse(
        job.video_path,
        media_type="video/mp4",
        filename=f"neural_nexus_{job_id}_annotated.mp4",
    )


@app.websocket("/ws/camera")
async def camera_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket: Camera stream connected")

    if not pipeline.is_ready:
        await websocket.send_json({"error": "Models not loaded"})
        await websocket.close()
        return

    tracker = pipeline.create_tracker()
    smoother = pipeline.create_smoother()
    behavior_engine = pipeline.create_behavior_engine()
    visualizer = pipeline.visualizer

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await websocket.send_json({"error": "Cannot open camera"})
        await websocket.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    event_engine = pipeline.create_event_engine(fps=1.0)
    metrics_computer = pipeline.create_metrics_computer(fps=1.0)

    frame_count = 0
    running = True

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            frame_count += 1

            sample_rate = max(1, int(fps))
            if frame_count % sample_rate != 0:
                continue

            timestamp = frame_count / fps
            t0 = time.time()

            det_result = pipeline.detector.detect(frame)
            tracked = tracker.update(
                det_result["person_detections"],
                det_result["activity_detections"],
                frame=frame,
            )
            smoothed = smoother.update(tracked)

            teacher_present = any(
                "teacher" in e.get("confirmed_activities", {})
                for e in smoothed
            )
            teacher_engaging = any(
                any(a in e.get("confirmed_activities", {})
                    for a in ["guide", "answer", "On-stage interaction"])
                for e in smoothed
            )

            events = event_engine.extract_events(
                smoothed, timestamp=timestamp,
                teacher_present=teacher_present,
                teacher_engaging=teacher_engaging,
            )
            event_summary = event_engine.get_event_summary(events)
            behavior = behavior_engine.infer(smoothed, events, event_summary)
            metrics = metrics_computer.update(smoothed, behavior)

            inference_ms = (time.time() - t0) * 1000

            vis_frame = visualizer.annotate_frame(
                frame,
                smoothed_entities=smoothed,
                behavior_result=behavior,
                metrics=metrics,
                events=events,
                raw_detections=det_result["activity_detections"],
            )

            _, buffer = cv2.imencode(".jpg", vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            message = {
                "type": "frame",
                "frame": frame_b64,
                "metrics": {
                    "engagement_score": metrics["engagement_score"],
                    "participation_rate": metrics["participation_rate"],
                    "disruption_index": metrics["disruption_index"],
                    "teacher_interaction_ratio": metrics["teacher_interaction_ratio"],
                },
                "classroom_state": behavior["classroom_state"],
                "state_confidence": behavior["state_confidence"],
                "student_signals": behavior["student_signals"],
                "teacher_signals": behavior["teacher_signals"],
                "active_events": event_summary.get("active_events", []),
                "tracked_entities": len(tracked),
                "inference_ms": round(inference_ms, 1),
                "timestamp": round(timestamp, 2),
            }

            await websocket.send_json(message)

            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=0.01
                )
                if data == "stop":
                    running = False
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        logger.info("WebSocket: Camera stream disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        cap.release()
        logger.info("WebSocket: Camera released")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
