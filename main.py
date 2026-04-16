#!/usr/bin/env python3
"""
Neural Nexus — AI Classroom Intelligence System
==================================================
Main entry point. Orchestrates the full pipeline:

  Video → Frames → Detection → Tracking → Smoothing
       → Events → Behavior → Metrics → JSON Output

Usage:
    python main.py --input video.mp4 --output output/results.json
    python main.py --input 0 --output output/results.json  (webcam)
    python main.py --input video.mp4 --visualize            (with annotated output)
"""

import argparse
import json
import logging
import os
import sys
import time

import cv2 as _cv2
_original_imshow = _cv2.imshow

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from pipeline.frame_extractor import FrameExtractor
from pipeline.detector import MultiModelDetector
from pipeline.tracker import SORTTracker
from pipeline.temporal_smoother import TemporalSmoother
from pipeline.event_engine import EventEngine
from pipeline.behavior_engine import BehaviorEngine
from pipeline.metrics import MetricsComputer
from utils.visualization import Visualizer


def setup_logging(verbose=False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-7s │ %(name)-20s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_pipeline(args):
    """Execute the full pipeline."""
    logger = logging.getLogger("neural_nexus")

    output_dir = os.path.dirname(args.output) or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    import torch
    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Initializing pipeline components …")

    detector = MultiModelDetector(device=device)
    detector.load_models()

    if detector.num_models_loaded == 0:
        logger.error("No models loaded! Cannot proceed.")
        sys.exit(1)

    tracker = SORTTracker(
        max_age=config.TRACKER_MAX_AGE,
        min_hits=config.TRACKER_MIN_HITS,
        iou_threshold=config.TRACKER_IOU_THRESHOLD,
    )

    smoother = TemporalSmoother(
        window_size=config.SMOOTHING_WINDOW_SIZE,
        threshold=config.SMOOTHING_THRESHOLD,
    )

    event_engine = None
    behavior_engine = BehaviorEngine()
    metrics_computer = None
    visualizer = Visualizer() if args.visualize else None
    video_writer = None

    extractor = FrameExtractor(
        source=args.input,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
    )

    with extractor:
        fps = extractor.fps or 30.0
        
        if getattr(extractor, "is_live", False):
            args.sample_rate = max(1, int(fps))
            logger.info(f"Live camera detected: Processing rate locked to exactly 1 frame per second "
                        f"(sampled every {args.sample_rate} original frames).")

        event_engine = EventEngine(fps=fps / args.sample_rate)
        metrics_computer = MetricsComputer(
            fps=fps / args.sample_rate,
            window_seconds=10.0,
        )

        if visualizer and extractor.width > 0:
            vis_path = os.path.splitext(args.output)[0] + "_annotated.mp4"
            video_writer = Visualizer.create_video_writer(
                vis_path, fps, extractor.width, extractor.height
            )
            logger.info(f"Annotated video will be saved to: {vis_path}")

        timeline = []
        frame_by_frame = []
        metrics_interval_frames = int(
            config.METRICS_INTERVAL_SEC * fps / args.sample_rate
        )
        metrics_interval_frames = max(metrics_interval_frames, 1)

        frame_count = 0
        total_inference_ms = 0.0
        t_start = time.time()

        logger.info("=" * 60)
        logger.info("Starting pipeline processing …")
        logger.info("=" * 60)

        for frame_num, timestamp, frame in extractor:
            frame_count += 1

            det_result = detector.detect(frame)
            total_inference_ms += det_result["inference_time_ms"]

            person_dets = det_result["person_detections"]
            activity_dets = det_result["activity_detections"]

            tracked_entities = tracker.update(person_dets, activity_dets, frame=frame)

            smoothed_entities = smoother.update(tracked_entities)

            teacher_present = any(
                "teacher" in e.get("confirmed_activities", {})
                for e in smoothed_entities
            )
            teacher_engaging = any(
                any(a in e.get("confirmed_activities", {})
                    for a in ["guide", "answer", "On-stage interaction"])
                for e in smoothed_entities
            )

            events = event_engine.extract_events(
                smoothed_entities,
                timestamp=timestamp,
                teacher_present=teacher_present,
                teacher_engaging=teacher_engaging,
            )
            event_summary = event_engine.get_event_summary(events)

            behavior_result = behavior_engine.infer(
                smoothed_entities, events, event_summary
            )

            metrics = metrics_computer.update(smoothed_entities,
                                              behavior_result)

            frame_by_frame.append({
                "frame": frame_num,
                "timestamp": round(timestamp, 2),
                "detections": [
                    {
                        "class_name": getattr(d, "class_name", ""),
                        "confidence": round(getattr(d, "confidence", 0.0), 3),
                        "bbox": getattr(d, "bbox", [])
                    } for d in activity_dets
                ]
            })

            if frame_count % metrics_interval_frames == 0:
                snapshot = {
                    "timestamp": format_timestamp(timestamp),
                    "frame": frame_num,
                    "seconds": round(timestamp, 2),
                    "classroom_state": behavior_result["classroom_state"],
                    "state_confidence": behavior_result["state_confidence"],
                    "engagement_score": metrics["engagement_score"],
                    "participation_rate": metrics["participation_rate"],
                    "disruption_index": metrics["disruption_index"],
                    "teacher_interaction_ratio": metrics[
                        "teacher_interaction_ratio"
                    ],
                    "active_events": event_summary.get("active_events", []),
                    "tracked_entities": len(tracked_entities),
                    "student_summary": behavior_result["student_signals"],
                    "teacher_signals": behavior_result["teacher_signals"],
                }
                timeline.append(snapshot)

            if visualizer:
                vis_frame = visualizer.annotate_frame(
                    frame,
                    smoothed_entities=smoothed_entities,
                    behavior_result=behavior_result,
                    metrics=metrics,
                    events=events,
                    raw_detections=activity_dets,
                )
                if video_writer:
                    video_writer.write(vis_frame)
                
                _original_imshow("Neural Nexus Live View", vis_frame)
                if _cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Live View closed by user 'q' key. Stopping processing....")
                    break

            if frame_count % 50 == 0:
                elapsed = time.time() - t_start
                fps_actual = frame_count / max(elapsed, 0.001)
                det_count = len(det_result["all_detections"])
                logger.info(
                    f"Frame {frame_num:>6d} │ "
                    f"{format_timestamp(timestamp)} │ "
                    f"Det: {det_count:>3d} │ "
                    f"Tracked: {len(tracked_entities):>2d} │ "
                    f"State: {behavior_result['classroom_state']:<18s} │ "
                    f"Eng: {metrics['engagement_score']:.2f} │ "
                    f"{fps_actual:.1f} FPS"
                )

    if video_writer:
        video_writer.release()
        logger.info(f"Saved annotated video: {vis_path}")

    elapsed = time.time() - t_start
    aggregate = metrics_computer.get_aggregate_metrics()

    output = {
        "video": os.path.basename(args.input) if not args.input.isdigit() else f"webcam_{args.input}",
        "total_frames_processed": frame_count,
        "fps": round(fps, 1),
        "sample_rate": args.sample_rate,
        "effective_fps": round(fps / args.sample_rate, 1),
        "duration_seconds": round(extractor.duration_seconds, 2),
        "processing_time_seconds": round(elapsed, 2),
        "avg_inference_ms": round(total_inference_ms / max(frame_count, 1), 1),
        "device": device,
        "models_loaded": detector.num_models_loaded,
        "timeline": timeline,
        "frame_by_frame": frame_by_frame,
        "aggregate": aggregate,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Pipeline complete!")
    logger.info(f"  Processed: {frame_count} frames in {elapsed:.1f}s")
    logger.info(f"  Avg inference: {output['avg_inference_ms']:.1f}ms/frame")
    logger.info(f"  Dominant state: {aggregate['dominant_state']}")
    logger.info(f"  Avg engagement: {aggregate['avg_engagement']:.1%}")
    logger.info(f"  Avg participation: {aggregate['avg_participation']:.1%}")
    logger.info(f"  Avg disruption: {aggregate['avg_disruption']:.1%}")
    logger.info(f"  Output saved to: {args.output}")
    logger.info("=" * 60)

    print(f"\n{'─' * 50}")
    print(f"  NEURAL NEXUS — Results Summary")
    print(f"{'─' * 50}")
    print(f"  Classroom State:   {aggregate['dominant_state'].upper()}")
    print(f"  Engagement:        {aggregate['avg_engagement']:.1%}")
    print(f"  Participation:     {aggregate['avg_participation']:.1%}")
    print(f"  Disruption:        {aggregate['avg_disruption']:.1%}")
    print(f"  Teacher Interact:  {aggregate['avg_teacher_interaction']:.1%}")
    print(f"{'─' * 50}")
    print(f"  State Distribution:")
    for state, frac in sorted(aggregate.get("state_distribution", {}).items(),
                              key=lambda x: -x[1]):
        bar = "█" * int(frac * 30)
        print(f"    {state:<20s} {frac:>5.1%}  {bar}")
    print(f"{'─' * 50}")
    print(f"  Full results: {args.output}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Neural Nexus — AI Classroom Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a video file:
    python main.py --input classroom.mp4 --output output/results.json

  Use webcam:
    python main.py --input 0 --output output/live.json

  With visualization:
    python main.py --input classroom.mp4 --output output/results.json --visualize

  Fast mode (process every 3rd frame):
    python main.py --input classroom.mp4 --output output/results.json --sample-rate 3
        """,
    )

    parser.add_argument(
        "--input", "-i", required=True,
        help="Video file path or camera index (e.g., 0 for webcam)",
    )
    parser.add_argument(
        "--output", "-o", default="output/results.json",
        help="Output JSON file path (default: output/results.json)",
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Save annotated video with bounding boxes and metrics overlay",
    )
    parser.add_argument(
        "--sample-rate", "-s", type=int, default=config.FRAME_SAMPLE_RATE,
        help=f"Process every Nth frame (default: {config.FRAME_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--max-frames", "-m", type=int, default=config.MAX_FRAMES,
        help="Maximum number of frames to process (default: all)",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference even if CUDA is available",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    run_pipeline(args)


if __name__ == "__main__":
    main()
