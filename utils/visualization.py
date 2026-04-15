"""
Neural Nexus — Visualization Utility
=======================================
Draws bounding boxes, track IDs, activity labels, and metrics overlay
on frames for debugging and demonstration purposes.
"""

import cv2
import numpy as np
import logging

import config

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Annotates frames with detection boxes, track IDs, and metrics overlay.
    """

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 1
        self.line_type = cv2.LINE_AA

    def annotate_frame(self, frame, tracked_entities=None,
                       smoothed_entities=None, behavior_result=None,
                       metrics=None, events=None, raw_detections=None):
        """
        Draw annotations on a frame.

        Args:
            frame: numpy BGR image (will be copied, not modified in place)
            tracked_entities: raw tracker output
            smoothed_entities: temporal smoother output
            behavior_result: behavior engine output
            metrics: metrics computer output
            events: list of Event objects
            raw_detections: pure output from detect() bounding boxes

        Returns:
            annotated frame (numpy BGR image)
        """
        vis = frame.copy()

        # Draw raw detection boxes to match user reference image
        if raw_detections:
            for det in raw_detections:
                self._draw_raw_detection(vis, det)
        else:
            # Fallback to drawing entities if no raw detections
            entities = smoothed_entities or tracked_entities or []
            for entity in entities:
                self._draw_entity(vis, entity)

        # Draw metrics overlay
        if metrics or behavior_result:
            self._draw_overlay(vis, metrics, behavior_result, events)

        return vis

    def _draw_raw_detection(self, frame, det):
        """Draw bounding box and label for a raw detection."""
        bbox = getattr(det, "bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(c) for c in bbox]
        class_name = getattr(det, "class_name", "unknown")

        # Color based on class
        color = config.VIS_BBOX_COLORS.get(class_name, (0, 255, 255))

        # Thicker bounding box (3px)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Draw bold label directly above box
        label_y = max(y1 - 10, 25)
        # Using larger scale and thicker font
        cv2.putText(frame, class_name, (x1, label_y),
                    self.font, 1.0, color, 2, self.line_type)

    def _draw_entity(self, frame, entity):
        """Draw bounding box and labels for a single entity."""
        bbox = entity.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(c) for c in bbox]
        eid = entity.get("id", -1)
        confirmed = entity.get("confirmed_activities", {})
        raw = entity.get("raw_activities", entity.get("activities", {}))

        # Determine color based on the primary activity
        if confirmed:
            primary_activity = max(confirmed, key=confirmed.get)
            color = config.VIS_BBOX_COLORS.get(primary_activity, (200, 200, 200))
        else:
            color = (128, 128, 128)  # Gray for no confirmed activity

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw ID label
        id_label = f"ID:{eid}"
        label_y = max(y1 - 8, 15)
        cv2.putText(frame, id_label, (x1, label_y),
                    self.font, self.font_scale, color,
                    self.thickness, self.line_type)

        # Draw confirmed activities
        if confirmed:
            for i, (act, conf) in enumerate(confirmed.items()):
                act_label = f"{act} ({conf:.2f})"
                y_offset = y2 + 15 + i * 18
                cv2.putText(frame, act_label, (x1, y_offset),
                            self.font, 0.4,
                            config.VIS_BBOX_COLORS.get(act, color),
                            1, self.line_type)

    def _draw_overlay(self, frame, metrics, behavior_result, events):
        """Draw a semi-transparent overlay with metrics and state info."""
        h, w = frame.shape[:2]

        # Background panel
        overlay = frame.copy()
        panel_w = 320
        panel_h = 200
        cv2.rectangle(overlay, (w - panel_w - 10, 10),
                      (w - 10, panel_h + 10),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text lines
        x_start = w - panel_w
        y_start = 30
        line_height = 22

        lines = []

        if behavior_result:
            state = behavior_result.get("classroom_state", "unknown")
            conf = behavior_result.get("state_confidence", 0)
            lines.append(
                (f"State: {state.upper()} ({conf:.0%})", (0, 255, 255))
            )

            student = behavior_result.get("student_signals", {})
            lines.append(
                (f"Students: {student.get('participating', 0)}P "
                 f"{student.get('distracted', 0)}D "
                 f"{student.get('inactive', 0)}I "
                 f"/ {student.get('total', 0)}",
                 (200, 200, 200))
            )

            teacher = behavior_result.get("teacher_signals", {})
            t_mode = teacher.get("teaching_mode", "absent")
            lines.append(
                (f"Teacher: {t_mode}", (255, 150, 150))
            )

        if metrics:
            lines.append(("", (0, 0, 0)))  # spacer
            lines.append(
                (f"Engagement:    {metrics.get('engagement_score', 0):.1%}",
                 (0, 255, 0))
            )
            lines.append(
                (f"Participation: {metrics.get('participation_rate', 0):.1%}",
                 (0, 200, 255))
            )
            lines.append(
                (f"Disruption:    {metrics.get('disruption_index', 0):.1%}",
                 (0, 0, 255))
            )
            lines.append(
                (f"Teacher Int:   "
                 f"{metrics.get('teacher_interaction_ratio', 0):.1%}",
                 (255, 200, 0))
            )

        for i, (text, color) in enumerate(lines):
            if text:
                cv2.putText(frame, text,
                            (x_start, y_start + i * line_height),
                            self.font, 0.45, color,
                            1, self.line_type)

    def save_frame(self, frame, path):
        """Save an annotated frame to disk."""
        cv2.imwrite(path, frame)

    @staticmethod
    def create_video_writer(output_path, fps, width, height):
        """Create a VideoWriter for saving annotated video."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
