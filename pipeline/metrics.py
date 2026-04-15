"""
Neural Nexus — Metrics Computation
=====================================
Quantifies classroom behaviors into actionable metrics computed over
configurable time windows.

Metrics:
  - Engagement Score (0–1)
  - Participation Rate (0–1)
  - Disruption Index (0–1)
  - Teacher Interaction Ratio (0–1)
"""

import logging
from collections import deque, defaultdict

import config

logger = logging.getLogger(__name__)


class MetricsComputer:
    """
    Computes quantified classroom metrics over sliding time windows.
    Accumulates per-frame behavior data and produces aggregate scores.
    """

    def __init__(self, fps=30.0, window_seconds=10.0):
        """
        Args:
            fps: Frames per second
            window_seconds: Time window for metrics aggregation
        """
        self.fps = fps
        self.window_size = int(fps * window_seconds)

        # Sliding window buffers
        self._engagement_buffer = deque(maxlen=self.window_size)
        self._participation_buffer = deque(maxlen=self.window_size)
        self._disruption_buffer = deque(maxlen=self.window_size)
        self._teacher_interaction_buffer = deque(maxlen=self.window_size)

        # Aggregation over the entire video
        self._total_frames = 0
        self._state_counts = defaultdict(int)
        self._cumulative_engagement = 0.0
        self._cumulative_participation = 0.0
        self._cumulative_disruption = 0.0
        self._cumulative_teacher_interaction = 0.0

    def update(self, smoothed_entities, behavior_result):
        """
        Update metrics with a new frame's data.

        Args:
            smoothed_entities: list of SmoothedEntity dicts
            behavior_result: dict from BehaviorEngine.infer()

        Returns:
            dict with current metrics snapshot
        """
        self._total_frames += 1

        # Track state distribution
        state = behavior_result.get("classroom_state", "idle")
        self._state_counts[state] += 1

        # --- Engagement Score ---
        engagement = self._compute_engagement(smoothed_entities)
        self._engagement_buffer.append(engagement)
        self._cumulative_engagement += engagement

        # --- Participation Rate ---
        student_signals = behavior_result.get("student_signals", {})
        total_students = max(student_signals.get("total", 0), 1)
        participating = student_signals.get("participating", 0)
        participation = participating / total_students
        self._participation_buffer.append(participation)
        self._cumulative_participation += participation

        # --- Disruption Index ---
        disruption = self._compute_disruption(smoothed_entities)
        self._disruption_buffer.append(disruption)
        self._cumulative_disruption += disruption

        # --- Teacher Interaction Ratio ---
        teacher_signals = behavior_result.get("teacher_signals", {})
        teacher_interaction = 1.0 if teacher_signals.get("engaging") else 0.0
        self._teacher_interaction_buffer.append(teacher_interaction)
        self._cumulative_teacher_interaction += teacher_interaction

        # Compute windowed metrics
        return self.get_current_metrics(behavior_result)

    def _compute_engagement(self, smoothed_entities):
        """
        Compute per-frame engagement score based on productive activities.

        Engagement = weighted sum of productive activity rates.
        """
        if not smoothed_entities:
            return 0.0

        total = len(smoothed_entities)
        weighted_score = 0.0
        max_possible = 0.0

        for activity, weight in config.ENGAGEMENT_WEIGHTS.items():
            count = sum(
                1 for e in smoothed_entities
                if activity in e.get("confirmed_activities", {})
            )
            weighted_score += (count / total) * weight
            max_possible += weight

        if max_possible > 0:
            return min(weighted_score / max_possible, 1.0)
        return 0.0

    def _compute_disruption(self, smoothed_entities):
        """
        Compute per-frame disruption index based on disruptive activities.
        """
        if not smoothed_entities:
            return 0.0

        total = len(smoothed_entities)
        weighted_score = 0.0
        max_possible = 0.0

        for activity, weight in config.DISRUPTION_WEIGHTS.items():
            if activity == "no_activity":
                count = sum(
                    1 for e in smoothed_entities
                    if not e.get("confirmed_activities", {})
                )
            else:
                count = sum(
                    1 for e in smoothed_entities
                    if activity in e.get("confirmed_activities", {})
                )
            weighted_score += (count / total) * weight
            max_possible += weight

        if max_possible > 0:
            return min(weighted_score / max_possible, 1.0)
        return 0.0

    def get_current_metrics(self, behavior_result=None):
        """
        Get the current windowed metrics snapshot.

        Returns:
            dict with metric values
        """
        def _avg(buf):
            return sum(buf) / len(buf) if buf else 0.0

        metrics = {
            "engagement_score": round(_avg(self._engagement_buffer), 3),
            "participation_rate": round(_avg(self._participation_buffer), 3),
            "disruption_index": round(_avg(self._disruption_buffer), 3),
            "teacher_interaction_ratio": round(
                _avg(self._teacher_interaction_buffer), 3
            ),
        }

        if behavior_result:
            metrics["classroom_state"] = behavior_result.get(
                "classroom_state", "idle"
            )
            metrics["state_confidence"] = behavior_result.get(
                "state_confidence", 0.0
            )

        return metrics

    def get_aggregate_metrics(self):
        """
        Get metrics aggregated over the entire video.

        Returns:
            dict with aggregate metric values
        """
        total = max(self._total_frames, 1)

        # State distribution
        state_dist = {}
        for state, count in self._state_counts.items():
            state_dist[state] = round(count / total, 3)

        # Find dominant state
        dominant_state = max(self._state_counts,
                            key=self._state_counts.get,
                            default="idle")

        return {
            "avg_engagement": round(self._cumulative_engagement / total, 3),
            "avg_participation": round(
                self._cumulative_participation / total, 3
            ),
            "avg_disruption": round(self._cumulative_disruption / total, 3),
            "avg_teacher_interaction": round(
                self._cumulative_teacher_interaction / total, 3
            ),
            "dominant_state": dominant_state,
            "state_distribution": state_dist,
            "total_frames_processed": self._total_frames,
        }

    def reset(self):
        """Reset all metrics."""
        self._engagement_buffer.clear()
        self._participation_buffer.clear()
        self._disruption_buffer.clear()
        self._teacher_interaction_buffer.clear()
        self._total_frames = 0
        self._state_counts.clear()
        self._cumulative_engagement = 0.0
        self._cumulative_participation = 0.0
        self._cumulative_disruption = 0.0
        self._cumulative_teacher_interaction = 0.0
