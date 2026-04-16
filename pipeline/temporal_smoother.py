"""
Neural Nexus — Temporal Smoother
==================================
Maintains a sliding window buffer per tracked entity to stabilize
activity detections. Removes single-frame noise by requiring an activity
to appear in a sufficient fraction of the window before confirming it.
"""

from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class TemporalSmoother:
    """
    Sliding-window temporal smoother for tracked entity activities.

    For each tracked entity, maintains a buffer of the last N frames
    of activity detections. An activity is considered "confirmed" only
    if it appears in at least (threshold * window_size) of the buffered
    frames.
    """

    def __init__(self, window_size=15, threshold=0.60):
        """
        Args:
            window_size: Number of frames in the sliding window
            threshold: Fraction of window frames where activity must appear
                       for it to be confirmed (e.g., 0.6 = 9/15)
        """
        self.window_size = window_size
        self.threshold = threshold
        self.min_count = int(window_size * threshold)

        self._buffers = defaultdict(lambda: deque(maxlen=window_size))

        self._activity_streaks = defaultdict(lambda: defaultdict(int))

    def update(self, tracked_entities):
        """
        Add new frame's tracked entities to the buffer and compute
        smoothed activities.

        Args:
            tracked_entities: list of entity dicts from the tracker:
                {
                    "id": int,
                    "bbox": [x1, y1, x2, y2],
                    "activities": {class_name: confidence, ...},
                    "confirmed": bool,
                }

        Returns:
            list of SmoothedEntity dicts:
                {
                    "id": int,
                    "bbox": [x1, y1, x2, y2],
                    "confirmed_activities": {class_name: avg_confidence, ...},
                    "raw_activities": {class_name: confidence, ...},
                    "stability_scores": {class_name: fraction, ...},
                    "activity_streaks": {class_name: consecutive_frames, ...},
                }
        """
        active_ids = set()
        results = []

        for entity in tracked_entities:
            eid = entity["id"]
            active_ids.add(eid)
            activities = entity.get("activities", {})

            self._buffers[eid].append(activities)

            for act_name in list(self._activity_streaks[eid].keys()):
                if act_name not in activities:
                    self._activity_streaks[eid][act_name] = 0
            for act_name in activities:
                self._activity_streaks[eid][act_name] = (
                    self._activity_streaks[eid].get(act_name, 0) + 1
                )

            confirmed, stability, avg_conf = self._compute_smoothed(eid)

            results.append({
                "id": eid,
                "bbox": entity["bbox"],
                "confirmed_activities": {
                    k: avg_conf.get(k, 0.0) for k in confirmed
                },
                "raw_activities": activities,
                "stability_scores": stability,
                "activity_streaks": dict(self._activity_streaks[eid]),
                "confirmed_track": entity.get("confirmed", False),
            })

        stale_ids = set(self._buffers.keys()) - active_ids
        for sid in stale_ids:
            if len(self._buffers[sid]) == 0:
                del self._buffers[sid]
                if sid in self._activity_streaks:
                    del self._activity_streaks[sid]

        return results

    def _compute_smoothed(self, entity_id):
        """
        Compute smoothed activities from the buffer.

        Returns:
            confirmed: set of confirmed activity names
            stability: dict {activity_name: fraction_of_window}
            avg_conf: dict {activity_name: average_confidence}
        """
        buffer = self._buffers[entity_id]
        if not buffer:
            return set(), {}, {}

        activity_counts = defaultdict(int)
        activity_conf_sum = defaultdict(float)

        for frame_activities in buffer:
            for act_name, conf in frame_activities.items():
                activity_counts[act_name] += 1
                activity_conf_sum[act_name] += conf

        num_frames = len(buffer)
        confirmed = set()
        stability = {}
        avg_conf = {}

        for act_name, count in activity_counts.items():
            fraction = count / num_frames
            stability[act_name] = round(fraction, 3)
            avg_conf[act_name] = round(activity_conf_sum[act_name] / count, 3)

            if count >= self.min_count:
                confirmed.add(act_name)

        return confirmed, stability, avg_conf

    def get_entity_history(self, entity_id):
        """Return the full buffer for a specific entity (for debugging)."""
        return list(self._buffers.get(entity_id, []))

    def reset(self):
        """Clear all buffers."""
        self._buffers.clear()
        self._activity_streaks.clear()
