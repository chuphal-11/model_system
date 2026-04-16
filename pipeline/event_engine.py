"""
Neural Nexus — Event Extraction Engine
=========================================
Converts smoothed, temporally-stable detections into meaningful semantic
events using rule-based logic with temporal conditions.

Detection → human-understandable event
"""

import logging
from collections import defaultdict

import config

logger = logging.getLogger(__name__)


class Event:
    """A semantic event extracted from smoothed detections."""

    __slots__ = ("event_type", "entity_id", "confidence", "duration_frames",
                 "description", "timestamp")

    def __init__(self, event_type, entity_id, confidence, duration_frames,
                 description="", timestamp=0.0):
        self.event_type = event_type
        self.entity_id = entity_id
        self.confidence = confidence
        self.duration_frames = duration_frames
        self.description = description
        self.timestamp = timestamp

    def __repr__(self):
        return (f"Event({self.event_type}, entity={self.entity_id}, "
                f"conf={self.confidence:.2f}, dur={self.duration_frames}f)")

    def to_dict(self):
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "confidence": self.confidence,
            "duration_frames": self.duration_frames,
            "description": self.description,
            "timestamp": self.timestamp,
        }


class EventEngine:
    """
    Extracts semantic events from smoothed entity data using configurable
    rules. Each rule defines:
      - required activity / activities
      - optional conditions (teacher present, etc.)
      - minimum duration threshold
    """

    def __init__(self, fps=30.0, rules=None):
        """
        Args:
            fps: Frames per second (for converting duration thresholds)
            rules: Event rule config (defaults to config.EVENT_RULES)
        """
        self.fps = fps
        self.rules = rules or config.EVENT_RULES

        self._active_events = defaultdict(dict)

    def extract_events(self, smoothed_entities, timestamp=0.0,
                       teacher_present=False, teacher_engaging=False):
        """
        Extract events from smoothed entity data.

        Args:
            smoothed_entities: list of SmoothedEntity dicts from TemporalSmoother
            timestamp: Current timestamp in seconds
            teacher_present: Whether a teacher is detected in the scene
            teacher_engaging: Whether the teacher is actively engaging students

        Returns:
            list of Event objects for the current frame
        """
        events = []

        for entity in smoothed_entities:
            eid = entity["id"]
            confirmed = entity.get("confirmed_activities", {})
            streaks = entity.get("activity_streaks", {})

            for event_type, rule in self.rules.items():
                event = self._evaluate_rule(
                    event_type=event_type,
                    rule=rule,
                    entity_id=eid,
                    confirmed_activities=confirmed,
                    activity_streaks=streaks,
                    timestamp=timestamp,
                    teacher_present=teacher_present,
                    teacher_engaging=teacher_engaging,
                )
                if event is not None:
                    events.append(event)

        return events

    def _evaluate_rule(self, event_type, rule, entity_id,
                       confirmed_activities, activity_streaks,
                       timestamp, teacher_present, teacher_engaging):
        """
        Evaluate a single event rule against an entity's state.

        Returns:
            Event object if the rule fires, else None.
        """
        required = rule["required_activity"]

        if isinstance(required, str):
            required = [required]

        matching_activity = None
        matching_confidence = 0.0
        for act in required:
            if act in confirmed_activities:
                matching_activity = act
                matching_confidence = confirmed_activities[act]
                break

        if matching_activity is None:
            if entity_id in self._active_events:
                self._active_events[entity_id].pop(event_type, None)
            return None

        condition = rule.get("condition")
        if condition == "no_teacher_engagement" and teacher_engaging:
            return None
        if condition == "teacher_present" and not teacher_present:
            return None

        min_duration_sec = rule.get("min_duration_sec", 0)
        min_frames = int(min_duration_sec * self.fps)
        streak = activity_streaks.get(matching_activity, 0)

        if streak < min_frames:
            return None

        return Event(
            event_type=event_type,
            entity_id=entity_id,
            confidence=matching_confidence,
            duration_frames=streak,
            description=rule.get("description", ""),
            timestamp=timestamp,
        )

    def get_event_summary(self, events):
        """
        Summarize a list of events into counts and active event types.

        Returns:
            dict: {
                "active_events": [event_type, ...],
                "event_counts": {event_type: count, ...},
                "entity_event_map": {entity_id: [event_type, ...], ...},
            }
        """
        event_counts = defaultdict(int)
        entity_map = defaultdict(list)
        active_types = set()

        for e in events:
            event_counts[e.event_type] += 1
            entity_map[e.entity_id].append(e.event_type)
            active_types.add(e.event_type)

        return {
            "active_events": sorted(active_types),
            "event_counts": dict(event_counts),
            "entity_event_map": dict(entity_map),
        }

    def reset(self):
        """Clear all active event tracking."""
        self._active_events.clear()
