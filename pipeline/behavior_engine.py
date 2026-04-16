"""
Neural Nexus — Behavior Inference Engine
==========================================
The core intelligence of the system. Combines multiple events across
multiple individuals to infer the overall classroom state.

Events → Classroom behavioral state
"""

import logging
from collections import defaultdict

import config

logger = logging.getLogger(__name__)


class BehaviorEngine:
    """
    Infers classroom-level behavioral state by combining individual
    student/teacher events and computing aggregate signals.

    Produces:
      - classroom_state: one of the defined states (interactive, distracted, etc.)
      - student_signals: participation, distraction, inactivity counts
      - teacher_signals: teaching, engagement, absence
    """

    STATES = [
        "interactive", "collaborative", "focused_learning",
        "lecture_mode", "distracted", "uncontrolled", "idle",
    ]

    def __init__(self, rules=None):
        """
        Args:
            rules: Behavior rule config (defaults to config.BEHAVIOR_RULES)
        """
        self.rules = rules or config.BEHAVIOR_RULES

    def infer(self, smoothed_entities, events, event_summary):
        """
        Infer the overall classroom behavioral state.

        Args:
            smoothed_entities: list of SmoothedEntity dicts
            events: list of Event objects for the current frame
            event_summary: dict from EventEngine.get_event_summary()

        Returns:
            dict: {
                "classroom_state": str,
                "state_confidence": float,
                "student_signals": {
                    "participating": int,
                    "distracted": int,
                    "inactive": int,
                    "total": int,
                },
                "teacher_signals": {
                    "present": bool,
                    "engaging": bool,
                    "at_board": bool,
                    "teaching_mode": str,
                },
                "signals": {
                    "hand_raise_rate": float,
                    "read_write_rate": float,
                    "talk_rate": float,
                    "discussion_rate": float,
                    ...
                }
            }
        """
        signals = self._compute_signals(smoothed_entities, events,
                                        event_summary)

        student_signals = self._categorize_students(smoothed_entities)

        teacher_signals = self._analyze_teacher(smoothed_entities)

        signals["teacher_present"] = teacher_signals["present"]
        signals["teacher_engaging"] = teacher_signals["engaging"]
        signals["teacher_at_board"] = teacher_signals["at_board"]

        classroom_state, state_confidence = self._determine_state(signals)

        return {
            "classroom_state": classroom_state,
            "state_confidence": round(state_confidence, 3),
            "student_signals": student_signals,
            "teacher_signals": teacher_signals,
            "signals": {k: round(v, 3) if isinstance(v, float) else v
                        for k, v in signals.items()},
        }

    def _compute_signals(self, smoothed_entities, events, event_summary):
        """Compute raw signal values from entity data and events."""
        total = len(smoothed_entities)
        if total == 0:
            return {
                "hand_raise_rate": 0.0,
                "read_write_rate": 0.0,
                "talk_rate": 0.0,
                "discussion_rate": 0.0,
                "stand_rate": 0.0,
                "participation_rate": 0.0,
                "disruption_index": 0.0,
                "student_activity_rate": 0.0,
            }

        hand_raisers = 0
        readers_writers = 0
        talkers = 0
        discussers = 0
        standers = 0
        active = 0
        inactive = 0

        for entity in smoothed_entities:
            confirmed = entity.get("confirmed_activities", {})

            if "hand-raising" in confirmed:
                hand_raisers += 1
                active += 1
            if "read" in confirmed or "write" in confirmed:
                readers_writers += 1
                active += 1
            if "talk" in confirmed:
                talkers += 1
            if "discuss" in confirmed:
                discussers += 1
                active += 1
            if "stand" in confirmed:
                standers += 1

            if confirmed and any(
                a in confirmed for a in
                ["hand-raising", "read", "write", "discuss"]
            ):
                pass
            elif not confirmed:
                inactive += 1

        return {
            "hand_raise_rate": hand_raisers / total,
            "read_write_rate": readers_writers / total,
            "talk_rate": talkers / total,
            "discussion_rate": discussers / total,
            "stand_rate": standers / total,
            "participation_rate": active / total,
            "disruption_index": (talkers + standers) / total,
            "student_activity_rate": active / total,
        }

    def _categorize_students(self, smoothed_entities):
        """Categorize each student as participating, distracted, or inactive."""
        participating = 0
        distracted = 0
        inactive = 0
        total = 0

        for entity in smoothed_entities:
            confirmed = entity.get("confirmed_activities", {})

            if "teacher" in confirmed:
                continue

            total += 1

            productive = {"hand-raising", "read", "write", "discuss"}
            disruptive = {"talk", "stand"}

            has_productive = bool(productive & set(confirmed.keys()))
            has_disruptive = bool(disruptive & set(confirmed.keys()))

            if has_productive:
                participating += 1
            elif has_disruptive:
                distracted += 1
            else:
                inactive += 1

        return {
            "participating": participating,
            "distracted": distracted,
            "inactive": inactive,
            "total": total,
        }

    def _analyze_teacher(self, smoothed_entities):
        """Analyze teacher presence and behavior."""
        teacher_present = False
        teacher_engaging = False
        teacher_at_board = False
        teaching_mode = "absent"

        for entity in smoothed_entities:
            confirmed = entity.get("confirmed_activities", {})

            if "teacher" in confirmed:
                teacher_present = True

            if "guide" in confirmed or "answer" in confirmed:
                teacher_engaging = True
                teaching_mode = "interactive"

            if ("blackBoard" in confirmed
                    or "blackboard-writing" in confirmed):
                teacher_at_board = True
                if not teacher_engaging:
                    teaching_mode = "lecturing"

            if "On-stage interaction" in confirmed:
                teacher_engaging = True
                teaching_mode = "interactive"

        if teacher_present and teaching_mode == "absent":
            teaching_mode = "present"

        return {
            "present": teacher_present,
            "engaging": teacher_engaging,
            "at_board": teacher_at_board,
            "teaching_mode": teaching_mode,
        }

    def _determine_state(self, signals):
        """
        Determine the classroom state by evaluating rules in priority order.

        Returns:
            (state_name, confidence)
        """
        sorted_rules = sorted(self.rules.items(),
                              key=lambda x: x[1].get("priority", 99))

        for state_name, rule in sorted_rules:
            match, confidence = self._evaluate_behavior_rule(
                rule["conditions"], signals
            )
            if match:
                return state_name, confidence

        return "idle", 0.5

    def _evaluate_behavior_rule(self, conditions, signals):
        """
        Evaluate a single behavior rule.

        Returns:
            (matched: bool, confidence: float)
        """
        confidence_sum = 0.0
        condition_count = 0
        all_match = True

        for signal_name, expected in conditions.items():
            actual = signals.get(signal_name, 0.0)

            if isinstance(expected, bool):
                if bool(actual) != expected:
                    all_match = False
                    break
                confidence_sum += 1.0
                condition_count += 1

            elif isinstance(expected, tuple) and len(expected) == 2:
                op, threshold = expected

                if op == ">" and actual > threshold:
                    confidence_sum += min(actual / max(threshold, 0.01), 2.0)
                    condition_count += 1
                elif op == "<" and actual < threshold:
                    confidence_sum += min(
                        (threshold - actual) / max(threshold, 0.01), 2.0
                    )
                    condition_count += 1
                elif op == ">=" and actual >= threshold:
                    confidence_sum += min(actual / max(threshold, 0.01), 2.0)
                    condition_count += 1
                elif op == "<=" and actual <= threshold:
                    confidence_sum += min(
                        (threshold - actual + 0.1) / max(threshold, 0.01), 2.0
                    )
                    condition_count += 1
                else:
                    all_match = False
                    break

        if all_match and condition_count > 0:
            avg_confidence = confidence_sum / condition_count
            return True, min(avg_confidence, 1.0)

        return False, 0.0
