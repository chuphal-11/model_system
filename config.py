"""
Neural Nexus — Configuration & Constants
==========================================
Central configuration for the AI Classroom Intelligence System.
All tunable parameters, model paths, and rule definitions live here.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
# Each entry maps a model file to its known class names (extracted from the
# binary .pt files).  The order of class names matches the index the model
# outputs (0, 1, 2 …).
MODEL_REGISTRY = {
    "2_BlackBoard_Sreen_Teacher.pt": {
        "classes": ["screen", "blackBoard", "teacher"],
        "category": "environment",
        "confidence_threshold": 0.35,
    },
    "3_Discuss.pt": {
        "classes": ["discuss"],
        "category": "student_activity",
        "confidence_threshold": 0.40,
    },
    "4_Handrise_Read_write.pt": {
        "classes": ["hand-raising", "read", "write"],
        "category": "student_activity",
        "confidence_threshold": 0.35,
    },
    "5_Stand.pt": {
        "classes": ["stand"],
        "category": "student_activity",
        "confidence_threshold": 0.40,
    },
    "6_Talk.pt": {
        "classes": ["talk"],
        "category": "student_activity",
        "confidence_threshold": 0.40,
    },
    "7_Talk_Teacher_Behavior.pt": {
        "classes": ["talk", "guide", "answer", "On-stage interaction", "blackboard-writing"],
        "category": "teacher_behavior",
        "confidence_threshold": 0.35,
    },
    "8_Teacher.pt": {
        "classes": ["teacher"],
        "category": "teacher_detection",
        "confidence_threshold": 0.40,
    },
    "9_Teacher_Behavior.pt": {
        "classes": ["guide", "answer", "On-stage interaction", "blackboard-writing"],
        "category": "teacher_behavior",
        "confidence_threshold": 0.35,
    },
}

# Person detector (COCO pre-trained YOLOv8n) — used for counting & tracking
PERSON_DETECTOR = "yolov8n.pt"
PERSON_CLASS_ID = 0  # 'person' in COCO
PERSON_CONFIDENCE_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# Frame Extraction
# ---------------------------------------------------------------------------
# Process every Nth frame (1 = every frame, 2 = every other frame, etc.)
FRAME_SAMPLE_RATE = 1
# Maximum frames to process (None = all)
MAX_FRAMES = None

# ---------------------------------------------------------------------------
# Tracking (SORT)
# ---------------------------------------------------------------------------
TRACKER_MAX_AGE = 30          # Frames before a lost track is deleted
TRACKER_MIN_HITS = 3          # Minimum detections before track is confirmed
TRACKER_IOU_THRESHOLD = 0.3   # IoU threshold for associating detections

# ---------------------------------------------------------------------------
# Temporal Smoothing
# ---------------------------------------------------------------------------
SMOOTHING_WINDOW_SIZE = 15    # Number of frames in the sliding window
SMOOTHING_THRESHOLD = 0.60    # Fraction of window where activity must appear
                               # to be confirmed (e.g., 9 out of 15 frames)

# ---------------------------------------------------------------------------
# Event Extraction Rules
# ---------------------------------------------------------------------------
# Duration thresholds (in seconds) before a detection becomes an event
EVENT_RULES = {
    "student_wants_to_answer": {
        "required_activity": "hand-raising",
        "min_duration_sec": 3.0,
        "description": "Continuous hand raise indicates student wants to answer",
    },
    "side_conversation": {
        "required_activity": "talk",
        "condition": "no_teacher_engagement",
        "min_duration_sec": 5.0,
        "description": "Talking without teacher engagement",
    },
    "potential_disruption": {
        "required_activity": "stand",
        "min_duration_sec": 5.0,
        "description": "Sustained standing without clear purpose",
    },
    "active_learning": {
        "required_activity": ["read", "write"],
        "min_duration_sec": 3.0,
        "description": "Student is actively reading or writing",
    },
    "group_discussion": {
        "required_activity": "discuss",
        "min_duration_sec": 5.0,
        "description": "Multiple students engaged in discussion",
    },
    "lecture_in_progress": {
        "required_activity": ["blackBoard", "blackboard-writing"],
        "condition": "teacher_present",
        "min_duration_sec": 5.0,
        "description": "Teacher using blackboard — lecture mode",
    },
    "teacher_student_interaction": {
        "required_activity": ["guide", "answer"],
        "min_duration_sec": 3.0,
        "description": "Teacher actively guiding or answering students",
    },
}

# ---------------------------------------------------------------------------
# Behavior Inference Rules
# ---------------------------------------------------------------------------
BEHAVIOR_RULES = {
    "interactive": {
        "conditions": {
            "hand_raise_rate": (">", 0.3),
            "teacher_engaging": True,
        },
        "priority": 1,
    },
    "collaborative": {
        "conditions": {
            "discussion_rate": (">", 0.3),
        },
        "priority": 2,
    },
    "focused_learning": {
        "conditions": {
            "read_write_rate": (">", 0.4),
            "disruption_index": ("<", 0.2),
        },
        "priority": 3,
    },
    "lecture_mode": {
        "conditions": {
            "teacher_at_board": True,
            "student_activity_rate": ("<", 0.3),
        },
        "priority": 4,
    },
    "distracted": {
        "conditions": {
            "talk_rate": (">", 0.4),
            "participation_rate": ("<", 0.3),
        },
        "priority": 5,
    },
    "uncontrolled": {
        "conditions": {
            "teacher_present": False,
            "disruption_index": (">", 0.5),
        },
        "priority": 6,
    },
}

# ---------------------------------------------------------------------------
# Metrics Weights
# ---------------------------------------------------------------------------
ENGAGEMENT_WEIGHTS = {
    "hand-raising": 0.35,
    "write": 0.30,
    "read": 0.20,
    "discuss": 0.15,
}

DISRUPTION_WEIGHTS = {
    "stand": 0.35,
    "talk": 0.40,  # side conversation only
    "no_activity": 0.25,
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
# How often (in seconds) to emit a metrics snapshot in the timeline
METRICS_INTERVAL_SEC = 5.0

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
VIS_BBOX_COLORS = {
    "hand-raising": (0, 255, 0),      # Green
    "read": (0, 200, 200),            # Cyan
    "write": (0, 200, 200),           # Cyan
    "talk": (0, 165, 255),            # Orange
    "stand": (0, 0, 255),             # Red
    "discuss": (255, 200, 0),         # Gold
    "teacher": (255, 0, 128),         # Pink
    "blackBoard": (128, 0, 255),      # Purple
    "screen": (200, 200, 0),          # Yellow
    "guide": (255, 100, 100),         # Light red
    "answer": (100, 255, 100),        # Light green
    "On-stage interaction": (255, 0, 255),  # Magenta
    "blackboard-writing": (128, 0, 255),    # Purple
    "person": (200, 200, 200),        # Gray
}
