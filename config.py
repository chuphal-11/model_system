"""
Neural Nexus — Configuration & Constants
==========================================
Central configuration for the AI Classroom Intelligence System.
All tunable parameters, model paths, and rule definitions live here.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

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

PERSON_DETECTOR = "yolov8n.pt"
PERSON_CLASS_ID = 0
PERSON_CONFIDENCE_THRESHOLD = 0.30

FRAME_SAMPLE_RATE = 1
MAX_FRAMES = None

TRACKER_MAX_AGE = 30
TRACKER_MIN_HITS = 3
TRACKER_IOU_THRESHOLD = 0.3

SMOOTHING_WINDOW_SIZE = 15
SMOOTHING_THRESHOLD = 0.60

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

ENGAGEMENT_WEIGHTS = {
    "hand-raising": 0.35,
    "write": 0.30,
    "read": 0.20,
    "discuss": 0.15,
}

DISRUPTION_WEIGHTS = {
    "stand": 0.35,
    "talk": 0.40,
    "no_activity": 0.25,
}

METRICS_INTERVAL_SEC = 5.0

VIS_BBOX_COLORS = {
    "hand-raising -": (0, 255, 0),
    "read": (0, 200, 200),
    "write": (0, 200, 200),
    "talk": (0, 165, 255),
    "stand": (0, 0, 255),
    "discuss": (255, 200, 0),
    "teacher": (255, 0, 128),
    "blackBoard": (128, 0, 255),
    "screen": (200, 200, 0),
    "guide": (255, 100, 100),
    "answer": (100, 255, 100),
    "On-stage interaction": (255, 0, 255),
    "blackboard-writing": (128, 0, 255),
    "person": (200, 200, 200),
}
