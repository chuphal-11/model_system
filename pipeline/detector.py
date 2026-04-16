"""
Neural Nexus — Multi-Model Detector
=====================================
Manages all 8 YOLO models + the person detector.
Runs each model on every frame and merges detections into a unified format.
"""

import os
import logging
import time

import config
from utils.model_loader import load_custom_model, load_person_detector

logger = logging.getLogger(__name__)


class Detection:
    """A single detection from one of the YOLO models."""

    __slots__ = ("bbox", "class_name", "class_id", "confidence",
                 "source_model")

    def __init__(self, bbox, class_name, class_id, confidence, source_model):
        self.bbox = bbox
        self.class_name = class_name
        self.class_id = class_id
        self.confidence = confidence
        self.source_model = source_model

    def __repr__(self):
        return (f"Detection({self.class_name}, conf={self.confidence:.2f}, "
                f"src={self.source_model})")

    def to_dict(self):
        return {
            "bbox": self.bbox,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "source_model": self.source_model,
        }


class MultiModelDetector:
    """
    Loads and manages all YOLO models.  Runs inference on each frame
    and returns a unified list of Detection objects.
    """

    def __init__(self, device="cpu"):
        """
        Args:
            device: 'cpu' or 'cuda:0'
        """
        self.device = device
        self.models = {}
        self.person_detector = None
        self._loaded = False

    def load_models(self):
        """Load all models from the registry."""
        logger.info("=" * 60)
        logger.info("Loading YOLO models …")
        logger.info("=" * 60)

        for model_name, meta in config.MODEL_REGISTRY.items():
            model_path = os.path.join(config.MODEL_DIR, model_name)
            if not os.path.exists(model_path):
                logger.warning(f"  ⚠ Model file not found: {model_name}")
                continue

            wrapper = load_custom_model(
                model_path=model_path,
                class_names=meta["classes"],
                device=self.device,
                conf_threshold=meta["confidence_threshold"],
            )
            if wrapper is not None:
                self.models[model_name] = {
                    "model": wrapper,
                    "meta": meta,
                }

        person_path = os.path.join(config.MODEL_DIR, config.PERSON_DETECTOR)
        if os.path.exists(person_path):
            self.person_detector = load_person_detector(
                person_path,
                device=self.device,
                conf_threshold=config.PERSON_CONFIDENCE_THRESHOLD,
            )

        self._loaded = True
        logger.info(f"Loaded {len(self.models)} custom models "
                     f"+ {'1 person detector' if self.person_detector else 'no person detector'}")
        logger.info("=" * 60)

    def detect(self, frame):
        """
        Run all models on a single frame.

        Args:
            frame: numpy BGR image

        Returns:
            dict with keys:
                - 'person_detections': list[Detection]
                - 'activity_detections': list[Detection]
                - 'all_detections': list[Detection]
                - 'inference_time_ms': float
        """
        if not self._loaded:
            self.load_models()

        all_detections = []
        person_detections = []
        activity_detections = []
        t0 = time.time()

        if self.person_detector is not None:
            try:
                raw = self.person_detector(frame)
                for d in raw:
                    det = Detection(
                        bbox=d["bbox"],
                        class_name="person",
                        class_id=0,
                        confidence=d["confidence"],
                        source_model="yolov8n",
                    )
                    person_detections.append(det)
                    all_detections.append(det)
            except Exception as e:
                logger.debug(f"Person detector error: {e}")

        for model_name, entry in self.models.items():
            try:
                raw = entry["model"](frame)
                for d in raw:
                    det = Detection(
                        bbox=d["bbox"],
                        class_name=d["class_name"],
                        class_id=d["class_id"],
                        confidence=d["confidence"],
                        source_model=model_name,
                    )
                    activity_detections.append(det)
                    all_detections.append(det)
            except Exception as e:
                logger.debug(f"Model {model_name} error: {e}")

        elapsed_ms = (time.time() - t0) * 1000

        return {
            "person_detections": person_detections,
            "activity_detections": activity_detections,
            "all_detections": all_detections,
            "inference_time_ms": elapsed_ms,
        }

    @property
    def num_models_loaded(self):
        return len(self.models) + (1 if self.person_detector else 0)
