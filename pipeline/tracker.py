"""
Neural Nexus — Object Tracker (DeepSORT)
======================================
Implements DeepSORT tracking to maintain identity consistency across
frames, occlusions, and variable frame rates by using visual appearance
(CNN features) in addition to basic Kalman Filtering.
"""

import numpy as np
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IoU computation for activity mapping
# ---------------------------------------------------------------------------

def iou_batch(bb_test, bb_gt):
    """
    Compute IoU between two sets of bounding boxes.

    Args:
        bb_test: (N, 4) array of [x1, y1, x2, y2]
        bb_gt: (M, 4) array of [x1, y1, x2, y2]

    Returns:
        (N, M) IoU matrix
    """
    bb_test = np.array(bb_test)
    bb_gt = np.array(bb_gt)

    if bb_test.size == 0 or bb_gt.size == 0:
        return np.empty((0, 0))

    xx1 = np.maximum(bb_test[:, 0:1], bb_gt[:, 0].T)
    yy1 = np.maximum(bb_test[:, 1:2], bb_gt[:, 1].T)
    xx2 = np.minimum(bb_test[:, 2:3], bb_gt[:, 2].T)
    yy2 = np.minimum(bb_test[:, 3:4], bb_gt[:, 3].T)

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_test = ((bb_test[:, 2] - bb_test[:, 0])
                 * (bb_test[:, 3] - bb_test[:, 1]))
    area_gt = ((bb_gt[:, 2] - bb_gt[:, 0])
               * (bb_gt[:, 3] - bb_gt[:, 1]))

    union = area_test[:, None] + area_gt[None, :] - intersection
    return intersection / np.maximum(union, 1e-6)


# ---------------------------------------------------------------------------
# Tracker Wrapper
# ---------------------------------------------------------------------------

class SORTTracker:
    """
    DeepSORT wrapper. Refactored from basic SORT.
    Uses 'deep_sort_realtime' for robust ID persistence.
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Args:
            max_age: Frames before a lost track is deleted
            min_hits: Minimum detections before a track is confirmed
            iou_threshold: Overlap required mapping activity to track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # Instantiate DeepSort object
        # Using default mobilenet embedding extractor.
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=min_hits,
            nms_max_overlap=1.0,
            max_cosine_distance=0.2,   
            nn_budget=100,
            embedder="mobilenet"
        )

    def update(self, person_detections, activity_detections=None, frame=None):
        """
        Update tracks with new detections and source frame.

        Args:
            person_detections: list of Detection objects (persons)
            activity_detections: list of Detection objects (activities)
            frame: raw BGR numpy array

        Returns:
            list of TrackedEntity dicts:
                {
                    "id": int,
                    "bbox": [x1, y1, x2, y2],
                    "activities": {class_name: confidence, ...},
                    "confirmed": bool,
                    "time_since_update": int
                }
        """
        # Convert person detections to deep_sort_realtime format:
        # [ [left, top, w, h] , confidence, detection_class ]
        ds_detections = []
        if person_detections:
            for d in person_detections:
                x1, y1, x2, y2 = d.bbox
                w = x2 - x1
                h = y2 - y1
                ds_detections.append(
                    ([x1, y1, w, h], d.confidence, d.class_name)
                )

        # Update deepsort with detections and raw frame (for appearance embed extraction)
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        entities = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            # Get bounding box (ltwh -> xyxy)
            ltrb = track.to_ltrb() 
            
            # Prevent invalid negative bbox due to filter overshoot
            bbox = [max(0, float(c)) for c in ltrb]

            entity = {
                "id": int(track.track_id),
                "bbox": bbox,
                "activities": {},
                "confirmed": True,
                "time_since_update": track.time_since_update,
            }
            entities.append(entity)

        # Map activity detections to tracked entities via IoU overlap
        if activity_detections and entities:
            self._map_activities(entities, activity_detections)

        return entities

    def _map_activities(self, entities, activity_detections):
        """
        Map activity detections to tracked entities by finding the
        entity whose bounding box has the highest IoU with each
        activity detection.
        """
        if not entities or not activity_detections:
            return

        entity_bboxes = np.array([e["bbox"] for e in entities])
        act_bboxes = np.array([getattr(d, "bbox", [0, 0, 0, 0]) for d in activity_detections],
                              dtype=float)

        if entity_bboxes.size == 0 or act_bboxes.size == 0:
            return

        iou_matrix = iou_batch(act_bboxes, entity_bboxes)

        for act_idx, det in enumerate(activity_detections):
            if iou_matrix.shape[1] == 0:
                break
            best_entity_idx = np.argmax(iou_matrix[act_idx])
            best_iou = iou_matrix[act_idx, best_entity_idx]

            if best_iou > 0.1:  # Minimum overlap threshold
                entity = entities[best_entity_idx]
                cls_name = getattr(det, "class_name", "unknown")
                conf = getattr(det, "confidence", 0.0)
                
                # Keep highest confidence for each activity
                if (cls_name not in entity["activities"]
                        or conf > entity["activities"][cls_name]):
                    entity["activities"][cls_name] = conf

    def reset(self):
        """Reset all tracks."""
        self.tracker.delete_all_tracks()
