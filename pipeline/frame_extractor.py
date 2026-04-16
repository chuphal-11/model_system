"""
Neural Nexus — Frame Extractor
================================
Extracts frames from video files or live camera feeds.
Yields (frame_number, timestamp_sec, frame_image) tuples.
"""

import cv2
import logging
import threading
import time

logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Extracts frames from a video source (file or webcam).

    Usage:
        extractor = FrameExtractor("video.mp4", sample_rate=2)
        for frame_num, timestamp, frame in extractor:
            process(frame)
    """

    def __init__(self, source, sample_rate=1, max_frames=None):
        """
        Args:
            source: Video file path or camera index (int or "0")
            sample_rate: Process every Nth frame (1 = all frames)
            max_frames: Stop after this many processed frames (None = all)
        """
        self.source = source
        self.sample_rate = max(1, sample_rate)
        self.max_frames = max_frames
        self._cap = None
        self.fps = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.is_live = False
        
        self._thread = None
        self._latest_frame = None
        self._running = False
        self._lock = threading.Lock()

    def open(self):
        """Open the video source."""
        if isinstance(self.source, int) or (
            isinstance(self.source, str) and self.source.isdigit()
        ):
            src = int(self.source)
            self._cap = cv2.VideoCapture(src)
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = 0
            self.is_live = True
            logger.info(f"Opened webcam {src} @ {self.fps:.1f} FPS (Multi-threaded)")
        else:
            self._cap = cv2.VideoCapture(self.source)
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.is_live = False
            logger.info(
                f"Opened video: {self.source} "
                f"({self.total_frames} frames, {self.fps:.1f} FPS, "
                f"{self.total_frames / self.fps:.1f}s)"
            )

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.is_live:
            self._running = True
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
            
        return self

    def _reader(self):
        """Daemon thread loop for constantly consuming frames from live stream."""
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                continue
            with self._lock:
                self._latest_frame = frame

    def close(self):
        """Release the video source."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    def __iter__(self):
        """
        Yield (frame_number, timestamp_sec, frame) tuples.
        frame is a numpy BGR image.
        """
        if self._cap is None:
            self.open()

        frame_idx = 0
        processed_count = 0

        while True:
            if self.is_live:
                with self._lock:
                    frame = self._latest_frame
                if frame is None:
                    time.sleep(0.01)
                    continue
                ret = True
            else:
                ret, frame = self._cap.read()

            if not ret:
                break

            if frame_idx % self.sample_rate == 0:
                timestamp = frame_idx / self.fps if self.fps > 0 else 0.0
                yield frame_idx, timestamp, frame
                processed_count += 1

                if (self.max_frames is not None
                        and processed_count >= self.max_frames):
                    break

            frame_idx += 1

        logger.info(f"Extracted {processed_count} frames "
                     f"(from {frame_idx} total)")

    @property
    def duration_seconds(self):
        """Total video duration in seconds."""
        if self.fps > 0 and self.total_frames > 0:
            return self.total_frames / self.fps
        return 0.0
