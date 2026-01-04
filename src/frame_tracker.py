# pylint: disable=no-member

import time
import numpy as np
import cv2


class FrameBasedTracker:
    """
    Frame-based tracker using background subtraction
    """

    def __init__(self, frame_rate: float = 30.0):
        self.frame_interval = 1.0 / frame_rate
        self.last_timestamp = None

        self.total_frames = 0
        self.latencies = []

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2() 

    def update(self, frame: np.ndarray, timestamp: float):
        start = time.perf_counter()

        # Enforce frame rate (simulate blocking)
        if self.last_timestamp is not None:
            elapsed = timestamp - self.last_timestamp
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)

        self.last_timestamp = timestamp
        self.total_frames += 1

        # Full-frame processing
        fg_mask = self.bg_subtractor.apply(frame)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        objects = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                objects.append({
                    "bbox": (x, y, w, h)
                })

        latency_ms = (time.perf_counter() - start) * 1000
        self.latencies.append(latency_ms)

        return objects

    def get_metrics(self):
        """
        Calculate and return tracking performance metrics.
        
        Returns:
            dict: A dictionary containing the following metrics:
                - avg_latency_ms (float): Average processing time per frame in milliseconds
                - max_latency_ms (float): Maximum processing time observed in milliseconds
                - total_frames (int): Total number of frames processed
                - effective_fps (float): Actual frames processed per second
                - target_fps (float): Target frame rate set during initialization
        """
        avg_latency = np.mean(self.latencies) if self.latencies else 0.0

        return {
            "avg_latency_ms": avg_latency,
            "max_latency_ms": np.max(self.latencies) if self.latencies else 0.0,
            "total_frames": self.total_frames,
            "effective_fps": 1000.0 / avg_latency if avg_latency > 0 else 0.0,
            "target_fps": 1.0 / self.frame_interval
        }
