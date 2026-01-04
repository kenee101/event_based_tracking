import time
import numpy as np
from collections import deque
from typing import List
from event_simulator import Event


class EventBasedTracker:
    """
    Simple event-based tracker using spatio-temporal clustering
    """

    def __init__(self, spatial_window: int = 30, temporal_window: int = 50000):
        """
        Args:
            spatial_window: max pixel distance for clustering
            temporal_window: max time difference (μs)
        """
        self.spatial_window = spatial_window
        self.temporal_window = temporal_window

        self.event_buffer = deque()
        self.total_events = 0
        self.latencies = []

    def update(self, events: List[Event]):
        start = time.perf_counter()

        tracked_objects = []

        for event in events:
            self.event_buffer.append(event)
            self.total_events += 1

        # Remove old events
        if self.event_buffer:
            newest_time = self.event_buffer[-1].t
            while self.event_buffer and newest_time - self.event_buffer[0].t > self.temporal_window:
                self.event_buffer.popleft()

        # Simple centroid-based clustering
        if self.event_buffer:
            xs = np.array([e.x for e in self.event_buffer])
            ys = np.array([e.y for e in self.event_buffer])

            centroid = (int(xs.mean()), int(ys.mean()))
            tracked_objects.append({
                "centroid": centroid,
                "num_events": len(self.event_buffer)
            })

        latency_us = (time.perf_counter() - start) * 1e6
        self.latencies.append(latency_us)

        return tracked_objects

    def get_metrics(self):
        return {    
            "avg_latency_us": np.mean(self.latencies) if self.latencies else 0.0,
            "max_latency_us": np.max(self.latencies) if self.latencies else 0.0,
            "total_events": self.total_events,
            "events_per_update": self.total_events / max(len(self.latencies), 1)
        }
