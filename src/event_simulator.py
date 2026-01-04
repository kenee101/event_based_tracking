"""
Event Camera Simulator
Converts frame-based video to event stream using change detection
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
# import cv2

@dataclass
class Event:
    """Single event representation"""
    x: int
    y: int
    t: float  # timestamp in microseconds
    p: int    # polarity (1 for increase, -1 for decrease)

class EventSimulator:
    """Simulates event camera behavior from frame sequences"""
    
    def __init__(self, contrast_threshold: float = 0.2, 
                 temporal_resolution: float = 1000.0):
        """
        Args:
            contrast_threshold: Minimum log intensity change to trigger event
            temporal_resolution: Time resolution in microseconds
        """
        self.C = contrast_threshold
        self.temporal_res = temporal_resolution
        self.last_frame = None
        self.reference_values = None
        self.current_time = 0
        
    def frame_to_events(self, frame: np.ndarray, 
                       timestamp: float = None) -> List[Event]:
        """
        Convert a frame to events based on intensity changes
        
        Args:
            frame: Grayscale image (H, W)
            timestamp: Current timestamp in microseconds
            
        Returns:
            List of Event objects
        """
        if timestamp is None:
            timestamp = self.current_time
            self.current_time += self.temporal_res
            
        # Convert to float and normalize
        frame = frame.astype(np.float32) / 255.0
        
        events = []
        
        if self.reference_values is None:
            # Initialize reference values
            self.reference_values = np.log(frame + 1e-5)
            self.last_frame = frame.copy()
            return events
        
        # Compute log intensity
        log_intensity = np.log(frame + 1e-5)
        
        # Find pixels that crossed threshold
        delta = log_intensity - self.reference_values
        
        # Positive events (brightness increase)
        pos_mask = delta >= self.C
        # Negative events (brightness decrease)
        neg_mask = delta <= -self.C
        
        # Generate events for positive changes
        pos_coords = np.argwhere(pos_mask)
        for y, x in pos_coords:
            events.append(Event(x=int(x), y=int(y), t=timestamp, p=1))
            self.reference_values[y, x] = log_intensity[y, x]
        
        # Generate events for negative changes
        neg_coords = np.argwhere(neg_mask)
        for y, x in neg_coords:
            events.append(Event(x=int(x), y=int(y), t=timestamp, p=-1))
            self.reference_values[y, x] = log_intensity[y, x]
        
        self.last_frame = frame.copy()
        
        return events
    
    def reset(self):
        """Reset simulator state"""
        self.last_frame = None
        self.reference_values = None
        self.current_time = 0
    
    @staticmethod
    def events_to_frame(events: List[Event], shape: Tuple[int, int],
                       polarity_mode: str = 'mixed') -> np.ndarray:
        """
        Visualize events as a frame
        
        Args:
            events: List of events
            shape: (height, width) of output frame
            polarity_mode: 'mixed', 'positive', 'negative'
            
        Returns:
            Visualization frame
        """
        frame = np.zeros(shape, dtype=np.float32)
        
        for event in events:
            if 0 <= event.y < shape[0] and 0 <= event.x < shape[1]:
                if polarity_mode == 'positive' and event.p > 0:
                    frame[event.y, event.x] += 1
                elif polarity_mode == 'negative' and event.p < 0:
                    frame[event.y, event.x] += 1
                elif polarity_mode == 'mixed':
                    frame[event.y, event.x] += event.p
        
        # Normalize for visualization
        if np.max(np.abs(frame)) > 0:
            frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
        
        return (frame * 255).astype(np.uint8)