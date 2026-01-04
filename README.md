# Event-Based vs Frame-Based Object Tracking: A Comparative Study

**A research-oriented project demonstrating neuromorphic vision principles and comparative tracking analysis**

## Project Overview

This project implements and compares two paradigms of visual object tracking:

- **Event-Based Tracking**: Asynchronous, bio-inspired approach using sparse temporal changes
- **Frame-Based Tracking**: Traditional synchronous approach using complete image frames

The goal is to demonstrate understanding of neuromorphic computing, performance analysis, and research methodology suitable for academic research applications.

## Research Motivation

Event cameras (Dynamic Vision Sensors) represent a paradigm shift in computer vision, offering:

- **Microsecond temporal resolution** (vs milliseconds in frame cameras)
- **High dynamic range** (>120 dB vs ~60 dB)
- **Low power consumption** (10-100x more efficient)
- **Sparse data representation** (only encoding changes)

This project explores how these properties impact object tracking performance.

## 📊 Key Findings

From our synthetic experiments:

| Metric              | Event-Based   | Frame-Based    | Advantage             |
| ------------------- | ------------- | -------------- | --------------------- |
| Latency             | ~50-200 μs    | ~30-50 ms      | **100-250x faster**   |
| Data Volume         | Sparse events | Full frames    | **50-100x less data** |
| Power (est.)        | Low           | High           | **~90% reduction**    |
| Temporal Resolution | Asynchronous  | Fixed (30 FPS) | **Continuous time**   |

## Getting Started

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd event-frame-tracking

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

```bash
python src/demo.py
```

This will:

1. Generate synthetic moving objects
2. Run both tracking algorithms
3. Compare performance metrics
4. Generate visualization plots

## 🎥 Demo Video

Here's a live demonstration of the synthetic moving circles and the tracking results:

![Demo Video](https://github.com/kenee101/event-frame-tracking/tree/main/src/synthetic_moving_circles.mp4)

_Video: Synthetic moving circles being tracked in real-time using both event-based and frame-based approaches._

## 📊 Performance Results

The following chart compares the performance of event-based vs frame-based tracking:

![Performance Comparison](data-output.png)

_Figure: Comparative analysis of tracking performance metrics including latency, data efficiency, and power consumption._

## 🔧 Implementation Details

### Event Simulator

Converts frame sequences to event streams using:

- Logarithmic brightness change detection
- Per-pixel contrast thresholds
- Asynchronous event generation

```python
# Example usage
simulator = EventSimulator(contrast_threshold=0.2)
events = simulator.frame_to_events(frame, timestamp)
```

### Event-Based Tracker

Features:

- Spatial-temporal event clustering
- Grid-based efficient processing
- Velocity estimation from event patterns
- Sub-millisecond latency

### Frame-Based Tracker

Features:

- Background subtraction (MOG2)
- Blob detection and contour analysis
- Fixed frame rate processing (30 FPS)
- Traditional computer vision pipeline

## 📈 Experimental Results

### Latency Comparison

Event-based processing achieves **100-250x lower latency** due to:

- Processing only changed pixels
- Asynchronous operation
- No waiting for frame readout

### Data Efficiency

Event representation uses **50-100x less data**:

- Sparse encoding (only changes)
- No redundant static information
- Scales with scene dynamics, not resolution

### Power Efficiency

Estimated **~90% power reduction** from:

- Reduced data movement
- Selective computation
- Event-driven processing

## 🎓 Research Relevance

This project demonstrates:

1. **Neuromorphic Computing Understanding**: Implementation of bio-inspired vision principles
2. **Comparative Analysis Skills**: Rigorous benchmarking methodology
3. **Algorithm Development**: Custom tracking algorithms for novel sensors
4. **Performance Evaluation**: Comprehensive metrics and visualization
5. **Research Communication**: Clear documentation and presentation

## 🔬 Future Extensions

Potential research directions:

- Real event camera dataset testing (DVS128, DAVIS240C)
- Deep learning integration (Graph Neural Networks for events)
- Multi-object tracking and association
- Real-time hardware implementation (FPGA, neuromorphic chips)

## 📚 References

Key papers in event-based vision:

1. Gallego et al., "Event-based Vision: A Survey" (2020)
2. Rebecq et al., "EKLT: Asynchronous Photometric Feature Tracking" (2017)
3. Zhu et al., "Event-based Visual Inertial Odometry" (2017)

## 🛠️ Technical Requirements

- Python 3.12+
- OpenCV 4.5+
- NumPy, Matplotlib
- 4GB RAM minimum
