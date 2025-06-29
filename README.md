# Player Re-Identification in Sports Videos

## Overview

This project implements a computer vision solution for **player re-identification** in sports videos. The system tracks players throughout a video and maintains consistent IDs even when players temporarily leave the frame and re-enter (e.g., during goal events).

## Problem Statement

Given a 15-second video (`15sec_input_720p.mp4`), the system:
- Detects players using a pre-trained YOLOv11 model
- Assigns unique IDs based on initial detections
- Maintains consistent IDs when players re-enter the frame
- Simulates real-time re-identification and tracking

## Technical Approach

### Core Components

1. **Player Detection**: YOLOv11 model fine-tuned for player detection
2. **Feature Extraction**: Multi-modal features (visual + spatial + temporal)
3. **Tracking Algorithm**: Kalman filter-based tracking with Hungarian assignment
4. **Re-identification**: Similarity matching using combined feature vectors
5. **ID Management**: Robust ID assignment and maintenance system

### Algorithm Pipeline

```
Input Frame → Player Detection → Feature Extraction → 
Track Association → Kalman Prediction → Re-identification → 
ID Assignment → Output with Consistent IDs
```

### Key Features

- **Multi-modal Feature Extraction**:
  - Deep CNN features (ResNet18)
  - Color histograms (HSV)
  - Texture features (LBP, gradients)
  - Spatial features (body proportions)

- **Robust Tracking**:
  - Kalman filters for motion prediction
  - Hungarian algorithm for optimal assignment
  - Spatial and temporal consistency checks

- **Re-identification**:
  - Feature similarity matching
  - Threshold-based re-ID decisions
  - Historical feature averaging

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8+ GB RAM

### Setup Instructions

1. **Clone/Download Project**
```bash
mkdir player_reidentification
cd player_reidentification
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Models and Data**
   - Download YOLOv11 model from: [Google Drive Link](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)
   - Place model file as: `models/yolov11_player_detection.pt`
   - Place input video as: `input/15sec_input_720p.mp4`

5. **Create Directory Structure**
```bash
mkdir -p models input output results
```

## Usage

### Basic Usage
```bash
python main.py --input input/15sec_input_720p.mp4 --output output/tracked_video.mp4
```

### Advanced Usage
```bash
python main.py \
    --input input/15sec_input_720p.mp4 \
    --output output/tracked_video.mp4 \
    --model models/yolov11_player_detection.pt \
    --confidence 0.5 \
    --max_disappeared 30 \
    --reid_threshold 0.7
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Input video path | `input/15sec_input_720p.mp4` |
| `--output` | Output video path | `output/tracked_video.mp4` |
| `--model` | YOLO model path | `models/yolov11_player_detection.pt` |
| `--confidence` | Detection confidence threshold | `0.5` |
| `--max_disappeared` | Max frames before losing track | `30` |
| `--reid_threshold` | Re-ID similarity threshold | `0.7` |

## Output

The system generates:

1. **Annotated Video** (`output/tracked_video.mp4`)
   - Bounding boxes with consistent IDs
   - Confidence scores
   - Color-coded tracks

2. **Tracking Results** (`output/tracked_video_results.json`)
   - Complete tracking history
   - Performance metrics
   - Re-identification statistics

3. **Visualization** (`results/tracking_metrics.png`)
   - Performance graphs
   - Track statistics
   - Detection metrics

## Performance Metrics

- **Tracking Accuracy**: Percentage of correctly maintained IDs
- **Re-ID Success Rate**: Success rate for re-entering players  
- **ID Switches**: Number of incorrect ID changes
- **Processing Speed**: Frames per second
- **Detection Consistency**: Variance in detections per frame

## Algorithm Details

### 1. Player Detection
```python
# YOLOv11 detection with confidence filtering
results = model(frame, conf=confidence_threshold, classes=[0])
```

### 2. Feature Extraction
- **Deep Features**: ResNet18 backbone (512-dim)
- **Color Features**: HSV histograms + dominant colors (54-dim)
- **Texture Features**: LBP + gradients (33-dim)
- **Spatial Features**: Aspect ratio + body proportions (5-dim)

### 3. Track Association
```python
# Cost matrix combining IoU, feature similarity, and spatial distance
cost = 1.0 - (0.4 * iou + 0.4 * feature_sim + 0.2 * spatial_consistency)
```

### 4. Re-identification
```python
# Feature similarity using cosine distance
similarity = 1 - cosine_distance(track_features, detection_features)
reid_success = similarity > reid_threshold
```

## File Structure

```
player_reidentification/
├── main.py                 # Main entry point
├── player_tracker.py       # Core tracking algorithm  
├── feature_extractor.py    # Visual feature extraction
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/
│   └── yolov11_player_detection.pt
├── input/
│   └── 15sec_input_720p.mp4
├── output/
│   ├── tracked_video.mp4
│   └── tracked_video_results.json
└── results/
    └── tracking_metrics.png
```

## Code Architecture

### Class Hierarchy
```
PlayerTracker (main tracking system)
├── PlayerTrack (individual track management)
├── FeatureExtractor (visual features)
├── KalmanTracker (motion prediction)
└── Visualizer (result visualization)
```

### Data Flow
```
Video Frame → Detection → Feature Extraction → 
Track Matching → Kalman Update → Re-ID Check → 
Output Generation
```

## Customization

### Tuning Parameters

1. **Detection Sensitivity**
   - Adjust `confidence` threshold (0.3-0.8)
   - Modify `max_disappeared` frames (20-50)

2. **Re-identification Accuracy**
   - Tune `reid_threshold` (0.5-0.9)
   - Adjust feature weights in cost matrix

3. **Performance vs Accuracy**
   - Enable/disable deep features
   - Modify feature vector dimensions

### Adding New Features

1. **Custom Feature Extractors**
```python
class CustomFeatureExtractor:
    def extract_features(self, image_crop):
        # Implement custom feature extraction
        return feature_vector
```

2. **Alternative Tracking Methods**
```python
class CustomTracker:
    def track_frame(self, frame, frame_num):
        # Implement custom tracking logic
        return detections
```

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure YOLOv11 model is in correct path
   - Check model file integrity

2. **Low Detection Rate**
   - Reduce confidence threshold
   - Check video quality and lighting

3. **Frequent ID Switches**
   - Increase `max_disappeared` parameter
   - Lower `reid_threshold` for more aggressive re-ID

4. **Slow Processing**
   - Disable deep feature extraction
   - Reduce video resolution
   - Use GPU acceleration

### Performance Optimization

1. **Speed Improvements**
   - Use smaller input resolution
   - Reduce feature vector dimensions
   - Enable GPU acceleration

2. **Accuracy Improvements**
   - Use higher confidence threshold
   - Enable