import cv2
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from collections import defaultdict
import json


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)


class VideoProcessor:
    
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        
        # Open input video
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
    def write_frame(self, frame):
        self.writer.write(frame)
        
    def release(self):
        self.cap.release()
        self.writer.release()


class KalmanTracker:
    """Kalman filter for tracking bounding box coordinates"""
    
    def __init__(self, bbox):
        # State: [x, y, w, h, dx, dy, dw, dh]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Covariance matrices
        self.kf.R *= 10  # Measurement uncertainty
        self.kf.P *= 1000  # Initial uncertainty
        self.kf.Q[-1, -1] *= 0.01  # Process uncertainty
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state - FIX: Convert to column vector
        state = self.bbox_to_state(bbox)
        self.kf.x[:4] = np.array(state).reshape(4, 1)
        
    def bbox_to_state(self, bbox):
        """Convert bbox [x1, y1, x2, y2] to state [x, y, w, h]"""
        x1, y1, x2, y2 = bbox
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return [x, y, w, h]
    
    def state_to_bbox(self, state):
        """Convert state [x, y, w, h] to bbox [x1, y1, x2, y2]"""
        # Handle both 1D and 2D state arrays
        if state.ndim > 1:
            state = state.flatten()
        
        x, y, w, h = state[:4]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return [x1, y1, x2, y2]
    
    def predict(self):
        """Predict next state"""
        self.kf.predict()
        return self.state_to_bbox(self.kf.x)
    
    def update(self, bbox):
        """Update with measurement"""
        state = self.bbox_to_state(bbox)
        self.kf.update(state)


class Visualizer:
    """Handle visualization of tracking results - IMPROVED VERSION"""
    
    def __init__(self):
        # Color palette for different track IDs
        self.colors = self.generate_colors(50)
        
    def generate_colors(self, n):
        """Generate distinct colors for track visualization"""
        colors = []
        for i in range(n):
            hue = (i * 137.508) % 360  # Golden angle
            color = self.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(tuple(map(int, color)))
        return colors
    
    def hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        h = h / 60.0
        i = int(h)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        if i == 0:
            return [v * 255, t * 255, p * 255]
        elif i == 1:
            return [q * 255, v * 255, p * 255]
        elif i == 2:
            return [p * 255, v * 255, t * 255]
        elif i == 3:
            return [p * 255, q * 255, v * 255]
        elif i == 4:
            return [t * 255, p * 255, v * 255]
        else:
            return [v * 255, p * 255, q * 255]
    
    def draw_tracking_results(self, frame, detections):
        """Draw tracking results on frame - IMPROVED VERSION"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            track_id = detection['track_id']
            bbox = detection['bbox']
            confidence = detection['confidence']
            is_stable = detection.get('stable', False)
            
            # Get color for this track ID
            color = self.colors[track_id % len(self.colors)]
            
            # Different visualization for stable vs unstable tracks
            thickness = 3 if is_stable else 2
            line_type = cv2.LINE_AA if is_stable else cv2.LINE_8
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness, line_type)
            
            # Draw stability indicator (small circle in top-right corner)
            if is_stable:
                cv2.circle(annotated_frame, (x2-10, y1+10), 5, (0, 255, 0), -1)
            else:
                cv2.circle(annotated_frame, (x2-10, y1+10), 5, (0, 165, 255), -1)
            
            # Draw track ID and confidence
            status_text = "STABLE" if is_stable else "TRACKING"
            label = f"ID:{track_id} ({confidence:.2f}) {status_text}"
            
            # Adjust font size based on bbox size
            bbox_area = (x2 - x1) * (y2 - y1)
            font_scale = min(0.8, max(0.4, bbox_area / 10000))
            font_thickness = 2 if is_stable else 1
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # Background for text
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Text
            text_color = (255, 255, 255) if is_stable else (0, 0, 0)
            cv2.putText(annotated_frame, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       text_color, font_thickness)
            
            # Draw center point (larger for stable tracks)
            center = detection['center']
            center_size = 4 if is_stable else 3
            cv2.circle(annotated_frame, 
                      (int(center[0]), int(center[1])), 
                      center_size, color, -1)
        
        # Add tracking statistics
        self.draw_statistics(annotated_frame, detections)
        
        return annotated_frame
    
    def draw_statistics(self, frame, detections):
        """Draw tracking statistics on frame"""
        h, w = frame.shape[:2]
        
        # Count stable vs unstable tracks
        stable_count = sum(1 for d in detections if d.get('stable', False))
        total_count = len(detections)
        unstable_count = total_count - stable_count
        
        # Draw statistics box
        stats_text = [
            f"Total Players: {total_count}",
            f"Stable: {stable_count}",
            f"Tracking: {unstable_count}"
        ]
        
        # Background for statistics
        max_text_width = 200
        text_height = 25
        stats_bg_height = len(stats_text) * text_height + 10
        
        cv2.rectangle(frame, (10, 10), (10 + max_text_width, 10 + stats_bg_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (10 + max_text_width, 10 + stats_bg_height), 
                     (255, 255, 255), 2)
        
        # Draw statistics text
        for i, text in enumerate(stats_text):
            y_pos = 35 + i * text_height
            color = (0, 255, 0) if "Stable" in text else (255, 255, 255)
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def draw_trajectory(self, frame, tracking_history, track_id, max_points=30):
        """Draw trajectory for a specific track - IMPROVED"""
        if track_id not in tracking_history:
            return frame
        
        trajectory = tracking_history[track_id]
        if len(trajectory) < 2:
            return frame
        
        # Get recent points
        recent_points = trajectory[-max_points:]
        
        # Draw trajectory
        color = self.colors[track_id % len(self.colors)]
        
        for i in range(1, len(recent_points)):
            pt1 = tuple(map(int, recent_points[i-1]['center']))
            pt2 = tuple(map(int, recent_points[i]['center']))
            
            # Fade older points and vary thickness
            alpha = i / len(recent_points)
            thickness = max(1, int(4 * alpha))
            
            # Different color intensity based on stability
            is_stable = recent_points[i].get('stable', False)
            if is_stable:
                trajectory_color = color
            else:
                # Lighter color for unstable tracks
                trajectory_color = tuple(int(c * 0.7) for c in color)
            
            cv2.line(frame, pt1, pt2, trajectory_color, thickness)
        
        return frame
    
class MetricsCalculator:
    """Calculate tracking performance metrics"""
    
    def __init__(self):
        self.frame_count = 0
        self.total_detections = 0
        self.id_switches = 0
        self.track_lengths = defaultdict(int)
        self.detection_counts = []
        
        # For calculating metrics over time
        self.metrics_history = {
            'frame': [],
            'active_tracks': [],
            'total_detections': [],
            'avg_confidence': []
        }
        
    def update_frame_metrics(self, detections, frame_num):
        """Update metrics for current frame"""
        self.frame_count = frame_num
        self.total_detections += len(detections)
        self.detection_counts.append(len(detections))
        
        # Track active IDs
        active_ids = [d['track_id'] for d in detections]
        
        # Update track lengths
        for track_id in active_ids:
            self.track_lengths[track_id] += 1
        
        # Calculate average confidence
        if detections:
            avg_conf = np.mean([d['confidence'] for d in detections])
        else:
            avg_conf = 0
        
        # Store metrics
        self.metrics_history['frame'].append(frame_num)
        self.metrics_history['active_tracks'].append(len(active_ids))
        self.metrics_history['total_detections'].append(len(detections))
        self.metrics_history['avg_confidence'].append(avg_conf)
    
    def get_metrics(self):
        """Get comprehensive metrics"""
        if self.frame_count == 0:
            return {}
        
        return {
            'total_frames': self.frame_count,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': self.total_detections / self.frame_count,
            'unique_tracks': len(self.track_lengths),
            'avg_track_length': np.mean(list(self.track_lengths.values())) if self.track_lengths else 0,
            'longest_track': max(self.track_lengths.values()) if self.track_lengths else 0,
            'id_switches': self.id_switches,
            'detection_variance': np.var(self.detection_counts) if self.detection_counts else 0
        }
    
    def save_visualization(self, output_path):
        """Save metrics visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Active tracks over time
        axes[0, 0].plot(self.metrics_history['frame'], self.metrics_history['active_tracks'])
        axes[0, 0].set_title('Active Tracks Over Time')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Number of Active Tracks')
        axes[0, 0].grid(True)
        
        # Detections per frame
        axes[0, 1].plot(self.metrics_history['frame'], self.metrics_history['total_detections'])
        axes[0, 1].set_title('Detections Per Frame')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Number of Detections')
        axes[0, 1].grid(True)
        
        # Average confidence over time
        axes[1, 0].plot(self.metrics_history['frame'], self.metrics_history['avg_confidence'])
        axes[1, 0].set_title('Average Detection Confidence')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].grid(True)
        
        # Track length distribution
        if self.track_lengths:
            track_lengths = list(self.track_lengths.values())
            axes[1, 1].hist(track_lengths, bins=20, alpha=0.7)
            axes[1, 1].set_title('Track Length Distribution')
            axes[1, 1].set_xlabel('Track Length (frames)')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate union
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_results(data, output_path):
    """Save results to JSON file with NumPy support"""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyJSONEncoder)


def convert_numpy_types(obj):
    """Recursively convert NumPy types to native Python types"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj