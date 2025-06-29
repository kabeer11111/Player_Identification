import numpy as np
import cv2
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from collections import defaultdict, deque
import time

from feature_extractor import FeatureExtractor
from utils import KalmanTracker, calculate_iou


class PlayerTrack:
   
    
    def __init__(self, track_id, detection, frame_num):
        self.track_id = track_id
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.features = detection['features']
        self.frame_num = frame_num
        
        # Tracking state
        self.kalman = KalmanTracker(self.bbox)
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        
        # Re-identification features - IMPROVED
        self.feature_history = deque(maxlen=15)  # Increased history
        self.feature_history.append(self.features)
        
        # Position history for spatial consistency - IMPROVED
        self.position_history = deque(maxlen=10)  # Increased history
        self.velocity_history = deque(maxlen=5)
        self.position_history.append(self.get_center())
        
        # Confidence tracking
        self.confidence_history = deque(maxlen=10)
        self.confidence_history.append(detection['confidence'])
        
        # First and last seen
        self.first_seen = frame_num
        self.last_seen = frame_num
        
        # Stability tracking
        self.stable_frames = 0
        self.unstable_frames = 0
        
    def update(self, detection, frame_num):
        self.time_since_update = 0
        self.hits += 1
        self.last_seen = frame_num
        
        # Calculate velocity for stability check
        old_center = self.get_center()
        
        # Update Kalman filter
        self.kalman.update(detection['bbox'])
        
        # Update features
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.features = detection['features']
        self.feature_history.append(self.features)
        self.confidence_history.append(self.confidence)
        
        # Update position and velocity
        new_center = self.get_center()
        self.position_history.append(new_center)
        
        # Calculate velocity
        velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
        self.velocity_history.append(velocity)
        
        # Update stability
        movement = np.sqrt(velocity[0]**2 + velocity[1]**2)
        if movement < 20:  # Small movement = stable
            self.stable_frames += 1
            self.unstable_frames = max(0, self.unstable_frames - 1)
        else:
            self.unstable_frames += 1
            self.stable_frames = max(0, self.stable_frames - 1)
        
    def predict(self):
        self.bbox = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        return self.bbox
        
    def get_center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
        
    def get_average_features(self):
        if len(self.feature_history) == 0:
            return None
        return np.mean(self.feature_history, axis=0)
    
    def get_weighted_features(self):
        if len(self.feature_history) == 0:
            return None
        
        features = np.array(self.feature_history)
        weights = np.linspace(0.5, 1.0, len(features))  # Recent features get higher weight
        weighted_features = np.average(features, axis=0, weights=weights)
        return weighted_features
    
    def get_predicted_position(self):
        if len(self.velocity_history) < 2:
            return self.get_center()
        
        # Average recent velocities
        recent_velocities = list(self.velocity_history)[-3:]
        avg_velocity = np.mean(recent_velocities, axis=0)
        
        current_center = self.get_center()
        predicted_center = (
            current_center[0] + avg_velocity[0],
            current_center[1] + avg_velocity[1]
        )
        
        return predicted_center
    
    def is_stable(self):
        return self.stable_frames > self.unstable_frames and self.stable_frames >= 3
        
    def is_valid(self):
        return self.hits >= 3  # Require at least 3 hits


class PlayerTracker:
    def __init__(self, model_path, confidence=0.5, max_disappeared=30, reid_threshold=0.7, 
                 player_class=2, ball_class=0, detect_ball=False):
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.confidence = confidence
        
        # Class settings
        self.player_class = player_class
        self.ball_class = ball_class
        self.detect_ball = detect_ball
        
        # Tracking parameters - IMPROVED
        self.max_disappeared = max_disappeared
        self.reid_threshold = reid_threshold
        self.min_detection_size = 100  # Minimum bbox area
        self.max_distance_threshold = 150  # Maximum movement between frames
        
        # Feature extractor for re-identification
        self.feature_extractor = FeatureExtractor()
        
        # Tracking state
        self.active_tracks = {}
        self.inactive_tracks = {}
        self.next_id = 1  # Start from 1, not 0
        
        # Ball tracking
        self.ball_detections = []
        self.ball_history = deque(maxlen=10)
        
        # Metrics
        self.reid_successes = 0
        self.reid_attempts = 0
        
        # History
        self.tracking_history = defaultdict(list)
        
        # Frame-to-frame tracking for stability
        self.previous_detections = []
        self.frame_count = 0
        
    def detect_players(self, frame):
        results = self.model(frame, conf=self.confidence, classes=[self.player_class])
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Filter small detections
                    area = (x2 - x1) * (y2 - y1)
                    if area < self.min_detection_size:
                        continue
                    
                    # Extract features for this detection
                    player_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if player_crop.size > 0:
                        features = self.feature_extractor.extract_features(player_crop)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'features': features,
                            'area': area,
                            'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                        })
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections
    
    def track_frame(self, frame, frame_num):
        self.frame_count = frame_num
        
        # Detect players
        player_detections = self.detect_players(frame)
        
        # Predict existing tracks
        for track in self.active_tracks.values():
            track.predict()
        
        # Associate detections with tracks
        if len(player_detections) > 0 and len(self.active_tracks) > 0:
            matched_tracks, unmatched_detections, unmatched_tracks = self.associate_detections_to_tracks(
                player_detections, list(self.active_tracks.values())
            )
        else:
            matched_tracks = []
            unmatched_detections = list(range(len(player_detections)))
            unmatched_tracks = list(self.active_tracks.keys())  # FIX: Use keys instead of indices
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            # FIX: Get track_id correctly
            track_ids = list(self.active_tracks.keys())
            track_id = track_ids[track_idx]
            self.active_tracks[track_id].update(player_detections[det_idx], frame_num)
        
        # Handle unmatched detections (new players or re-identification)
        for det_idx in unmatched_detections:
            detection = player_detections[det_idx]
            
            # Try re-identification first
            reid_id = self.attempt_reidentification(detection, frame_num)
            
            if reid_id is not None:
                # Re-identified player
                track = self.inactive_tracks.pop(reid_id)
                track.update(detection, frame_num)
                self.active_tracks[reid_id] = track
                self.reid_successes += 1
            else:
                # New player - assign new ID
                new_id = self.next_id
                self.next_id += 1
                self.active_tracks[new_id] = PlayerTrack(new_id, detection, frame_num)
        
        # Handle unmatched tracks (disappeared players)
        tracks_to_remove = []
        for track_id in unmatched_tracks:
            if track_id in self.active_tracks:  # Safety check
                track = self.active_tracks[track_id]
                if track.time_since_update >= self.max_disappeared:
                    # Move to inactive for potential re-identification
                    self.inactive_tracks[track_id] = track
                    tracks_to_remove.append(track_id)
        
        # Remove tracks that moved to inactive
        for track_id in tracks_to_remove:
            if track_id in self.active_tracks:
                del self.active_tracks[track_id]
        
        # Clean up old inactive tracks
        inactive_to_remove = []
        for track_id, track in self.inactive_tracks.items():
            if frame_num - track.last_seen >= 2 * self.max_disappeared:
                inactive_to_remove.append(track_id)
        
        for track_id in inactive_to_remove:
            del self.inactive_tracks[track_id]
        
        # Record tracking history
        self.record_tracking_history(frame_num)
        
        # Store current detections for next frame
        self.previous_detections = player_detections
        
        # Return current detections with track IDs
        return self.get_current_detections()
    
    def associate_detections_to_tracks(self, detections, tracks):
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        # Compute enhanced cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for t, track in enumerate(tracks):
            for d, detection in enumerate(detections):
                # 1. IoU similarity
                iou = calculate_iou(track.bbox, detection['bbox'])
                
                # 2. Feature similarity (cosine similarity)
                track_features = track.get_weighted_features()  # Use weighted features
                if track_features is not None:
                    try:
                        feature_sim = 1 - cosine(track_features, detection['features'])
                        feature_sim = max(0, feature_sim)
                    except:
                        feature_sim = 0
                else:
                    feature_sim = 0
                
                # 3. Spatial distance penalty (improved)
                track_center = track.get_center()
                det_center = detection['center']
                spatial_dist = np.sqrt((track_center[0] - det_center[0])**2 + 
                                     (track_center[1] - det_center[1])**2)
                
                # Predict where track should be
                predicted_center = track.get_predicted_position()
                predicted_dist = np.sqrt((predicted_center[0] - det_center[0])**2 + 
                                       (predicted_center[1] - det_center[1])**2)
                
                # Use minimum of actual and predicted distance
                effective_dist = min(spatial_dist, predicted_dist)
                spatial_penalty = min(1.0, effective_dist / self.max_distance_threshold)
                
                # 4. Size consistency
                track_area = (track.bbox[2] - track.bbox[0]) * (track.bbox[3] - track.bbox[1])
                det_area = detection['area']
                size_ratio = min(track_area, det_area) / max(track_area, det_area)
                size_similarity = size_ratio
                
                # 5. Confidence boost for stable tracks
                stability_bonus = 0.1 if track.is_stable() else 0
                
                # In player_tracker.py -> associate_detections_to_tracks
                # Emphasize features over location to handle crowded scenes
                cost = 1.0 - (
                    0.15 * iou +                    # Drastically lower the weight of IoU
                    0.55 * feature_sim +            # Make appearance the most critical factor
                    0.15 * (1 - spatial_penalty) +  # Spatial location is still a clue, but not dominant
                    0.15 * size_similarity  #It needs few tweak will have a look into it......
                )
                
                # Penalize very far detections heavily
                if effective_dist > self.max_distance_threshold:
                    cost += 0.5
                
                cost_matrix[t, d] = max(0, cost)  # Ensure non-negative
        
        # Hungarian algorithm
        try:
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        except:
            # Fallback if algorithm fails
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # Filter out associations with high cost - IMPROVED THRESHOLD
        matched_tracks = []
        association_threshold = 0.6  # Lower threshold for better associations
        
        for t, d in zip(track_indices, detection_indices):
            if cost_matrix[t, d] < association_threshold:
                matched_tracks.append([t, d])
        
        # Find unmatched detections
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in detection_indices or \
               cost_matrix[track_indices[list(detection_indices).index(d)], d] >= association_threshold:
                unmatched_detections.append(d)
        
        # Find unmatched tracks
        unmatched_tracks = []
        for t in range(len(tracks)):
            if t not in track_indices or \
               cost_matrix[t, detection_indices[list(track_indices).index(t)]] >= association_threshold:
                # Get the actual track ID
                track_id = tracks[t].track_id
                unmatched_tracks.append(track_id)
        
        return matched_tracks, unmatched_detections, unmatched_tracks
    
    def attempt_reidentification(self, detection, frame_num):
        if len(self.inactive_tracks) == 0:
            return None
        
        self.reid_attempts += 1
        
        best_match_id = None
        best_similarity = 0
        
        for track_id, track in self.inactive_tracks.items():
            # Skip tracks that have been inactive too long
            if frame_num - track.last_seen > self.max_disappeared:
                continue
            
            # Feature similarity
            track_features = track.get_weighted_features()  # Use weighted features
            if track_features is not None:
                try:
                    feature_sim = 1 - cosine(track_features, detection['features'])
                    
                    # Additional spatial check
                    last_position = track.position_history[-1] if track.position_history else (0, 0)
                    current_position = detection['center']
                    spatial_dist = np.sqrt((last_position[0] - current_position[0])**2 + 
                                         (last_position[1] - current_position[1])**2)
                    
                    # Allow larger distance for re-identification
                    max_reid_distance = self.max_distance_threshold * 2
                    if spatial_dist > max_reid_distance:
                        continue
                    
                    # Spatial consistency bonus
                    spatial_bonus = max(0, 1 - spatial_dist / max_reid_distance) * 0.1
                    
                    total_similarity = feature_sim + spatial_bonus
                    
                    if total_similarity > best_similarity and total_similarity > self.reid_threshold:
                        best_similarity = total_similarity
                        best_match_id = track_id
                        
                except Exception as e:
                    print(f"Error in re-identification: {e}")
                    continue
        
        return best_match_id
    
    def record_tracking_history(self, frame_num):
        for track_id, track in self.active_tracks.items():
            if track.is_valid():
                self.tracking_history[track_id].append({
                    'frame': frame_num,
                    'bbox': track.bbox.copy(),
                    'confidence': track.confidence,
                    'center': track.get_center(),
                    'stable': track.is_stable()
                })
    
    def get_current_detections(self):
        detections = []
        
        # Add player detections
        for track_id, track in self.active_tracks.items():
            if track.is_valid() and track.time_since_update == 0:
                detections.append({
                    'track_id': track_id,
                    'bbox': track.bbox,
                    'confidence': track.confidence,
                    'center': track.get_center(),
                    'type': 'player',
                    'stable': track.is_stable()
                })
        
        return detections
    
    def get_tracking_history(self):
        return dict(self.tracking_history)