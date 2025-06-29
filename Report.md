<img width="1352" alt="Screenshot 2025-06-30 at 2 28 15 AM" src="https://github.com/user-attachments/assets/ab5e9ada-7dd0-4eaa-83ff-1645127992d9" /># Player Re-Identification System: Project Report

## 1. Approach and Methodology

My primary goal was to develop a system that could consistently track and identify players across a sports video, maintaining their unique IDs even if they temporarily disappear from view.

### System Pipeline

The system operates in a comprehensive pipeline:

#### Player Detection
- Used a pre-trained **YOLOv11 model** to accurately detect players in each video frame

#### Multi-Modal Feature Extraction ("Player Fingerprint")
For every detected player, I extracted a unique set of features to describe their appearance. This was crucial for distinguishing players. The system combines:

- **Deep CNN Features (ResNet18)**: Powerful visual descriptors learned by a deep neural network
- **Color Features**: HSV histograms and dominant colors to capture jersey and shorts color information
- **Texture Features**: Local Binary Patterns (LBP) and Gradient Magnitude for surface details, and Edge Density for sharpness
- **Spatial Features**: Aspect ratio, body part variances, and symmetry to describe player shape

#### Tracking and Association
- **Kalman Filter**: Used to predict player movement and position across frames, enabling smooth tracking
- **Hungarian Algorithm**: Efficiently matched new detections to existing player tracks, minimizing ID swaps
- **Association Cost**: Carefully weighted matching system, prioritizing feature similarity (appearance) heavily over simple bounding box overlap or position, which was vital in crowded scenes

#### Re-identification
When a player disappeared and reappeared, the system leveraged their stored "feature history" to match them to an existing "lost" track, re-assigning their original ID rather than creating a new one.

#### ID Management
A robust system that:
- Assigns new IDs
- Updates existing tracks
- Removes tracks that had been "disappeared" for too long

## 2. Techniques Tried and Their Outcomes

### Initial Feature Set (Jersey Color Only)
**Attempt**: Started by primarily using just jersey color for player identification.

**Outcome**: Poor performance. Frequent ID switches, especially with players from the same team or in crowded scenes. Re-identification barely worked.

### Jersey Color with Movement Style
**Attempt**: Added basic movement patterns (speed, direction from Kalman filter) to jersey color for distinction.

**Outcome**: Small improvement. Movement helped when colors were similar, but wasn't robust enough for fast-paced or unpredictable player actions.

### Jersey Color, Movement Style, and Team Information
**Attempt**: Incorporated jersey color, movement, and attempted to group players by team based on their colors.

**Outcome**: Still insufficient. Poor video quality was a major hurdle. Players were often pixelated and unclear, making it extremely difficult to reliably identify individuals or extract solid visual data.

### Adding Deep CNN Features (ResNet18 Embeddings) ⭐
**Attempt**: Integrated a powerful pre-trained AI model (ResNet18) for feature extraction.

**Outcome**: **Significantly improved accuracy** and reduced ID switches. These "deep features" captured much more subtle visual details, making players more distinguishable even with video quality issues. This was the most impactful breakthrough.

### Refining Association Cost Function ⭐
**Attempt**: Fine-tuned how different factors (overlap, visual features, movement) were weighted when matching players to tracks.

**Outcome**: **Crucial for success**. By giving much more importance to a player's visual "fingerprint" (~55% weight) over just their position, ID swaps were drastically reduced, especially when players were close together.

### Improving Re-identification Logic
**Attempt**: Used a history of each player's "fingerprints" to re-identify them, not just their latest one. Also allowed tracks to be "lost" for longer before being permanently discarded.

**Outcome**: Much more robust re-identification. Players who disappeared and reappeared were more accurately linked back to their original IDs, even if their appearance changed slightly or they moved significantly while off-screen.

## 3. Challenges Encountered

### Frequent ID Swapping in Crowded Scenes
**Challenge**: Players overlapping made it difficult to assign the correct ID.

**Solution**: Prioritizing visual appearance (features) in the matching algorithm was key.

### Occlusion (Hidden Players)
**Challenge**: Players getting completely hidden made them disappear from tracking.

**Mitigation**: Kalman filter helped for short gaps, and re-identification for longer disappearances, but full, long-term occlusion remains a challenge.

### Feature Robustness to Pose/Lighting Changes
**Challenge**: Player appearance could change with movement or lighting, making consistent identification difficult.

**Mitigation**: Deep CNN features were more adaptable to these changes, and averaging feature history helped stabilize identification.

### Video Quality
**Challenge**: Poor video quality leading to pixelation made it inherently difficult to recognize players clearly and reliably. This compounded all other challenges.

### Computational Performance
**Challenge**: Running powerful AI models is resource-intensive, especially without a dedicated graphics card.

**Mitigation**: Offered an option to simplify features for faster processing, though at a cost to accuracy.

## 4. Incompleteness and Future Work

The current system is a strong foundation, but there are many opportunities for improvement:

### Short-term Enhancements
- **Smarter Occlusion Handling**: More advanced methods to predict where hidden players will reappear
- **Improved Movement Prediction**: Using more sophisticated AI to forecast player trajectories
- **Better Performance Optimization**: Making the system faster for real-time use

### Medium-term Goals
- **Gait Recognition**: Exploring identification of players by their unique walking/running style
- **Jersey Number Identification**: Developing OCR capabilities to read and use jersey numbers for definitive identification
- **Team Identification**: Automatically classifying players by their team affiliation

### Long-term Vision
- **Event-Driven Tracking**: Making the system aware of game events (like goals) to adjust tracking behavior
- **Multi-Camera Tracking**: Extending the system to track players across multiple cameras in a stadium
- **Advanced AI for Matching**: Using cutting-edge AI models for even more robust player matching
- **Interactive Visualization Tools**: Creating comprehensive ways to explore tracking data and player movement patterns

## Key Takeaways

1. **Deep learning features** (ResNet18) were the most significant improvement over traditional computer vision approaches
2. **Proper weighting** of association costs is crucial for reducing ID swaps
3. **Feature history** and temporal consistency are essential for robust re-identification
4. **Video quality** remains a fundamental limiting factor that affects all other aspects of the system
5. **Multi-modal feature extraction** provides better robustness than any single feature type alone

The system demonstrates strong potential for sports analytics applications, with clear pathways for continued improvement and expansion.
<img width="1352" alt="Screenshot 2025-06-30 at 2 28 15 AM" src="https://github.com/user-attachments/assets/2737502e-6f57-4577-ab75-ecf69fcbfc50" />

