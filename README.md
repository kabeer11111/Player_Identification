Player Re-Identification in Sports VideosWhat this project doesThis project helps track players in sports videos and makes sure each player keeps the same ID throughout the video. This is useful even if they go off-screen and come back. Think of it as giving each player a permanent name tag for the whole game footage.How it works (Simplified)Find Players: We use an AI model (YOLOv11) to spot all the players in each frame.Unique "Look": For each player, we create a digital "fingerprint" based on their appearance (like their jersey color, patterns, and overall shape). We use advanced AI (ResNet18) and other visual details for this.Follow Movement: We predict where players will move next to keep track of them smoothly.Match & Re-identify: We then match new sightings to existing players. If a player was previously tracked but disappeared, we use their "fingerprint" to identify them when they reappear, giving them back their original ID.Getting StartedWhat you needPython 3.8 or newerA decent computer, a graphics card (GPU) helps a lot!At least 8GB of RAMQuick SetupGet the code:Download the zip file of my repository.
Unzip the file and open it in VS Code.
Set up your environment:python -m venv venv
source venv/bin/activate  # Use `venv\Scripts\activate` on Windows
pip install -r requirements.txt
Download AI Model & Sample Video:Download the pre-trained YOLOv11 model from the below link which is used for player detection: Google Drive LinkPlace it in a folder named models/yolov11_player_detection.ptPlace your input video (e.g., 15sec_input_720p.mp4) in a folder named input/ its already placed in the repository do cross-check once in your VS Code.How to Run ItJust open your terminal in the project folder and run:python main.py 
or
python3 main.py 
This will process the video and save the result to output/tracked_video.mp4.What you getA new video showing players with consistent ID numbers and their bounding boxes.A report (.json file) with tracking details and statistics.A graph (.png image) summarizing tracking performance.Troubleshooting & TipsModel Not Found?: Make sure models/yolov11_player_detection.pt is in the right place.IDs Jumping Around?: This means the system is confused. You can try adjusting parameters like --max_disappeared (how long a player can be gone before losing their ID) or --reid_threshold (how similar they need to be to be re-identified).Too Slow?: Processing videos, especially with AI, can take time. A good graphics card helps a lot. You can also try using a lower resolution video or disabling "deep features" in the code for speed.File Structureplayer_reidentification/
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

