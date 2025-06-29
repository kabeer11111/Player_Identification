import argparse
import cv2
import json
import os
from pathlib import Path
import time
from tqdm import tqdm

from player_tracker import PlayerTracker
from utils import VideoProcessor, MetricsCalculator, Visualizer, NumpyJSONEncoder, convert_numpy_types


def main():
    parser = argparse.ArgumentParser(description='Player Re-Identification System')
    parser.add_argument('--input', type=str, default='input/15sec_input_720p.mp4',
                       help='Input video path')
    parser.add_argument('--output', type=str, default='output/tracked_video.mp4',
                       help='Output video path')
    parser.add_argument('--model', type=str, default='models/yolov11_player_detection.pt',
                       help='YOLO model path')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--max_disappeared', type=int, default=60,
                       help='Max frames a player can disappear before losing ID')
    parser.add_argument('--reid_threshold', type=float, default=0.7,
                       help='Re-identification similarity threshold')
    
    args = parser.parse_args()
    
    # Create output directories
    Path('output').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # Initialize video processor
    video_processor = VideoProcessor(args.input, args.output)
    
    # Initialize player tracker
    tracker = PlayerTracker(
        model_path=args.model,
        confidence=args.confidence,
        max_disappeared=args.max_disappeared,
        reid_threshold=args.reid_threshold
    )
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator()
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    print(f"Processing video: {args.input}")
    print(f"Output will be saved to: {args.output}")
    
    # Process video
    process_video(video_processor, tracker, metrics_calc, visualizer)
    
    # Save results
    save_results(tracker, metrics_calc, args.output)
    
    print("Processing complete!")


def process_video(video_processor, tracker, metrics_calc, visualizer):
    frame_count = 0
    start_time = time.time()
    
    # Get video properties
    total_frames = int(video_processor.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_processor.cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames at {fps:.2f} FPS")
    
    # Progress bar
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while True:
        ret, frame = video_processor.cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Track players in current frame
        detections = tracker.track_frame(frame, frame_count)
        
        # Update metrics
        metrics_calc.update_frame_metrics(detections, frame_count)
        
        # Visualize results
        annotated_frame = visualizer.draw_tracking_results(frame, detections)
        
        # Write frame to output video
        video_processor.write_frame(annotated_frame)
        
        # Update progress
        pbar.update(1)
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_current = frame_count / elapsed
            pbar.set_postfix({
                'FPS': f'{fps_current:.1f}',
                'Active_IDs': len(tracker.active_tracks),
                'Total_IDs': tracker.next_id - 1
            })
    
    pbar.close()
    video_processor.release()
    
    # Final metrics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    
    print(f"\nProcessing Summary:")
    print(f"Total frames processed: {frame_count}")
    print(f"Processing time: {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total unique players tracked: {tracker.next_id - 1}")
    print(f"Active tracks at end: {len(tracker.active_tracks)}")


def save_results(tracker, metrics_calc, output_path):
    # Prepare results data
    results = {
        'tracking_summary': {
            'total_unique_players': tracker.next_id - 1,
            'total_detections': metrics_calc.total_detections,
            'total_frames': metrics_calc.frame_count,
            'id_switches': metrics_calc.id_switches,
            'reid_successes': tracker.reid_successes,
            'reid_attempts': tracker.reid_attempts
        },
        'tracking_data': tracker.get_tracking_history(),
        'metrics': metrics_calc.get_metrics()
    }
    
    # Convert NumPy types to native Python types
    results = convert_numpy_types(results)
    
    # Save to JSON
    results_path = output_path.replace('.mp4', '_results.json')
    try:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyJSONEncoder)
        print(f"Results saved to: {results_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
        # Fallback: save without problematic data
        simplified_results = {
            'tracking_summary': results['tracking_summary'],
            'metrics': results['metrics']
        }
        with open(results_path, 'w') as f:
            json.dump(simplified_results, f, indent=2, cls=NumpyJSONEncoder)
        print(f"Simplified results saved to: {results_path}")
    
    # Save metrics visualization
    try:
        metrics_calc.save_visualization('results/tracking_metrics.png')
        print("Metrics visualization saved to: results/tracking_metrics.png")
    except Exception as e:
        print(f"Error saving metrics visualization: {e}")
    
    # Print final statistics
    print_final_statistics(results)


def print_final_statistics(results):
    
    summary = results['tracking_summary']
    
    print("\n" + "="*50)
    print("FINAL TRACKING STATISTICS")
    print("="*50)
    print(f"Total unique players: {summary['total_unique_players']}")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Total frames: {summary['total_frames']}")
    print(f"Average detections per frame: {summary['total_detections']/summary['total_frames']:.2f}")
    
    if summary['reid_attempts'] > 0:
        reid_success_rate = summary['reid_successes'] / summary['reid_attempts'] * 100
        print(f"Re-identification success rate: {reid_success_rate:.1f}%")
    else:
        print("No re-identification attempts")
    
    print(f"ID switches: {summary['id_switches']}")
    print("="*50)


if __name__ == "__main__":
    main()