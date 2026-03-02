import cv2
import argparse
from pathlib import Path
import json
from datetime import timedelta

def detect_cuts_batch(video_path, threshold=30, min_scene_len=15):
    """Batch-friendly cut detection."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, f"Error: Could not open {video_path}"
    
    prev_frame = None
    cuts = []
    frame_count = 0
    last_cut_frame = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_frame is not None:
            frame_delta = cv2.absdiff(prev_frame, gray)
            diff_score = frame_delta.mean()
            
            if diff_score > threshold and (frame_count - last_cut_frame) > min_scene_len:
                timecode = frame_count / fps
                hours = int(timecode // 3600)
                minutes = int((timecode % 3600) // 60)
                seconds = int(timecode % 60)
                frames = int((timecode * fps) % fps)
                
                cuts.append({
                    "frame": frame_count,
                    "timecode": f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}",
                    "timestamp": str(timedelta(seconds=int(timecode)))
                })
                last_cut_frame = frame_count
                
        prev_frame = gray
        frame_count += 1
        
    cap.release()
    
    return {
        "filename": video_path.name,
        "total_cuts": len(cuts),
        "fps": fps,
        "cuts": cuts
    }, None

def process_directory(input_dir, output_file="batch_results.json"):
    """Process all videos in a directory."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: {input_dir} is not a valid directory")
        return
    
    video_extensions = {'.mp4', '.mov', '.avi', '.mxf', '.mkv'}
    results = []
    errors = []
    
    print(f"🔍 Scanning {input_path} for video files...")
    video_files = [f for f in input_path.iterdir() if f.suffix.lower() in video_extensions]
    
    if not video_files:
        print("No video files found!")
        return
        
    print(f"📹 Found {len(video_files)} video files\n")
    
    for i, video_file in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Processing {video_file.name}...")
        result, error = detect_cuts_batch(video_file)
        
        if error:
            errors.append(error)
            print(f"   ❌ {error}")
        else:
            results.append(result)
            print(f"   ✅ Found {result['total_cuts']} cuts")
    
    # Save results
    output_data = {
        "summary": {
            "total_files": len(video_files),
            "successful": len(results),
            "failed": len(errors)
        },
        "results": results,
        "errors": errors
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n📊 Batch processing complete!")
    print(f"✅ Success: {len(results)} | ❌ Failed: {len(errors)}")
    print(f"📄 Results saved to: {output_file}")
    
    # Print summary table
    print("\n📈 Summary:")
    print("-" * 60)
    for r in results:
        print(f"{r['filename']:<40} {r['total_cuts']:>3} cuts")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch video cut detection")
    parser.add_argument("input_dir", help="Directory containing video files")
    parser.add_argument("-o", "--output", default="batch_results.json", help="Output JSON file")
    parser.add_argument("-t", "--threshold", type=int, default=30, help="Cut detection sensitivity")
    
    args = parser.parse_args()
    process_directory(args.input_dir, args.output)
