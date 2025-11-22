import cv2
import numpy as np
import argparse
import os
import math
from hoopDetectorAlgorithmFloodfill import detect, HoopConfig

def get_sim_k(width, height, hfov_deg=80.0):
    """Generate a generic camera matrix for testing."""
    cx = width / 2.0
    cy = height / 2.0
    fx = cx / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx 
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

def run_video(video_path, output_dir):
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    
    # Create Output Directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Video Properties
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup Algorithm Config
    # TUNING PARAMETERS HERE
    cfg = HoopConfig()
    cfg.hue_min = 0     # Adjust for red/orange
    cfg.hue_max = 15
    cfg.sat_min = 100
    cfg.val_min = 100
    cfg.contour_thickness = 10 # How much to erode the hole (remove peanut bridges)
    cfg.ring_radius = 0.10     # Radius in meters
    
    # Mock Camera Matrix
    K = get_sim_k(W, H)
    
    # Video Writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_main = cv2.VideoWriter(os.path.join(output_dir, "result.mp4"), fourcc, fps, (W, H))
    out_mask = cv2.VideoWriter(os.path.join(output_dir, "mask_holes.mp4"), fourcc, fps, (W, H))

    print(f"Processing {video_path}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        pose, debug_frames = detect(frame, K, cfg)
        
        # Draw additional 3D info if pose found
        res_frame = debug_frames['result']
        if pose:
            text = f"POS: [{pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f}]"
            cv2.putText(res_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        out_main.write(res_frame)
        out_mask.write(debug_frames['mask_holes'])
        
        cv2.imshow('Preview', cv2.resize(res_frame, (W//2, H//2)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out_main.release()
    out_mask.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help="Path to input video")
    parser.add_argument('--out', type=str, default="SimpleAlgTestingOnRealSim/run_new_01", help="Output folder")
    args = parser.parse_args()
    
    run_video(args.input, args.out)