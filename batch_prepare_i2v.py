"""
Batch Data Preparation Script for I2V Models
Processes all *_mp4 folders in data/I2V
"""
import os
import glob
import cv2
import numpy as np
import torch
import sys
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Add core to path for RAFT
sys.path.append('core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configuration
SOURCE_ROOT = "data/I2V"
RGB_OUTPUT_ROOT = "data/test/original/I2V"
FLOW_OUTPUT_ROOT = "data/test/I2V"
RAFT_MODEL = "raft_model/raft-things.pth"
MAX_VIDEOS = 200  # Limit to 200 videos per dataset

def load_image(imfile):
    """Load image and convert to tensor"""
    img = Image.open(imfile)
    
    # Resize if too large to prevent OOM
    max_dim = 1024
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.BILINEAR)
        
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def save_flow(img, flo, output_path):
    """Save optical flow as image"""
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)
    cv2.imwrite(output_path, flo)

def extract_frames(video_path, output_folder, max_frames=95):
    """Extract frames from video (limit to max_frames)"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if already extracted
    existing = glob.glob(os.path.join(output_folder, "*.png"))
    if len(existing) > 0:
        return sorted(existing)[:max_frames]  # Return only up to max_frames
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = os.path.join(output_folder, f"{frame_count:08d}.png")
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        frame_count += 1
    
    cap.release()
    return frames

def generate_optical_flow(model, frames, output_dir):
    """Generate optical flow for frame sequence"""
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (imfile1, imfile2) in enumerate(tqdm(zip(frames[:-1], frames[1:]), 
                                                     total=len(frames)-1, 
                                                     desc="  Generating flow")):
            flow_path = os.path.join(output_dir, f"{i:08d}.png")
            
            if os.path.exists(flow_path):
                continue
                
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            save_flow(image1, flow_up, flow_path)

def process_dataset(dataset_name, source_dir, model):
    """Process a single I2V dataset"""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")
    
    # Get all videos
    videos = glob.glob(os.path.join(source_dir, "*.mp4")) + \
             glob.glob(os.path.join(source_dir, "*.avi")) + \
             glob.glob(os.path.join(source_dir, "*.mov"))
    
    # Apply limit
    if len(videos) > MAX_VIDEOS:
        videos = videos[:MAX_VIDEOS]
        print(f"Limited to {MAX_VIDEOS} videos")
    
    print(f"Found {len(videos)} videos to process")
    
    # Output directories
    rgb_base = os.path.join(RGB_OUTPUT_ROOT, dataset_name, "1_fake")
    flow_base = os.path.join(FLOW_OUTPUT_ROOT, dataset_name, "1_fake")
    
    # Process each video
    for idx, video_path in enumerate(videos, 1):
        video_name = Path(video_path).stem
        print(f"\n[{idx}/{len(videos)}] {video_name}")
        
        rgb_out = os.path.join(rgb_base, video_name)
        flow_out = os.path.join(flow_base, video_name)
        
        # Extract frames
        print("  Extracting frames...")
        frames = extract_frames(video_path, rgb_out)
        
        if len(frames) < 2:
            print("  ⚠ Skipping (too few frames)")
            continue
        
        # Generate optical flow
        generate_optical_flow(model, frames, flow_out)
        
    print(f"\n✅ Completed {dataset_name}")

def main():
    # Find all I2V dataset folders
    source_folders = glob.glob(os.path.join(SOURCE_ROOT, "*_mp4"))
    
    if not source_folders:
        print(f"❌ No *_mp4 folders found in {SOURCE_ROOT}")
        return
    
    print("="*60)
    print("BATCH I2V DATA PREPARATION")
    print("="*60)
    print(f"\nFound {len(source_folders)} dataset(s):")
    for folder in source_folders:
        dataset_name = os.path.basename(folder).replace("_mp4", "")
        print(f"  📁 {dataset_name}")
    
    print(f"\nConfiguration:")
    print(f"  • Max videos per dataset: {MAX_VIDEOS}")
    print(f"  • RAFT model: {RAFT_MODEL}")
    print("="*60)
    
    # Load RAFT model
    print("\nLoading RAFT model...")
    import argparse
    raft_args = argparse.Namespace(
        model=RAFT_MODEL,
        small=False,
        mixed_precision=False,
        alternate_corr=False,
        dropout=0
    )
    
    model = torch.nn.DataParallel(RAFT(raft_args))
    model.load_state_dict(torch.load(RAFT_MODEL, map_location=torch.device(DEVICE)))
    model = model.module
    model.to(DEVICE)
    model.eval()
    print("✓ RAFT model loaded")
    
    # Process each dataset
    for idx, source_dir in enumerate(source_folders, 1):
        dataset_name = os.path.basename(source_dir).replace("_mp4", "")
        
        try:
            process_dataset(dataset_name, source_dir, model)
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            response = input("Continue? (Y/n): ").strip().lower()
            if response == 'n':
                break
    
    print("\n" + "="*60)
    print("✅ BATCH PROCESSING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
