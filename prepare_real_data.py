import argparse
import os
import glob
import shutil
import cv2
import numpy as np
import torch
import torch.nn
from PIL import Image
from tqdm import tqdm
import sys

# Add core to path to import RAFT
sys.path.append('core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from natsort import natsorted

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(imfile):
    img = Image.open(imfile)
    
    # Resize if too large (e.g. > 1024px) to prevent OOM
    max_dim = 1024
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.BILINEAR)
        
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz(img, flo, output_dir, filename):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    
    # Save flow image
    # The filename should match the input frame filename
    save_path = os.path.join(output_dir, os.path.basename(filename))
    cv2.imwrite(save_path, flo)

def generate_optical_flow(model, frames_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get all frames
    images = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')) + 
                   glob.glob(os.path.join(frames_dir, '*.png')))
    images = natsorted(images)
    
    print(f"Generating optical flow for {len(images)} frames...")
    
    with torch.no_grad():
        for imfile1, imfile2 in tqdm(zip(images[:-1], images[1:]), total=len(images)-1, desc="Flow Generation"):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            viz(image1, flow_up, output_dir, imfile1)

def main():
    parser = argparse.ArgumentParser(description="Prepare Real Data for all datasets")
    parser.add_argument("--source", type=str, required=True, help="Path to source folder containing multiple video folders (e.g. 1_real)")
    parser.add_argument("--raft_model", type=str, default="raft_model/raft-things.pth", help="Path to RAFT model checkpoint")
    parser.add_argument("--start_index", type=int, default=1, help="Index to start processing from (1-based)")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"❌ Source folder not found: {args.source}")
        return

    # Find all subdirectories (video folders)
    video_folders = [f.path for f in os.scandir(args.source) if f.is_dir()]
    video_folders = sorted(video_folders)
    
    if not video_folders:
        print(f"❌ No video folders found in {args.source}")
        return

    print(f"Found {len(video_folders)} video folders to process")

    # Load RAFT model
    print("Loading RAFT model...")
    
    raft_args = argparse.Namespace(
        model=args.raft_model,
        small=False,
        mixed_precision=False,
        alternate_corr=False,
        dropout=0
    )
    
    model = torch.nn.DataParallel(RAFT(raft_args))
    
    try:
        model.load_state_dict(torch.load(args.raft_model, map_location=torch.device(DEVICE)))
    except FileNotFoundError:
        print(f"❌ RAFT model not found at {args.raft_model}")
        print("Please download it first using: python download_data.py --skip-data --skip-checkpoints")
        return

    model = model.module
    model.to(DEVICE)
    model.eval()
    print("✓ RAFT model loaded")

    # Target datasets
    datasets = ["moonvalley", "videocraft", "pika", "neverends"]
    
    # Process each video folder
    for idx, video_path in enumerate(video_folders, 1):
        if idx < args.start_index:
            continue
            
        video_name = os.path.basename(video_path)
        print(f"\n{'='*60}")
        print(f"Processing Video {idx}/{len(video_folders)}: {video_name}")
        print(f"{'='*60}")
        
        # 1. Generate Optical Flow ONCE in a temp location
        temp_flow_dir = os.path.join("data", "temp_flow", video_name)
        # Skip if already generated in temp
        if os.path.exists(temp_flow_dir) and len(os.listdir(temp_flow_dir)) > 0:
             print(f"  Using existing flow in temp: {temp_flow_dir}")
        else:
            print(f"  Generating optical flow to temp location: {temp_flow_dir}")
            generate_optical_flow(model, video_path, temp_flow_dir)
        
        # 2. Distribute to all datasets
        print(f"  Distributing to {len(datasets)} datasets...")
        
        for dataset in datasets:
            # Paths
            rgb_dest = os.path.join("data", "test", "original", "T2V", dataset, "0_real", video_name)
            flow_dest = os.path.join("data", "test", "T2V", dataset, "0_real", video_name)
            
            # Create directories
            os.makedirs(rgb_dest, exist_ok=True)
            os.makedirs(flow_dest, exist_ok=True)
            
            # Copy RGB frames
            # print(f"    -> RGB: {dataset}")
            if os.path.exists(rgb_dest):
                shutil.rmtree(rgb_dest)
            shutil.copytree(video_path, rgb_dest)
            
            # Copy Flow frames
            # print(f"    -> Flow: {dataset}")
            if os.path.exists(flow_dest):
                shutil.rmtree(flow_dest)
            shutil.copytree(temp_flow_dir, flow_dest)
        
    print("\n" + "="*80)
    print("✓ REAL DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"Source: {args.source}")
    print(f"Processed {len(video_folders)} videos")
    print(f"Distributed to: {', '.join(datasets)}")
    
    # Clean up temp
    if os.path.exists(os.path.join("data", "temp_flow")):
        shutil.rmtree(os.path.join("data", "temp_flow"))

if __name__ == "__main__":
    main()
