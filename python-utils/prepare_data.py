import sys
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from natsort import natsorted

# Add core to path for imports
sys.path.append('core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(imfile):
    """
    Load an image, convert to tensor, and move to the appropriate device.
    """
    img = np.array(Image.open(imfile)).astype(np.uint8)
        
    return img[None].to(DEVICE)

def save_flow(img, flo, output_path):
    """
    Save optical flow as a color image.
    """
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # Convert flow to RGB
    flo = flow_viz.flow_to_image(flo)
    cv2.imwrite(output_path, flo)

def video_to_frames(video_path, output_folder, max_frames=95):
    """
    Extract frames from a video and save them as images in the output folder.
    Limits to max_frames (default 95) to match paper methodology.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Check if frames already exist to skip
    existing_frames = glob.glob(os.path.join(output_folder, "*.png"))
    if len(existing_frames) > 0:
        return sorted(existing_frames)[:max_frames]

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    
    # Return sorted list of extracted frame files
    images = glob.glob(os.path.join(output_folder, '*.png')) + \
             glob.glob(os.path.join(output_folder, '*.jpg'))
    return sorted(images)[:max_frames]

def process_dataset(args):
    """
    Process each video in the dataset, extracting frames and computing optical flow.
    """
    # Load RAFT model once
    print(f"Loading RAFT model from {args.model}...")
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))
    model = model.module
    model.to(DEVICE)
    model.eval()
    print("✓ RAFT model loaded")

    # Structure: source_dir / [0_real, 1_fake] / video.mp4
    for label in ["0_real", "1_fake"]:
        source_label_dir = os.path.join(args.source_dir, label)
        if not os.path.exists(source_label_dir):
            print(f"Skipping {label}, directory not found: {source_label_dir}")
            continue
            
        videos = glob.glob(os.path.join(source_label_dir, "*.mp4")) + \
                 glob.glob(os.path.join(source_label_dir, "*.avi")) + \
                 glob.glob(os.path.join(source_label_dir, "*.mov"))
        
        print(f"Found {len(videos)} videos in {label}")
        
        for video_path in tqdm(videos, desc=f"Processing {label}"):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Define output paths for RGB frames and optical flow
            rgb_out_dir = os.path.join(args.output_rgb_dir, label, video_name)
            flow_out_dir = os.path.join(args.output_flow_dir, label, video_name)
            
            if not os.path.exists(flow_out_dir):
                os.makedirs(flow_out_dir)
            
            # 1. Extract Frames
            images = video_to_frames(video_path, rgb_out_dir)
            images = natsorted(images)
            
            if len(images) < 2:
                continue

            # 2. Generate Optical Flow
            # Check if flow already exists for the extracted frames
            existing_flow = glob.glob(os.path.join(flow_out_dir, "*.png"))
            if len(existing_flow) >= len(images) - 1:
                continue

            with torch.no_grad():
                for i, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
                    # Output filename matches input filename
                    flow_filename = os.path.basename(imfile1)
                    flow_output_path = os.path.join(flow_out_dir, flow_filename)
                    
                    if os.path.exists(flow_output_path):
                        continue

                    # Load the consecutive frames
                    image1 = load_image(imfile1)
                    image2 = load_image(imfile2)

                    # Pad images if necessary
                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)

                    # Compute flow with the RAFT model
                    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                    
                    # Save the optical flow
                    save_flow(image1, flow_up, flow_output_path)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', required=True, help="Path to folder containing 0_real/1_fake video folders")
    parser.add_argument('--output_rgb_dir', required=True, help="Output path for RGB frames")
    parser.add_argument('--output_flow_dir', required=True, help="Output path for Optical Flow frames")
    parser.add_argument('--model', default="raft_model/raft-things.pth", help="Path to RAFT model checkpoint")
    
    # RAFT arguments
    parser.add_argument('--small', action='store_true', help='Use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='Use efficient correlation implementation')
    
    args = parser.parse_args()
    
    # Process the dataset with the provided arguments
    process_dataset(args)
