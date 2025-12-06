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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid")

# Add core to path for imports
sys.path.append('core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a GPU.")
DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def save_flow(img, flo, output_path):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # Save only the flow image as per typical requirements, or concatenated if requested.
    # The user's code had: img_flo = np.concatenate([img, flo], axis=0)
    # But usually for training we just want the flow.
    # The user's demo code saved 'flo'.
    cv2.imwrite(output_path, flo)

def video_to_frames(video_path, output_folder, max_frames=95, resize=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Print resolution to help user understand speed
    print(f"  -> Video: {os.path.basename(video_path)} | Resolution: {width}x{height}")

    frame_count = 0
    saved_frames = []
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize if requested
        if resize is not None:
            h, w = frame.shape[:2]
            if min(h, w) > resize:
                scale = resize / min(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame = cv2.resize(frame, (new_w, new_h))

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_filename = os.path.join(output_folder, f"{video_name}_{frame_count+1:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_frames.append(frame_filename)
        frame_count += 1
    
    cap.release()
    return sorted(saved_frames)

def process_videos(args):
    # Verify model path
    if not os.path.exists(args.model):
        # Try fallback for common issue (hyphen vs underscore)
        if "raft-model" in args.model and os.path.exists(args.model.replace("raft-model", "raft_model")):
            args.model = args.model.replace("raft-model", "raft_model")
            print(f"Found model at alternate path: {args.model}")
        elif "raft_model" in args.model and os.path.exists(args.model.replace("raft_model", "raft-model")):
            args.model = args.model.replace("raft_model", "raft-model")
            print(f"Found model at alternate path: {args.model}")
        else:
            print(f"Error: Model file not found at {args.model}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Available directories: {[d for d in os.listdir('.') if os.path.isdir(d)]}")
            raise FileNotFoundError(f"Model file not found: {args.model}")

    # Load model
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))
    model = model.module
    model.to(DEVICE)
    model.eval()

    print(f"Processing on device: {DEVICE}")
    if args.resize:
        print(f"Resizing frames to short edge: {args.resize}px")
    
    # Get list of videos
    videos = glob.glob(os.path.join(args.input_path, '*.mp4')) + \
             glob.glob(os.path.join(args.input_path, '*.avi')) + \
             glob.glob(os.path.join(args.input_path, '*.mov'))
    
    print(f"Found {len(videos)} videos in {args.input_path}")

    for video_path in tqdm(videos):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Use main output directories directly (flat structure)
        video_rgb_dir = args.output_rgb
        video_flow_dir = args.output_flow
        
        if not os.path.exists(video_rgb_dir):
            os.makedirs(video_rgb_dir)
        if not os.path.exists(video_flow_dir):
            os.makedirs(video_flow_dir)

        # 1. Extract RGB frames (max 95)
        # Pass resize argument here
        images = video_to_frames(video_path, video_rgb_dir, max_frames=95, resize=args.resize)
        
        # 2. Compute Optical Flow (max 94)
        if len(images) < 2:
            continue
            
        with torch.no_grad():
            images = natsorted(images)
            
            # Load first image
            image1 = load_image(images[0])
            
            for i in range(len(images) - 1):
                if i >= 94:
                    break
                
                imfile1 = images[i]
                imfile2 = images[i+1]
                
                # Check if flow already exists
                flow_filename = os.path.basename(imfile1)
                flow_output_path = os.path.join(video_flow_dir, flow_filename)
                
                if os.path.exists(flow_output_path):
                    # If skipping, we still need to update image1 for the next iteration if we weren't reloading
                    # But since we are skipping, we might not have loaded image2 yet.
                    # To be safe and simple with skipping logic:
                    image1 = load_image(imfile2) # Prepare for next iter
                    continue

                image2 = load_image(imfile2)

                # Debug: Print shape once to confirm resize
                if i == 0 and video_path == videos[0]:
                    print(f"  -> Model input shape: {image1.shape}")

                padder = InputPadder(image1.shape)
                image1_padded, image2_padded = padder.pad(image1, image2)

                flow_low, flow_up = model(image1_padded, image2_padded, iters=20, test_mode=True)
                
                save_flow(image1, flow_up, flow_output_path)
                
                # Move image2 to image1 for next iteration
                image1 = image2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default="raft-model/raft-things.pth")
    parser.add_argument('--input_path', help="path to videos", default="data/T2V/moonvalley_mp4")
    parser.add_argument('--output_rgb', help="output path for RGB frames", default="data/T2V/moonvalley_rgb")
    parser.add_argument('--output_flow', help="output path for Optical Flow frames", default="data/T2V/moonvalley_flow")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--resize', type=int, default=None, help='Resize smaller edge of frames to this value (e.g. 512) for faster processing')
    
    args = parser.parse_args()
    
    process_videos(args)
