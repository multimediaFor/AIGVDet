import streamlit as st
import sys
import os
import time
import torch
import numpy as np
import cv2
import glob
import argparse
import tempfile
import shutil
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from natsort import natsorted

# Add core to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'core'))

try:
    from core.raft import RAFT
    from core.utils import flow_viz
    from core.utils.utils import InputPadder
    from core.utils1.utils import get_network
except ImportError as e:
    st.error(f"Error importing core modules: {e}. Please ensure you are in the AIGVDet root directory.")

st.set_page_config(page_title="AIGVDet GUI", layout="wide")

st.title("AIGVDet: AI-Generated Video Detection")

# Tabs
tab1, tab2 = st.tabs(["1. Extract (RGB & Optical Flow)", "2. Detection (Architecture)"])

# Global Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def save_vis(img, flo, folder_optical_flow_path, imfile1):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    
    # We only save the flow image as per demo.py logic (it saves 'flo')
    # demo.py: cv2.imwrite(folder_optical_flow_path, flo)
    
    content = os.path.basename(imfile1)
    save_path = os.path.join(folder_optical_flow_path, content)
    
    # cv2 expects BGR, flow_viz returns RGB likely? 
    # flow_viz.flow_to_image returns RGB. cv2.imwrite expects BGR.
    # demo.py uses cv2.imwrite(..., flo). 
    # Let's check flow_viz.flow_to_image implementation if possible, but assuming demo.py works, we follow it.
    # Actually demo.py does: `flo = flow_viz.flow_to_image(flo)` then `cv2.imwrite(..., flo)`.
    # If flow_viz returns RGB, cv2 saves it as BGR (swapping channels), so colors might be inverted if not handled.
    # But we will stick to demo.py logic.
    
    # Convert RGB to BGR for cv2
    flo_bgr = cv2.cvtColor(flo, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, flo_bgr)

def video_to_frames(video_path, output_folder, progress_bar=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        
        if progress_bar:
            progress_bar.progress(min(frame_count / total_frames, 1.0), text=f"Extracting frame {frame_count}/{total_frames}")
            
    cap.release()
    return sorted(glob.glob(os.path.join(output_folder, '*.png')))

# --- TAB 1: EXTRACTION ---
with tab1:
    st.header("Extract RGB Frames and Optical Flow")
    
    # Model Upload
    raft_model_file = st.file_uploader("Upload RAFT Model (raft.pth)", type=['pth'], key="raft_uploader")
    
    # Check file size (500MB limit)
    if raft_model_file:
        file_size_mb = raft_model_file.size / (1024 * 1024)
        if file_size_mb > 500:
            st.error(f"File size ({file_size_mb:.1f} MB) exceeds 500MB limit. Please upload a smaller model.")
            raft_model_file = None
        else:
            st.success(f"Model size: {file_size_mb:.1f} MB")
    
    # Video Upload (Batch or Solo)
    uploaded_videos = st.file_uploader("Upload Video(s)", type=['mp4', 'avi', 'mov', 'mkv'], accept_multiple_files=True, key="video_uploader")
    
    # Output Directory
    output_root = st.text_input("Output Directory Root", value="output_data")
    
    if st.button("Start Extraction", key="extract_btn"):
        extraction_start_time = time.time()
        
        if not raft_model_file:
            st.error("Please upload the RAFT model.")
        elif not uploaded_videos:
            st.error("Please upload at least one video.")
        else:
            # Save RAFT model to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_raft:
                tmp_raft.write(raft_model_file.read())
                raft_model_path = tmp_raft.name
            
            st.info(f"Loaded RAFT model. Using device: {DEVICE}")
            
            # Load RAFT
            try:
                # Args object for RAFT - needs to support 'in' operator
                class Args:
                    def __init__(self):
                        self.model = raft_model_path
                        self.small = False
                        self.mixed_precision = False
                        self.alternate_corr = False
                    
                    def __contains__(self, key):
                        return hasattr(self, key)
                
                args = Args()
                model = torch.nn.DataParallel(RAFT(args))
                model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))
                model = model.module
                model.to(DEVICE)
                model.eval()
                
                st.success("RAFT Model Loaded Successfully!")
                
                # Process Videos
                total_videos = len(uploaded_videos)
                
                for i, video_file in enumerate(uploaded_videos):
                    video_start_time = time.time()
                    video_name = video_file.name
                    st.subheader(f"Processing: {video_name} ({i+1}/{total_videos})")
                    
                    # Save video to temp
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_name)[1]) as tmp_vid:
                        tmp_vid.write(video_file.read())
                        video_path = tmp_vid.name
                    
                    # Define output paths
                    base_name = os.path.splitext(video_name)[0]
                    frame_output_dir = os.path.join(output_root, "frames", base_name)
                    flow_output_dir = os.path.join(output_root, "optical_flow", base_name)
                    
                    # 1. Extract Frames
                    st.write("Extracting frames...")
                    frame_start = time.time()
                    p_bar = st.progress(0)
                    images = video_to_frames(video_path, frame_output_dir, p_bar)
                    frame_duration = time.time() - frame_start
                    st.write(f"✓ Extracted {len(images)} frames to `{frame_output_dir}` in {frame_duration:.2f}s")
                    
                    # 2. Generate Optical Flow
                    if not os.path.exists(flow_output_dir):
                        os.makedirs(flow_output_dir)
                    
                    st.write("Generating Optical Flow...")
                    flow_start = time.time()
                    images = natsorted(images)
                    flow_p_bar = st.progress(0)
                    
                    with torch.no_grad():
                        for idx, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
                            image1 = load_image(imfile1)
                            image2 = load_image(imfile2)
                            
                            padder = InputPadder(image1.shape)
                            image1, image2 = padder.pad(image1, image2)
                            
                            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                            
                            save_vis(image1, flow_up, flow_output_dir, imfile1)
                            
                            flow_p_bar.progress((idx + 1) / (len(images) - 1))
                    
                    flow_duration = time.time() - flow_start
                    st.write(f"✓ Optical Flow saved to `{flow_output_dir}` in {flow_duration:.2f}s")
                    
                    video_duration = time.time() - video_start_time
                    st.info(f"**Video processed in {video_duration:.2f} seconds**")
                    
                    # Cleanup temp video
                    os.remove(video_path)
                
                total_extraction_time = time.time() - extraction_start_time
                st.success(f"✓ All videos processed! Total time: {total_extraction_time:.2f} seconds")
                # Cleanup temp model
                os.remove(raft_model_path)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                import traceback
                st.code(traceback.format_exc())

# --- TAB 2: DETECTION ---
with tab2:
    st.header("Run Detection (AIGVDet)")
    
    col1, col2 = st.columns(2)
    with col1:
        optical_model_file = st.file_uploader("Upload Optical Flow Model (optical.pth)", type=['pth'], key="opt_uploader")
        if optical_model_file:
            opt_size_mb = optical_model_file.size / (1024 * 1024)
            if opt_size_mb > 500:
                st.error(f"File size ({opt_size_mb:.1f} MB) exceeds 500MB limit.")
                optical_model_file = None
            else:
                st.success(f"Optical model: {opt_size_mb:.1f} MB")
    
    with col2:
        original_model_file = st.file_uploader("Upload RGB Model (original.pth)", type=['pth'], key="orig_uploader")
        if original_model_file:
            orig_size_mb = original_model_file.size / (1024 * 1024)
            if orig_size_mb > 500:
                st.error(f"File size ({orig_size_mb:.1f} MB) exceeds 500MB limit.")
                original_model_file = None
            else:
                st.success(f"RGB model: {orig_size_mb:.1f} MB")
        
    # Input for processed data path
    # Default to the output of Tab 1 if available
    target_dir = st.text_input("Path to Processed Data (Root folder containing 'frames' and 'optical_flow')", value="output_data")
    
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
    
    if st.button("Run Detection", key="detect_btn"):
        if not optical_model_file or not original_model_file:
            st.error("Please upload both Optical Flow and RGB models.")
        elif not os.path.exists(target_dir):
            st.error(f"Directory `{target_dir}` does not exist.")
        else:
            start_time = time.time()
            
            # Save models to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_opt:
                tmp_opt.write(optical_model_file.read())
                opt_model_path = tmp_opt.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_orig:
                tmp_orig.write(original_model_file.read())
                orig_model_path = tmp_orig.name
                
            try:
                st.info("Loading models...")
                
                # Load Models
                # Assuming ResNet50 as per demo.py default
                model_op = get_network("resnet50")
                state_dict_op = torch.load(opt_model_path, map_location="cpu")
                if "model" in state_dict_op:
                    state_dict_op = state_dict_op["model"]
                model_op.load_state_dict(state_dict_op)
                model_op.eval()
                if DEVICE == 'cuda':
                    model_op.cuda()
                
                model_or = get_network("resnet50")
                state_dict_or = torch.load(orig_model_path, map_location="cpu")
                if "model" in state_dict_or:
                    state_dict_or = state_dict_or["model"]
                model_or.load_state_dict(state_dict_or)
                model_or.eval()
                if DEVICE == 'cuda':
                    model_or.cuda()
                
                load_duration = time.time() - start_time
                st.write(f"Models loaded in {load_duration:.2f} seconds.")
                
                # Find subfolders in frames
                frames_root = os.path.join(target_dir, "frames")
                flow_root = os.path.join(target_dir, "optical_flow")
                
                if not os.path.exists(frames_root):
                    st.error(f"Could not find `frames` folder in {target_dir}")
                    st.stop()
                
                # Get list of video folders
                video_folders = [f for f in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, f))]
                
                if not video_folders:
                    st.warning("No subfolders found in `frames` directory.")
                
                # Transforms
                trans = transforms.Compose((
                    transforms.CenterCrop((448,448)),
                    transforms.ToTensor(),
                ))
                
                results = []
                
                for vid_folder in video_folders:
                    video_detect_start = time.time()
                    st.subheader(f"Analyzing: {vid_folder}")
                    
                    rgb_path = os.path.join(frames_root, vid_folder)
                    opt_path = os.path.join(flow_root, vid_folder)
                    
                    if not os.path.exists(opt_path):
                        st.warning(f"No optical flow found for {vid_folder}, skipping.")
                        continue
                        
                    # RGB Detection
                    rgb_files = sorted(glob.glob(os.path.join(rgb_path, "*.jpg")) + 
                                     glob.glob(os.path.join(rgb_path, "*.png")) + 
                                     glob.glob(os.path.join(rgb_path, "*.JPEG")))
                    
                    rgb_prob_sum = 0
                    rgb_bar = st.progress(0, text="RGB Detection")
                    rgb_start = time.time()
                    
                    for i, img_path in enumerate(rgb_files):
                        img = Image.open(img_path).convert("RGB")
                        img = trans(img)
                        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        in_tens = img.unsqueeze(0).to(DEVICE)
                        
                        with torch.no_grad():
                            prob = model_or(in_tens).sigmoid().item()
                            rgb_prob_sum += prob
                        
                        rgb_bar.progress((i + 1) / len(rgb_files))
                    
                    rgb_duration = time.time() - rgb_start
                    rgb_score = rgb_prob_sum / len(rgb_files) if rgb_files else 0
                    st.write(f"✓ RGB Score: {rgb_score:.4f} ({len(rgb_files)} frames in {rgb_duration:.2f}s)")
                    
                    # Optical Flow Detection
                    opt_files = sorted(glob.glob(os.path.join(opt_path, "*.jpg")) + 
                                     glob.glob(os.path.join(opt_path, "*.png")) + 
                                     glob.glob(os.path.join(opt_path, "*.JPEG")))
                    
                    opt_prob_sum = 0
                    opt_bar = st.progress(0, text="Optical Flow Detection")
                    opt_start = time.time()
                    
                    for i, img_path in enumerate(opt_files):
                        img = Image.open(img_path).convert("RGB")
                        img = trans(img)
                        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        in_tens = img.unsqueeze(0).to(DEVICE)
                        
                        with torch.no_grad():
                            prob = model_op(in_tens).sigmoid().item()
                            opt_prob_sum += prob
                            
                        opt_bar.progress((i + 1) / len(opt_files))
                    
                    opt_duration = time.time() - opt_start    
                    opt_score = opt_prob_sum / len(opt_files) if opt_files else 0
                    st.write(f"✓ Optical Flow Score: {opt_score:.4f} ({len(opt_files)} frames in {opt_duration:.2f}s)")
                    
                    # Final Decision
                    final_score = (rgb_score * 0.5) + (opt_score * 0.5)
                    decision = "FAKE VIDEO (AI-Generated)" if final_score >= threshold else "REAL VIDEO"
                    color = "red" if final_score >= threshold else "green"
                    
                    video_detect_duration = time.time() - video_detect_start
                    
                    st.markdown(f"### Result: :{color}[{decision}]")
                    st.write(f"**Combined Probability:** {final_score:.4f}")
                    st.write(f"**Detection Time:** {video_detect_duration:.2f}s")
                    st.divider()
                    
                    results.append({
                        "Video": vid_folder,
                        "RGB Score": rgb_score,
                        "Optical Score": opt_score,
                        "Final Score": final_score,
                        "Decision": decision
                    })
                
                # Summary Table
                if results:
                    st.subheader("Batch Summary")
                    st.dataframe(results)
                    
                total_duration = time.time() - start_time
                st.success(f"Total Processing Time: {total_duration:.2f} seconds")
                
                # Cleanup
                os.remove(opt_model_path)
                os.remove(orig_model_path)
                
            except Exception as e:
                st.error(f"An error occurred during detection: {e}")
                import traceback
                st.code(traceback.format_exc())
