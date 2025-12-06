import sys
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import pandas as pd
import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

sys.path.append('core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from natsort import natsorted
from core.utils1.utils import get_network, str2bool, to_cuda
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score,roc_auc_score




DEVICE = 'cuda'
# DEVICE = 'cpu'  # Changed to 'cpu'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, folder_optical_flow_path, imfile1):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)


    # print(folder_optical_flow_path)
    # Use os.path.basename to get filename (works on both Windows and Linux)
    content = os.path.basename(imfile1)
    folder_optical_flow_path = os.path.join(folder_optical_flow_path, content)
    print(folder_optical_flow_path)
    cv2.imwrite(folder_optical_flow_path, flo)


def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(f"Extracting frames from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    print(f"✓ Extracted {frame_count} frames")

    images = glob.glob(os.path.join(output_folder, '*.png')) + \
             glob.glob(os.path.join(output_folder, '*.jpg'))
    images = sorted(images)
    
    return images


# generate optical flow images
def OF_gen(args):
    print("Loading RAFT model...")
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))

    model = model.module
    model.to(DEVICE)
    model.eval()
    print("✓ RAFT model loaded")

    if not os.path.exists(args.folder_optical_flow_path):
        os.makedirs(args.folder_optical_flow_path)
        print(f'{args.folder_optical_flow_path}')

    with torch.no_grad():

        images = video_to_frames(args.path, args.folder_original_path)
        images = natsorted(images)

        print("Generating optical flow...")
        for imfile1, imfile2 in tqdm(zip(images[:-1], images[1:]), total=len(images)-1, desc="Processing optical flow", unit="frame"):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            viz(image1, flow_up,args.folder_optical_flow_path,imfile1)
        
        print("✓ Optical flow generation complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint",default="raft_model/raft-things.pth")
    parser.add_argument('--path', help="dataset for evaluation",default="video/000000.mp4")
    parser.add_argument('--folder_original_path', help="dataset for evaluation_frames",default="frame/000000")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--folder_optical_flow_path',help="the results to save",default="optical_result/000000")
    parser.add_argument(
        "-mop",
        "--model_optical_flow_path",
        type=str,
        default="checkpoints/optical.pth",
    )
    parser.add_argument(
        "-mor",
        "--model_original_path",
        type=str,
        default="checkpoints/original.pth",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument("--use_cpu", action="store_true", help="uses gpu by default, turn on to use cpu")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--aug_norm", type=str2bool, default=True)
    args = parser.parse_args()

    print("=" * 60)
    print("AIGVDet - AI-Generated Video Detection")
    print("=" * 60)
    print(f"Input video: {args.path}")
    print(f"Using device: {'GPU' if not args.use_cpu else 'CPU'}")
    print("=" * 60)

    OF_gen(args)

    print("\nLoading detection models...")
    model_op = get_network(args.arch)
    state_dict = torch.load(args.model_optical_flow_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model_op.load_state_dict(state_dict)
    model_op.eval()
    if not args.use_cpu:
        model_op.cuda()
    print("✓ Optical flow model loaded")

    model_or = get_network(args.arch)
    state_dict = torch.load(args.model_original_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model_or.load_state_dict(state_dict)
    model_or.eval()
    if not args.use_cpu:
        model_or.cuda()
    print("✓ RGB frame model loaded")


    trans = transforms.Compose(
        (
            transforms.CenterCrop((448,448)),
            transforms.ToTensor(),
        )
    )

    print("\n" + "=" * 60)
    print("Running detection...")
    print("=" * 60)

    # optical_subfolder_path = args.folder_optical_flow_path
    # original_subfolder_path = args.folder_original_path


    original_subsubfolder_path = args.folder_original_path
    optical_subsubfolder_path = args.folder_optical_flow_path
                    
    #RGB frame detection
    print("\nAnalyzing RGB frames...")
    original_file_list = sorted(glob.glob(os.path.join(original_subsubfolder_path, "*.jpg")) + glob.glob(os.path.join(original_subsubfolder_path, "*.png"))+glob.glob(os.path.join(original_subsubfolder_path, "*.JPEG")))
    original_prob_sum=0
    for img_path in tqdm(original_file_list, desc="RGB detection", unit="frame"):
                        
        img = Image.open(img_path).convert("RGB")
        img = trans(img)
        if args.aug_norm:
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        in_tens = img.unsqueeze(0)
        if not args.use_cpu:
            in_tens = in_tens.cuda()
                        
        with torch.no_grad():
            prob = model_or(in_tens).sigmoid().item()
            original_prob_sum+=prob
                            
            # print(f"original_path: {img_path}, original_pro: {prob}" )
                        
                        
    original_predict=original_prob_sum/len(original_file_list)
    print(f"✓ RGB detection probability: {original_predict:.4f}")
                    
    #optical flow detection
    print("\nAnalyzing optical flow...")
    optical_file_list = sorted(glob.glob(os.path.join(optical_subsubfolder_path, "*.jpg")) + glob.glob(os.path.join(optical_subsubfolder_path, "*.png"))+glob.glob(os.path.join(optical_subsubfolder_path, "*.JPEG")))
    optical_prob_sum=0
    for img_path in tqdm(optical_file_list, desc="Optical flow detection", unit="frame"):
                        
        img = Image.open(img_path).convert("RGB")
        img = trans(img)
        if args.aug_norm:
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        in_tens = img.unsqueeze(0)
        if not args.use_cpu:
            in_tens = in_tens.cuda()

        with torch.no_grad():
            prob = model_op(in_tens).sigmoid().item()
            optical_prob_sum+=prob

                    
    optical_predict=optical_prob_sum/len(optical_file_list)
    print(f"✓ Optical flow detection probability: {optical_predict:.4f}")
                    
    predict=original_predict*0.5+optical_predict*0.5
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Combined probability: {predict:.4f}")
    print(f"Threshold: {args.threshold}")
    if predict<args.threshold:
        print("🎬 Result: REAL VIDEO")
    else:
        print("🤖 Result: FAKE VIDEO (AI-Generated)")
    print("=" * 60)
