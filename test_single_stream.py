"""
Modified test.py that supports single-stream evaluation AND flat directory structures.
Supports: RGB-only (Sspatial), Optical-only (Soptical), or Fused (AIGVDet)
"""
import argparse
import glob
import os
import pandas as pd
import re

import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from core.utils1.utils import get_network, str2bool, to_cuda
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

def get_video_name_from_filename(filename):
    # Assumes format: video_name_XXXXX.png
    # We split by underscore and take everything except the last part (frame number)
    parts = os.path.basename(filename).rsplit('_', 1)
    if len(parts) > 1:
        return parts[0]
    return "unknown_video"

if __name__=="__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-fop", "--folder_optical_flow_path", default="data/test/T2V/videocraft", type=str)
    parser.add_argument("-for", "--folder_original_path", default="data/test/original/T2V/videocraft", type=str)
    parser.add_argument("-mop", "--model_optical_flow_path", type=str, default="checkpoints/optical.pth")
    parser.add_argument("-mor", "--model_original_path", type=str, default="checkpoints/original.pth")
    parser.add_argument("--eval_mode", type=str, choices=["fused", "rgb_only", "optical_only"], default="fused")
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("-e", "--excel_path", type=str, default="data/results/result.csv")
    parser.add_argument("-ef", "--excel_frame_path", type=str, default="data/results/frame_result.csv")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--aug_norm", type=str2bool, default=True)
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos per class (optional)")

    args = parser.parse_args()

    # Load models
    if args.eval_mode in ["fused", "optical_only"]:
        print(f"Loading optical flow model: {args.model_optical_flow_path}")
        model_op = get_network(args.arch)
        state_dict = torch.load(args.model_optical_flow_path, map_location="cpu")
        if "model" in state_dict: state_dict = state_dict["model"]
        model_op.load_state_dict(state_dict)
        model_op.eval()
        if not args.use_cpu: model_op.cuda()
    else: model_op = None
        
    if args.eval_mode in ["fused", "rgb_only"]:
        print(f"Loading RGB model: {args.model_original_path}")
        model_or = get_network(args.arch)
        state_dict = torch.load(args.model_original_path, map_location="cpu")
        if "model" in state_dict: state_dict = state_dict["model"]
        model_or.load_state_dict(state_dict)
        model_or.eval()
        if not args.use_cpu: model_or.cuda()
    else: model_or = None
    
    if args.no_crop:
        trans = transforms.Compose((transforms.ToTensor(),))
    else:
        trans = transforms.Compose((transforms.CenterCrop((448,448)), transforms.ToTensor(),))

    print("*" * 50)
    print(f"Evaluation Mode: {args.eval_mode}")
    print("*" * 50)

    flag=0
    p=0; n=0; tp=0; tn=0
    y_true=[]; y_pred=[]
    y_pred_original=[]; y_pred_optical=[]

    df = pd.DataFrame(columns=['name', 'pro','flag','optical_pro','original_pro'])
    df1 = pd.DataFrame(columns=['original_path', 'original_pro','optical_path','optical_pro','flag'])
    index1=0
    
    # Check if standard structure (0_real/1_fake) exists
    has_standard_structure = os.path.exists(os.path.join(args.folder_original_path, "1_fake")) or \
                             os.path.exists(os.path.join(args.folder_optical_flow_path, "1_fake"))

    if has_standard_structure:
        print("Detected standard folder structure (0_real/1_fake)")
        subfolders = ["0_real", "1_fake"]
    else:
        print("Detected FLAT folder structure (treating all as 1_fake)")
        subfolders = ["flat_fake"]

    for subfolder_name in subfolders:
        if subfolder_name == "0_real":
            flag = 0
            current_label_path = "0_real"
        elif subfolder_name == "1_fake":
            flag = 1
            current_label_path = "1_fake"
        else:
            flag = 1 # Flat structure assumed to be fake/generated videos
            current_label_path = "" # Root dir

        optical_subfolder_path = os.path.join(args.folder_optical_flow_path, current_label_path)
        original_subfolder_path = os.path.join(args.folder_original_path, current_label_path)
        
        # Get list of videos
        # In flat structure, we need to group images by video prefix
        video_groups = {}
        
        if args.eval_mode != "optical_only":
            # Scan RGB folder
            if os.path.exists(original_subfolder_path):
                files = sorted(glob.glob(os.path.join(original_subfolder_path, "*.png")) + 
                             glob.glob(os.path.join(original_subfolder_path, "*.jpg")) +
                             glob.glob(os.path.join(original_subfolder_path, "*.JPEG")))
                
                # If files found directly, it's flat structure
                if len(files) > 0 and not os.path.isdir(files[0]):
                    for f in files:
                        vname = get_video_name_from_filename(f)
                        if vname not in video_groups: video_groups[vname] = []
                        video_groups[vname].append(f)
                else:
                    # Standard structure: folders are videos
                    try:
                        video_list = os.listdir(original_subfolder_path)
                        for v in video_list:
                            v_path = os.path.join(original_subfolder_path, v)
                            if os.path.isdir(v_path):
                                frames = sorted(glob.glob(os.path.join(v_path, "*")))
                                if frames:
                                    video_groups[v] = frames
                    except Exception as e:
                        print(f"Error listing directory: {e}")
        
        # If optical_only, we MUST scan the optical folder
        if args.eval_mode == "optical_only":
            if os.path.exists(optical_subfolder_path):
                files = sorted(glob.glob(os.path.join(optical_subfolder_path, "*.png")) + 
                             glob.glob(os.path.join(optical_subfolder_path, "*.jpg")) +
                             glob.glob(os.path.join(optical_subfolder_path, "*.JPEG")))
                
                if len(files) > 0 and not os.path.isdir(files[0]):
                    for f in files:
                        vname = get_video_name_from_filename(f)
                        if vname not in video_groups: video_groups[vname] = []
                        video_groups[vname].append(f)
                else:
                    try:
                        video_list = os.listdir(optical_subfolder_path)
                        for v in video_list:
                            v_path = os.path.join(optical_subfolder_path, v)
                            if os.path.isdir(v_path):
                                frames = sorted(glob.glob(os.path.join(v_path, "*")))
                                if frames:
                                    video_groups[v] = frames
                    except Exception as e:
                        print(f"Error listing directory: {e}")

        # If optical only or fused, we might need to check optical folder too
        # But usually RGB folder structure defines the videos.
        
        print(f"Found {len(video_groups)} videos in {subfolder_name}")
        
        video_list = list(video_groups.items())
        
        # Apply limit if specified
        if args.limit is not None and len(video_list) > args.limit:
            print(f"⚠️  Limiting to first {args.limit} videos (out of {len(video_list)})")
            video_list = video_list[:args.limit]
            
        for idx, (video_name, frames) in enumerate(video_list, 1):
            progress_pct = (idx / len(video_list)) * 100
            print(f"\r[{idx}/{len(video_list)} - {progress_pct:.1f}%] Processing: {video_name[:50]}...", end='', flush=True)
            
            # Detect RGB stream
            original_predict = 0
            if args.eval_mode in ["fused", "rgb_only"] and model_or is not None:
                original_prob_sum=0
                count = 0
                for img_path in frames:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img = trans(img)
                        if args.aug_norm:
                            img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        in_tens = img.unsqueeze(0)
                        if not args.use_cpu: in_tens = in_tens.cuda()
                        
                        with torch.no_grad():
                            prob = model_or(in_tens).sigmoid().item()
                            original_prob_sum+=prob
                        
                        df1 = pd.concat([df1, pd.DataFrame([{'original_path': img_path, 'original_pro': prob , 'flag':flag}])], ignore_index=True)
                        count += 1
                    except Exception as e:
                        print(f"Error processing RGB frame {img_path}: {e}")
                
                if count > 0:
                    original_predict = original_prob_sum/count

            # Detect Optical Flow stream
            optical_predict = 0
            if args.eval_mode in ["fused", "optical_only"] and model_op is not None:
                # Construct optical flow paths
                # In flat structure: optical_path/video_name_XXXX.png
                # In standard: optical_path/video_name/frame_XXXX.png
                
                optical_prob_sum=0
                count = 0
                
                # We iterate through the SAME frames as RGB to ensure alignment
                # But we need to find the corresponding optical flow file
                for img_path in frames:
                    basename = os.path.basename(img_path)
                    
                    # Construct optical flow path
                    # Try standard path: .../1_fake/video_name/frame.png
                    opt_path_standard = os.path.join(optical_subfolder_path, video_name, basename)
                    # Try flat path: .../1_fake/frame.png
                    opt_path_flat = os.path.join(optical_subfolder_path, basename)
                    
                    if os.path.exists(opt_path_standard):
                        opt_path = opt_path_standard
                    elif os.path.exists(opt_path_flat):
                        opt_path = opt_path_flat
                    else:
                        # Fallback to standard for error reporting, or try root if flat structure
                        if current_label_path == "":
                             opt_path = os.path.join(optical_subfolder_path, basename)
                        else:
                             opt_path = opt_path_standard
                    
                    if os.path.exists(opt_path):
                        try:
                            img = Image.open(opt_path).convert("RGB")
                            img = trans(img)
                            if args.aug_norm:
                                img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            in_tens = img.unsqueeze(0)
                            if not args.use_cpu: in_tens = in_tens.cuda()

                            with torch.no_grad():
                                prob = model_op(in_tens).sigmoid().item()
                                optical_prob_sum+=prob
                            
                            df1.loc[index1, 'optical_path'] = opt_path
                            df1.loc[index1, 'optical_pro'] = prob
                            index1+=1
                            count+=1
                        except Exception as e:
                            print(f"Error processing Flow frame {opt_path}: {e}")
                            index1+=1
                    else:
                        # Flow frame missing
                        index1+=1

                if count > 0:
                    optical_predict = optical_prob_sum/count

            # Final Prediction
            if args.eval_mode == "fused":
                predict = original_predict * 0.5 + optical_predict * 0.5
            elif args.eval_mode == "rgb_only":
                predict = original_predict
            else:
                predict = optical_predict
            
            y_true.append(flag)
            y_pred.append(predict)
            y_pred_original.append(original_predict)
            y_pred_optical.append(optical_predict)
            
            if flag==0:
                n+=1
                if predict<args.threshold: tn+=1
            else:
                p+=1
                if predict>=args.threshold: tp+=1
            
            df = pd.concat([df, pd.DataFrame([{'name': video_name, 'pro': predict , 'flag':flag ,
                          'optical_pro':optical_predict,'original_pro':original_predict}])], ignore_index=True)
        
        # Print newline after completing subfolder
        print(f"\n✓ Completed {subfolder_name}: {len(video_list)} videos processed")

    # Metrics
    # Metrics
    try:
        if len(y_true) == 0:
            print("Error: No videos were processed. Cannot calculate metrics.")
            ap = 0.0; auc = 0.0; acc = 0.0
        elif len(set(y_true)) > 1:
            ap = average_precision_score(y_true, y_pred)
            auc = roc_auc_score(y_true,y_pred)
            acc = accuracy_score(y_true, [1 if p >= args.threshold else 0 for p in y_pred])
        else:
            ap = 0.0; auc = 0.0
            # Calculate accuracy even if only one class
            acc = accuracy_score(y_true, [1 if p >= args.threshold else 0 for p in y_pred])
            print(f"Warning: Only one class present (Class {y_true[0]}). AUC/AP cannot be calculated.")
    except Exception as e: 
        print(f"Error calculating metrics: {e}")
        ap=0.0; auc=0.0; acc=0.0
    
    print("-" * 30)
    print(f"Evaluation Mode: {args.eval_mode}")
    print(f"tnr: {tn/n if n > 0 else 0:.4f}")
    print(f"tpr: {tp/p if p > 0 else 0:.4f}")
    print(f"acc: {acc:.4f}")
    print(f"auc: {auc:.4f}")
    print("-" * 30)

    # Save results
    csv_folder = os.path.dirname(args.excel_path)
    if not os.path.exists(csv_folder): os.makedirs(csv_folder)
    
    if not os.path.exists(args.excel_path): df.to_csv(args.excel_path, index=False)
    else: df.to_csv(args.excel_path, mode='a', header=False, index=False)
    
    csv_folder1 = os.path.dirname(args.excel_frame_path)
    if not os.path.exists(csv_folder1): os.makedirs(csv_folder1)

    if not os.path.exists(args.excel_frame_path): df1.to_csv(args.excel_frame_path, index=False)
    else: df1.to_csv(args.excel_frame_path, mode='a', header=False, index=False)
    
    print(f"Results saved to {args.excel_path}")
