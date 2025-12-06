import argparse
import glob
import os
import pandas as pd
import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
from core.utils1.utils import get_network, str2bool

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-fop", "--folder_optical_flow_path", type=str, required=True)
    parser.add_argument("-for", "--folder_original_path", type=str, required=True)
    parser.add_argument("-mop", "--model_optical_flow_path", type=str, default="checkpoints/optical.pth")
    parser.add_argument("-mor", "--model_original_path", type=str, default="checkpoints/original.pth")
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("-e", "--excel_path", type=str, default="results.csv")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--aug_norm", type=str2bool, default=True)
    parser.add_argument("--no_crop", action="store_true")
    
    args = parser.parse_args()

    # Load Models
    device = torch.device("cpu" if args.use_cpu else "cuda")
    
    print("Loading models...")
    model_op = get_network(args.arch).to(device)
    model_op.load_state_dict(torch.load(args.model_optical_flow_path, map_location=device)["model"])
    model_op.eval()
    
    model_or = get_network(args.arch).to(device)
    model_or.load_state_dict(torch.load(args.model_original_path, map_location=device)["model"])
    model_or.eval()
    
    # Transforms
    if args.no_crop:
        trans = transforms.Compose([transforms.ToTensor()])
    else:
        trans = transforms.Compose([transforms.CenterCrop((448, 448)), transforms.ToTensor()])

    print(f"Processing flat directories...")
    print(f"RGB: {args.folder_original_path}")
    print(f"Flow: {args.folder_optical_flow_path}")

    # Get list of images
    # We assume filenames match between RGB and Flow (except maybe extension)
    # Actually, process_custom_videos outputs:
    # RGB: video_name_frame_00001.jpg
    # Flow: video_name_frame_00001.jpg (or .png)
    
    rgb_files = sorted(glob.glob(os.path.join(args.folder_original_path, "*")))
    rgb_files = [f for f in rgb_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(rgb_files) == 0:
        print("No RGB images found!")
        return

    print(f"Found {len(rgb_files)} RGB frames.")
    
    y_true = []
    y_pred = []
    y_pred_rgb = []
    y_pred_flow = []
    
    results = []

    # Iterate
    for rgb_path in tqdm(rgb_files):
        filename = os.path.basename(rgb_path)
        # Try to find corresponding flow file
        # Flow might be .png even if RGB is .jpg
        flow_path = os.path.join(args.folder_optical_flow_path, filename)
        if not os.path.exists(flow_path):
            # Try replacing extension
            name, ext = os.path.splitext(filename)
            flow_path_png = os.path.join(args.folder_optical_flow_path, name + ".png")
            flow_path_jpg = os.path.join(args.folder_optical_flow_path, name + ".jpg")
            if os.path.exists(flow_path_png):
                flow_path = flow_path_png
            elif os.path.exists(flow_path_jpg):
                flow_path = flow_path_jpg
            else:
                # print(f"Warning: Flow file not found for {filename}")
                continue

        # Load and Preprocess
        try:
            img_rgb = Image.open(rgb_path).convert("RGB")
            img_flow = Image.open(flow_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        # Transform
        t_rgb = trans(img_rgb)
        t_flow = trans(img_flow)
        
        if args.aug_norm:
            t_rgb = TF.normalize(t_rgb, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            t_flow = TF.normalize(t_flow, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        t_rgb = t_rgb.unsqueeze(0).to(device)
        t_flow = t_flow.unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            prob_rgb = model_or(t_rgb).sigmoid().item()
            prob_flow = model_op(t_flow).sigmoid().item()
            
        prob_fused = 0.5 * prob_rgb + 0.5 * prob_flow
        
        # Assume Fake (1) since we are testing generators
        label = 1 
        
        y_true.append(label)
        y_pred.append(prob_fused)
        y_pred_rgb.append(prob_rgb)
        y_pred_flow.append(prob_flow)
        
        results.append({
            "filename": filename,
            "prob_fused": prob_fused,
            "prob_rgb": prob_rgb,
            "prob_flow": prob_flow,
            "label": label
        })

    # Metrics
    if len(y_true) == 0:
        print("No valid pairs processed.")
        return

    acc_fused = accuracy_score(y_true, [1 if p >= args.threshold else 0 for p in y_pred])
    acc_rgb = accuracy_score(y_true, [1 if p >= args.threshold else 0 for p in y_pred_rgb])
    acc_flow = accuracy_score(y_true, [1 if p >= args.threshold else 0 for p in y_pred_flow])
    
    print("-" * 30)
    print(f"Results (Assuming all inputs are FAKE/Generated)")
    print(f"Total Frames: {len(y_true)}")
    print("-" * 30)
    print(f"Fused Accuracy (Recall): {acc_fused:.4f}")
    print(f"RGB Accuracy (Recall):   {acc_rgb:.4f}")
    print(f"Flow Accuracy (Recall):  {acc_flow:.4f}")
    print("-" * 30)
    
    # Save CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.excel_path), exist_ok=True)
    df.to_csv(args.excel_path, index=False)
    print(f"Saved results to {args.excel_path}")

if __name__ == "__main__":
    main()
