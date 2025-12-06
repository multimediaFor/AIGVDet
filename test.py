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

from core.utils1.utils import get_network, str2bool, to_cuda
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score,roc_auc_score

if __name__=="__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-fop", "--folder_optical_flow_path", default="data/test/T2V/videocraft", type=str, help="path to optical flow imagefile folder"
    )
    parser.add_argument(
        "-for", "--folder_original_path", default="data/test/original/T2V/videocraft", type=str, help="path to RGB image file folder"
    )
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
    
    parser.add_argument(
        "-e",
        "--excel_path",
        type=str,
        help="path to excel of frames",
        default="data/results/moonvalley_wang.csv",
    )
    
    parser.add_argument(
        "-ef",
        "--excel_frame_path",
        type=str,
        help="path to excel of frame detection result",
        default="data/results/frame/moonvalley_wang.csv",
    )
    
    
    
    
    parser.add_argument("--use_cpu", action="store_true", help="uses gpu by default, turn on to use cpu")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--aug_norm", type=str2bool, default=True)
    parser.add_argument("--no_crop", action="store_true", help="disable center crop")

    args = parser.parse_args()
    subfolder_count = 0

    model_op = get_network(args.arch)
    state_dict = torch.load(args.model_optical_flow_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model_op.load_state_dict(state_dict)
    model_op.eval()
    if not args.use_cpu:
        model_op.cuda()
        
        
    model_or = get_network(args.arch)
    state_dict = torch.load(args.model_original_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model_or.load_state_dict(state_dict)
    model_or.eval()
    if not args.use_cpu:
        model_or.cuda()
    
    
    if args.no_crop:
        trans = transforms.Compose(
            (
                transforms.ToTensor(),
            )
        )
    else:
        trans = transforms.Compose(
            (
                transforms.CenterCrop((448,448)),
                transforms.ToTensor(),
            )
        )

    print("*" * 50)

    flag=0
    p=0
    n=0
    tp=0
    tn=0
    y_true=[]
    y_pred=[]
    y_pred_original=[]
    y_pred_optical=[]

    # create an empty DataFrame
    df = pd.DataFrame(columns=['name', 'pro','flag','optical_pro','original_pro'])
    df1 = pd.DataFrame(columns=['original_path', 'original_pro','optical_path','optical_pro','flag'])
    index1=0
    
    # Traverse through subfolders in a large folder.
    for subfolder_name in ["0_real", "1_fake"]:
        optical_subfolder_path = os.path.join(args.folder_optical_flow_path, subfolder_name)
        original_subfolder_path = os.path.join(args.folder_original_path, subfolder_name)

        if subfolder_name=="0_real":
            flag=0
        else:
            flag=1
            
        if os.path.isdir(optical_subfolder_path):
            pass
        else:
            print("Subfolder does not exist.", optical_subfolder_path)
        
        # Check if the subfolder path exists.
        if os.path.isdir(original_subfolder_path):
            print("test subfolder:", subfolder_name)

            # Traverse through sub-subfolders within a subfolder.
            video_list = os.listdir(original_subfolder_path)
            for subsubfolder_name in tqdm(video_list, desc=f"Testing {subfolder_name}"):
                original_subsubfolder_path = os.path.join(original_subfolder_path, subsubfolder_name)
                optical_subsubfolder_path = os.path.join(optical_subfolder_path, subsubfolder_name)
                if os.path.isdir(optical_subsubfolder_path):
                    pass
                else:
                    print("Sub-subfolder does not exist.",optical_subsubfolder_path)
                    
                if os.path.isdir(original_subsubfolder_path):
                    print("test subsubfolder:", subsubfolder_name)
                    
                    #Detect original
                    original_file_list = sorted(glob.glob(os.path.join(original_subsubfolder_path, "*.jpg")) + glob.glob(os.path.join(original_subsubfolder_path, "*.png"))+glob.glob(os.path.join(original_subsubfolder_path, "*.JPEG")))

                    original_prob_sum=0
                    # Inner loop for frames - disable tqdm to avoid nested clutter
                    for img_path in original_file_list:
                        
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
                            
                        df1 = pd.concat([df1, pd.DataFrame([{'original_path': img_path, 'original_pro': prob , 'flag':flag}])], ignore_index=True)
                        
                        
                    original_predict=original_prob_sum/len(original_file_list)
                    # print("original prob",original_predict)
                    
                    #Detect optical flow
                    optical_file_list = sorted(glob.glob(os.path.join(optical_subsubfolder_path, "*.jpg")) + glob.glob(os.path.join(optical_subsubfolder_path, "*.png"))+glob.glob(os.path.join(optical_subsubfolder_path, "*.JPEG")))
                    optical_prob_sum=0
                    # Inner loop for frames - disable tqdm
                    for img_path in optical_file_list:
                        
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
                        
                        df1.loc[index1, 'optical_path'] = img_path
                        df1.loc[index1, 'optical_pro'] = prob
                        index1=index1+1
                    index1=index1+1
                    
                    optical_predict=optical_prob_sum/len(optical_file_list)
                    # print("optical prob",optical_predict)
                    
                    predict=original_predict*0.5+optical_predict*0.5
                    # print(f"flag:{flag} predict:{predict}")
                    # y_true.append((float)(flag))
                    y_true.append((flag))
                    y_pred.append(predict)
                    y_pred_original.append(original_predict)
                    y_pred_optical.append(optical_predict)
                    if flag==0:
                        n+=1
                        if predict<args.threshold:
                            tn+=1
                    else:
                        p+=1
                        if predict>=args.threshold:
                            tp+=1
                    df = pd.concat([df, pd.DataFrame([{'name': subsubfolder_name, 'pro': predict , 'flag':flag ,'optical_pro':optical_predict,'original_pro':original_predict}])], ignore_index=True)
        else:
            print("Subfolder does not exist:", original_subfolder_path)
    # r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > args.threshold)
    # f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > args.threshold)
    # acc = accuracy_score(y_true, y_pred > args.threshold)
    
    ap = average_precision_score(y_true, y_pred)
    auc=roc_auc_score(y_true,y_pred)
    acc = accuracy_score(y_true, [1 if p >= args.threshold else 0 for p in y_pred])
    
    # Calculate metrics for individual streams
    acc_original = accuracy_score(y_true, [1 if p >= args.threshold else 0 for p in y_pred_original])
    auc_original = roc_auc_score(y_true, y_pred_original)
    
    acc_optical = accuracy_score(y_true, [1 if p >= args.threshold else 0 for p in y_pred_optical])
    auc_optical = roc_auc_score(y_true, y_pred_optical)

    print("-" * 30)
    print("AIGVDet (Fused) Results:")
    print(f"tnr:{tn/n}")
    print(f"tpr:{tp/p}")
    print(f"acc:{acc}")
    print(f"ap:{ap}")
    print(f"auc:{auc}")
    
    print("-" * 30)
    print("Sspatial (Original RGB) Results:")
    print(f"acc:{acc_original}")
    print(f"auc:{auc_original}")
    
    print("-" * 30)
    print("Soptical (Optical Flow) Results:")
    print(f"acc:{acc_optical}")
    print(f"auc:{auc_optical}")
    print("-" * 30)
    print(f"p:{p}")
    print(f"n:{n}")
    print(f"tp:{tp}")
    print(f"tn:{tn}")

    # Write the DataFrame to a csv file.
    csv_filename = args.excel_path
    csv_folder = os.path.dirname(csv_filename) 
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)


    if not os.path.exists(csv_filename):
        df.to_csv(csv_filename, index=False)
    else:
        df.to_csv(csv_filename, mode='a', header=False, index=False)
    print(f"Results have been saved to {csv_filename}")
        
    # Write the prediction probabilities of the frame to a CSV file.
    csv_filename1 = args.excel_frame_path
    csv_folder1 = os.path.dirname(csv_filename1) 
    if not os.path.exists(csv_folder1):
        os.makedirs(csv_folder1)

    if not os.path.exists(csv_filename1):
        df1.to_csv(csv_filename1, index=False)
    else:
        df1.to_csv(csv_filename1, mode='a', header=False, index=False)
    
    # if not os.path.exists(excel_filename):
    #     with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
    #         df.to_excel(writer, sheet_name='Sheet1', index=False)
    # else:
    #     with pd.ExcelWriter(excel_filename, mode='a', engine='openpyxl') as writer:
    #         df.to_excel(writer, sheet_name='Sheet1', index=False, startrow=0, header=False)
    print(f"Results have been saved to {csv_filename1}")
    


