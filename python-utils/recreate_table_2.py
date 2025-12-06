import argparse
import subprocess
import os
import sys

def run_command(command):
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Recreate Table 2 results for a specific dataset.")
    parser.add_argument("dataset", help="Name of the dataset (e.g., moonvalley, videocraft, pika, neverends)")
    parser.add_argument("--rgb_dir", help="Path to RGB frames directory", default=None)
    parser.add_argument("--flow_dir", help="Path to Optical Flow frames directory", default=None)
    args = parser.parse_args()

    dataset_name = args.dataset
    
    # Define paths
    # Default paths (relative to current directory)
    base_data_dir = "data"
    
    # If arguments are provided, use them. Otherwise, construct default paths.
    if args.rgb_dir:
        output_rgb_dir = args.rgb_dir
    else:
        # Matches structure: data/test/videocraft_rgb
        output_rgb_dir = os.path.join(base_data_dir, "test", f"{dataset_name}_rgb")
        
    if args.flow_dir:
        output_flow_dir = args.flow_dir
    else:
        # Matches structure: data/test/videocraft_flow
        output_flow_dir = os.path.join(base_data_dir, "test", f"{dataset_name}_flow")
    
    # Source video dir (only needed for preparation step, which is skipped)
    source_video_dir = os.path.join(base_data_dir, "test", "T2V", dataset_name)
    
    result_csv = os.path.join(base_data_dir, "results", f"{dataset_name}.csv")
    result_no_cp_csv = os.path.join(base_data_dir, "results", f"{dataset_name}_no_cp.csv")
    
    # Model paths (relative)
    raft_model = "raft_model/raft-things.pth"
    optical_model = "checkpoints/optical.pth"
    original_model = "checkpoints/original.pth"

    print(f"--- Processing Dataset: {dataset_name} ---")
    print(f"RGB Directory: {output_rgb_dir}")
    print(f"Flow Directory: {output_flow_dir}")
    
    # Check if directories exist
    if not os.path.exists(output_rgb_dir):
        print(f"Warning: RGB directory not found: {output_rgb_dir}")
    if not os.path.exists(output_flow_dir):
        print(f"Warning: Flow directory not found: {output_flow_dir}")

    # Step 1: Prepare Data
    # print("\n[Step 1] Preparing Data (Extracting Frames & Generating Optical Flow)...")
    # # Note: We use sys.executable to ensure we use the same python interpreter
    # cmd_prepare = f'"{sys.executable}" prepare_data.py --source_dir "{source_video_dir}" --output_rgb_dir "{output_rgb_dir}" --output_flow_dir "{output_flow_dir}" --model "{raft_model}"'
    # run_command(cmd_prepare)
    
    # Step 2: Run Standard Evaluation
    print("\n[Step 2] Running Standard Evaluation (AIGVDet, Sspatial, Soptical)...")
    cmd_test = f'"{sys.executable}" test_flat.py -fop "{output_flow_dir}" -for "{output_rgb_dir}" -mop "{optical_model}" -mor "{original_model}" -e "{result_csv}"'
    run_command(cmd_test)
    
    # Step 3: Run No-Crop Evaluation
    print("\n[Step 3] Running No-Crop Evaluation (Soptical no cp)...")
    cmd_test_no_cp = f'"{sys.executable}" test_flat.py --no_crop -fop "{output_flow_dir}" -for "{output_rgb_dir}" -mop "{optical_model}" -mor "{original_model}" -e "{result_no_cp_csv}"'
    run_command(cmd_test_no_cp)
    
    print("\n--- Done! ---")
    print(f"Standard Results saved to: {result_csv}")
    print(f"No-Crop Results saved to: {result_no_cp_csv}")

if __name__ == "__main__":
    main()
