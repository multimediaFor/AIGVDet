"""
Batch Data Preparation Script
Automatically finds all *_mp4 folders in data/test/T2V and processes them.
"""
import os
import glob
import subprocess
from pathlib import Path

# Configuration
SOURCE_ROOT = "data/test/T2V"
RGB_OUTPUT_ROOT = "data/test/original/T2V"
FLOW_OUTPUT_ROOT = "data/test/T2V"
RAFT_MODEL = "raft-model/raft-things.pth"
MAX_VIDEOS = 100  # Limit to 100 videos per dataset for faster processing

def main():
    # Find all folders ending in _mp4
    source_folders = glob.glob(os.path.join(SOURCE_ROOT, "*_mp4"))
    
    print(f"Found {len(source_folders)} datasets to process: {[os.path.basename(f) for f in source_folders]}")
    
    for source_dir in source_folders:
        dataset_name = os.path.basename(source_dir).replace("_mp4", "")
        
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")
        
        # Define output paths
        rgb_out = os.path.join(RGB_OUTPUT_ROOT, dataset_name)
        flow_out = os.path.join(FLOW_OUTPUT_ROOT, dataset_name)
        
        cmd = [
            "python", "prepare_data.py",
            "--source_dir", source_dir,
            "--output_rgb_dir", rgb_out,
            "--output_flow_dir", flow_out,
            "--model", RAFT_MODEL,
            "--label", "1_fake",  # Assuming these are all generated video folders
            "--max_videos", str(MAX_VIDEOS)
        ]
        
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
