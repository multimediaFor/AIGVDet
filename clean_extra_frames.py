import os
import glob
import argparse

def clean_extra_frames(base_dir, max_frame_index=94):
    print(f"Cleaning frames with index > {max_frame_index} in {base_dir}")
    
    # Datasets to check
    datasets = ["moonvalley", "videocraft", "pika", "neverends"]
    
    total_deleted = 0
    
    for dataset in datasets:
        target_dir = os.path.join(base_dir, dataset, "0_real")
        
        if not os.path.exists(target_dir):
            print(f"Skipping {target_dir} (not found)")
            continue
            
        print(f"Scanning {target_dir}...")
        
        # Check if it's flat files or folders
        # Based on previous ls, it seems to be flat files for 0_real in some contexts, 
        # but prepare_real_data.py copies folders: shutil.copytree(temp_flow_dir, flow_dest)
        # Let's handle both cases.
        
        items = os.listdir(target_dir)
        for item in items:
            item_path = os.path.join(target_dir, item)
            
            if os.path.isdir(item_path):
                # It's a video folder
                video_name = item
                frames = glob.glob(os.path.join(item_path, "*"))
                for frame in frames:
                    if should_delete(frame, max_frame_index):
                        os.remove(frame)
                        total_deleted += 1
            else:
                # It's a flat file
                if should_delete(item_path, max_frame_index):
                    os.remove(item_path)
                    total_deleted += 1
                    
    print(f"\n✓ Cleanup complete. Deleted {total_deleted} extra frames.")

def should_delete(filepath, max_index):
    filename = os.path.basename(filepath)
    name_part = os.path.splitext(filename)[0]
    
    # Case 1: Filename is just a number (e.g. 00000094.jpg)
    if name_part.isdigit():
        idx = int(name_part)
        if idx > 90:
            print(f"  Checking {filename}: Index {idx} > {max_index}? {idx > max_index}")
        if idx > max_index:
            return True
            
    # Case 2: Filename has underscores (e.g. video_00000094.jpg)
    parts = name_part.split('_')
    if len(parts) > 1:
        last_part = parts[-1]
        if last_part.isdigit():
            idx = int(last_part)
            if idx > 90:
                print(f"  Checking {filename}: Index {idx} > {max_index}? {idx > max_index}")
            if idx > max_index:
                return True
    
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    args = parser.parse_args()
    
    # Optical Flow path (Limit to 94 frames -> max index 93)
    clean_extra_frames("data/test/T2V", max_frame_index=93)
    
    # RGB path (Limit to 95 frames -> max index 94)
    clean_extra_frames("data/test/original/T2V", max_frame_index=94)
