"""
Simple script to download and extract test data for AIGVDet evaluation
Google Drive folder: https://drive.google.com/drive/u/3/folders/1gSAUUqYK33262aukdTjZIxUuGrgU8REU

This script:
1. Downloads the test folder from Google Drive using gdown
2. Extracts all zip files to the correct locations
3. Organizes data structure for recreate_table2_final.py
"""

import os
import subprocess
import zipfile
from pathlib import Path
import sys
import shutil

# Google Drive folder link
GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1D1jm1_HCu0Nv21NVjuyL1CB5gF5sy0hx"

# Paths
DATA_DIR = Path("data")
TEST_DIR = DATA_DIR / "test"
TEMP_DIR = DATA_DIR / "temp_download"

def install_gdown():
    """Install gdown if not already installed"""
    print("Checking for gdown...")
    try:
        import gdown
        print("✓ gdown is already installed")
        return True
    except ImportError:
        print("Installing gdown...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            print("✓ gdown installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install gdown: {e}")
            return False

def download_data():
    """Download test data from Google Drive"""
    print(f"\n{'='*80}")
    print("DOWNLOADING TEST DATA")
    print(f"{'='*80}")
    print(f"Source: {GDRIVE_FOLDER}")
    print(f"Destination: {TEMP_DIR}")
    
    # Create temp directory
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if files already exist
    existing_zips = list(TEMP_DIR.rglob("*.zip"))
    if existing_zips:
        print(f"\n⚠️  Found {len(existing_zips)} existing zip files in {TEMP_DIR}")
        print("Skipping download. If you want to re-download, delete the temp_download folder first.")
        return True
    
    # Try method 1: gdown with cookies
    try:
        import gdown
        print("\nMethod 1: Trying download with cookies authentication...")
        gdown.download_folder(GDRIVE_FOLDER, output=str(TEMP_DIR), quiet=False, use_cookies=True)
        print("\n✓ Download complete")
        return True
    except Exception as e:
        print(f"\n⚠️  Method 1 failed: {e}")
    
    # Try method 2: gdown without cookies
    try:
        import gdown
        print("\nMethod 2: Trying download without cookies...")
        gdown.download_folder(GDRIVE_FOLDER, output=str(TEMP_DIR), quiet=False, use_cookies=False)
        print("\n✓ Download complete")
        return True
    except Exception as e:
        print(f"\n⚠️  Method 2 failed: {e}")
    
    # Try method 3: Command line with cookies
    try:
        print("\nMethod 3: Trying command line with cookies...")
        cmd = ["gdown", "--folder", GDRIVE_FOLDER, "-O", str(TEMP_DIR), "--use-cookies"]
        subprocess.run(cmd, check=True)
        print("\n✓ Download complete")
        return True
    except Exception as e:
        print(f"\n⚠️  Method 3 failed: {e}")
    
    # All methods failed
    print("\n" + "="*80)
    print("❌ AUTOMATIC DOWNLOAD FAILED - MANUAL DOWNLOAD REQUIRED")
    print("="*80)
    print("\nGoogle Drive has rate-limited downloads. Please download manually:")
    print(f"\n1. Open in browser: {GDRIVE_FOLDER}")
    print("2. Download all files/folders")
    print(f"3. Extract to: {TEMP_DIR.absolute()}")
    print("4. Run this script again")
    print("\nAlternatively, wait 24 hours and try again.")
    print("="*80)
    return False

def extract_zip(zip_path, extract_to):
    """Extract a single zip file"""
    print(f"  Extracting: {zip_path.name} → {extract_to.name}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False

def organize_data():
    """Extract and organize all zip files"""
    print(f"\n{'='*80}")
    print("EXTRACTING AND ORGANIZING DATA")
    print(f"{'='*80}")
    
    # Create target directories
    optical_base = TEST_DIR / "T2V"
    rgb_base = TEST_DIR / "original" / "T2V"
    optical_base.mkdir(parents=True, exist_ok=True)
    rgb_base.mkdir(parents=True, exist_ok=True)
    
    # Find all zip files in temp directory
    all_zips = list(TEMP_DIR.rglob("*.zip"))
    
    if not all_zips:
        print("❌ No zip files found in downloaded data")
        return False
    
    print(f"\nFound {len(all_zips)} zip files")
    
    # Process optical flow zips
    print("\n1. Processing optical flow data...")
    optical_zips = [z for z in all_zips if "-optical" in z.name.lower()]
    for zip_file in optical_zips:
        # Extract dataset name (e.g., "videocraft" from "videocraft-optical.zip")
        dataset_name = zip_file.stem.replace("-optical", "").replace("-Optical", "")
        target_dir = optical_base / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        extract_zip(zip_file, target_dir)
    
    # Process RGB zips
    print("\n2. Processing RGB data...")
    rgb_zips = [z for z in all_zips if "-rgb" in z.name.lower()]
    for zip_file in rgb_zips:
        # Extract dataset name (e.g., "videocraft" from "videocraft-rgb.zip")
        dataset_name = zip_file.stem.replace("-rgb", "").replace("-RGB", "")
        target_dir = rgb_base / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        extract_zip(zip_file, target_dir)
    
    print("\n✓ Extraction complete")
    return True

def verify_structure():
    """Verify the extracted data structure"""
    print(f"\n{'='*80}")
    print("VERIFYING DATA STRUCTURE")
    print(f"{'='*80}")
    
    datasets = ["moonvalley", "videocraft", "pika", "neverends"]
    all_good = True
    
    print("\nOptical flow data:")
    for dataset in datasets:
        path = TEST_DIR / "T2V" / dataset
        if path.exists():
            real = (path / "0_real").exists()
            fake = (path / "1_fake").exists()
            status = f"[Real: {'✓' if real else '✗'}, Fake: {'✓' if fake else '✗'}]"
            print(f"  {'✓' if (real and fake) else '⚠️ '} {dataset}: {status}")
            if not (real and fake):
                all_good = False
        else:
            print(f"  ❌ {dataset}: NOT FOUND")
            all_good = False
    
    print("\nRGB data:")
    for dataset in datasets:
        path = TEST_DIR / "original" / "T2V" / dataset
        if path.exists():
            real = (path / "0_real").exists()
            fake = (path / "1_fake").exists()
            status = f"[Real: {'✓' if real else '✗'}, Fake: {'✓' if fake else '✗'}]"
            print(f"  {'✓' if (real and fake) else '⚠️ '} {dataset}: {status}")
            if not (real and fake):
                all_good = False
        else:
            print(f"  ❌ {dataset}: NOT FOUND")
            all_good = False
    
    return all_good

def cleanup():
    """Remove temporary files"""
    print(f"\n{'='*80}")
    print("CLEANUP")
    print(f"{'='*80}")
    
    if TEMP_DIR.exists():
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"✓ Removed temporary directory: {TEMP_DIR}")
        except Exception as e:
            print(f"⚠️  Could not remove temp directory: {e}")
            print(f"You can manually delete: {TEMP_DIR}")

def main():
    print("="*80)
    print("AIGVDET TEST DATA SETUP")
    print("="*80)
    
    # Step 1: Install gdown
    if not install_gdown():
        print("\n❌ Setup failed: Could not install gdown")
        return
    
    # Step 2: Download data
    if not download_data():
        print("\n❌ Setup failed: Could not download data")
        return
    
    # Step 3: Extract and organize
    if not organize_data():
        print("\n❌ Setup failed: Could not extract data")
        return
    
    # Step 4: Verify structure
    success = verify_structure()
    
    # Step 5: Cleanup
    cleanup()
    
    # Final message
    print(f"\n{'='*80}")
    if success:
        print("✓ SETUP COMPLETE!")
        print("="*80)
        print("\nAll test data is ready!")
        print("\nNext step:")
        print("  python recreate_table2_final.py")
    else:
        print("⚠️  SETUP COMPLETED WITH WARNINGS")
        print("="*80)
        print("\nSome data may be missing. Please check the warnings above.")
        print("You may need to manually extract some files.")
    print("="*80)

if __name__ == "__main__":
    main()
