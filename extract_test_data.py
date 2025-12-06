"""
Manual extraction script for test data
Use this if automatic download fails due to Google Drive rate limits

INSTRUCTIONS:
1. Manually download the test folder from Google Drive
2. Place all zip files in: data/temp_download/
3. Run this script to extract everything

Expected zip files:
- moonvalley-optical.zip
- videocraft-optical.zip  
- pika-optical.zip
- neverends-optical.zip
- moonvalley-rgb.zip
- videocraft-rgb.zip
- pika-rgb.zip
- neverends-rgb.zip
"""

import zipfile
from pathlib import Path
import shutil

# Paths
DATA_DIR = Path("data")
TEST_DIR = DATA_DIR / "test"
TEMP_DIR = DATA_DIR / "temp_download"

def extract_zip(zip_path, extract_to):
    """Extract a single zip file"""
    print(f"  Extracting: {zip_path.name}")
    print(f"    → {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"    ✓ Done")
        return True
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False

def main():
    print("="*80)
    print("MANUAL TEST DATA EXTRACTION")
    print("="*80)
    
    # Check if temp directory exists
    if not TEMP_DIR.exists():
        print(f"\n❌ Directory not found: {TEMP_DIR}")
        print(f"\nPlease create it and place your zip files there:")
        print(f"  mkdir {TEMP_DIR}")
        return
    
    # Find all zip files
    all_zips = list(TEMP_DIR.rglob("*.zip"))
    
    if not all_zips:
        print(f"\n❌ No zip files found in: {TEMP_DIR}")
        print("\nPlease download the following files from Google Drive:")
        print("  https://drive.google.com/drive/folders/1D1jm1_HCu0Nv21NVjuyL1CB5gF5sy0hx")
        print(f"\nAnd place them in: {TEMP_DIR.absolute()}")
        return
    
    print(f"\n✓ Found {len(all_zips)} zip files")
    for z in all_zips:
        print(f"  - {z.name}")
    
    # Create target directories
    optical_base = TEST_DIR / "T2V"
    rgb_base = TEST_DIR / "original" / "T2V"
    optical_base.mkdir(parents=True, exist_ok=True)
    rgb_base.mkdir(parents=True, exist_ok=True)
    
    # Extract optical flow data
    print(f"\n{'='*80}")
    print("EXTRACTING OPTICAL FLOW DATA")
    print("="*80)
    
    optical_zips = [z for z in all_zips if "-optical" in z.name.lower()]
    for zip_file in optical_zips:
        dataset_name = zip_file.stem.replace("-optical", "").replace("-Optical", "")
        target_dir = optical_base / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        extract_zip(zip_file, target_dir)
    
    # Extract RGB data
    print(f"\n{'='*80}")
    print("EXTRACTING RGB DATA")
    print("="*80)
    
    rgb_zips = [z for z in all_zips if "-rgb" in z.name.lower()]
    for zip_file in rgb_zips:
        dataset_name = zip_file.stem.replace("-rgb", "").replace("-RGB", "")
        target_dir = rgb_base / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        extract_zip(zip_file, target_dir)
    
    # Verify structure
    print(f"\n{'='*80}")
    print("VERIFYING STRUCTURE")
    print("="*80)
    
    datasets = ["moonvalley", "videocraft", "pika", "neverends"]
    all_good = True
    
    print("\nOptical flow:")
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
    
    print("\nRGB:")
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
    
    # Cleanup
    print(f"\n{'='*80}")
    cleanup = input("\nRemove temporary zip files? (y/n): ").strip().lower()
    if cleanup == 'y':
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"✓ Removed: {TEMP_DIR}")
        except Exception as e:
            print(f"⚠️  Could not remove: {e}")
    
    # Final message
    print(f"\n{'='*80}")
    if all_good:
        print("✓ EXTRACTION COMPLETE!")
        print("="*80)
        print("\nAll test data is ready!")
        print("\nNext step:")
        print("  python recreate_table2_final.py")
    else:
        print("⚠️  EXTRACTION COMPLETED WITH WARNINGS")
        print("="*80)
        print("\nSome data may be missing. Check the warnings above.")
    print("="*80)

if __name__ == "__main__":
    main()
