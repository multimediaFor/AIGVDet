#!/usr/bin/env python3
"""
Download script for AIGVDet data, checkpoints, and RAFT model.
Converts the original download_data.sh to Python with additional downloads.

Usage:
    python download_data.py [--data-id FILE_ID] [--skip-data] [--skip-checkpoints] [--skip-raft]
"""

import os
import sys
import argparse
import subprocess
import zipfile
from pathlib import Path


def check_and_install_gdown():
    """Check if gdown is installed, install if missing."""
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
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install gdown: {e}")
            return False


def ask_redownload(path_name):
    """Ask user if they want to re-download existing data."""
    response = input(f"Do you want to re-download? (y/N): ").strip().lower()
    return response in ['y', 'yes']


def download_from_drive(url_or_id, output_path, is_folder=False):
    """
    Download file or folder from Google Drive using gdown.
    
    Args:
        url_or_id: Google Drive URL or file ID
        output_path: Path where to save the downloaded file/folder
        is_folder: True if downloading a folder, False for single file
    """
    import gdown
    
    try:
        if is_folder:
            # Extract folder ID from URL if needed
            if 'folders/' in url_or_id:
                folder_id = url_or_id.split('folders/')[-1].split('?')[0]
            else:
                folder_id = url_or_id
            
            print(f"Downloading folder (ID: {folder_id})...")
            gdown.download_folder(id=folder_id, output=str(output_path), quiet=False)
        else:
            # Extract file ID from URL if needed
            if 'drive.google.com' in url_or_id:
                if '/file/d/' in url_or_id:
                    file_id = url_or_id.split('/file/d/')[-1].split('/')[0]
                elif 'id=' in url_or_id:
                    file_id = url_or_id.split('id=')[-1].split('&')[0]
                else:
                    file_id = url_or_id
            else:
                file_id = url_or_id
            
            print(f"Downloading file (ID: {file_id})...")
            gdown.download(id=file_id, output=str(output_path), quiet=False)
        
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def download_dataset(data_dir, file_id):
    """Download and extract the main dataset."""
    print("\n" + "="*60)
    print("DOWNLOADING DATASET")
    print("="*60)
    
    data_dir = Path(data_dir)
    zip_file = data_dir / "data.zip"
    
    # Check if data already exists
    train_dir = data_dir / "train"
    if train_dir.exists():
        print(f"Data directory {train_dir} already exists.")
        if not ask_redownload("dataset"):
            print("Skipping dataset download.")
            return True
    
    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    if not download_from_drive(file_id, zip_file, is_folder=False):
        return False
    
    # Extract
    if not extract_zip(zip_file, data_dir):
        return False
    
    # Cleanup
    print("Cleaning up zip file...")
    zip_file.unlink()
    
    print(f"✓ Dataset setup complete!")
    print(f"Contents of {data_dir}:")
    for item in sorted(data_dir.iterdir()):
        print(f"  {'📁' if item.is_dir() else '📄'} {item.name}")
    
    return True


def download_checkpoints(checkpoint_dir, folder_url):
    """Download checkpoint files (original.pth and optical.pth)."""
    print("\n" + "="*60)
    print("DOWNLOADING CHECKPOINTS")
    print("="*60)
    
    checkpoint_dir = Path(checkpoint_dir)
    
    # Check if checkpoints already exist
    if checkpoint_dir.exists() and any(checkpoint_dir.glob('*.pth')):
        print(f"Checkpoint directory {checkpoint_dir} already has .pth files:")
        for pth in checkpoint_dir.glob('*.pth'):
            print(f"  📄 {pth.name}")
        if not ask_redownload("checkpoints"):
            print("Skipping checkpoint download.")
            return True
    
    # Create checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Download folder from Google Drive
    # gdown will create files directly in the checkpoint_dir
    if not download_from_drive(folder_url, checkpoint_dir, is_folder=True):
        return False
    
    print(f"✓ Checkpoints download complete!")
    print(f"Contents of {checkpoint_dir}:")
    for item in sorted(checkpoint_dir.iterdir()):
        size_mb = item.stat().st_size / (1024 * 1024) if item.is_file() else 0
        size_str = f"({size_mb:.1f} MB)" if item.is_file() else ""
        print(f"  {'📁' if item.is_dir() else '📄'} {item.name} {size_str}")
    
    return True


def download_raft_model(raft_dir, file_url):
    """Download RAFT model file."""
    print("\n" + "="*60)
    print("DOWNLOADING RAFT MODEL")
    print("="*60)
    
    raft_dir = Path(raft_dir)
    model_file = raft_dir / "raft_things.pth"
    
    # Check if RAFT model already exists
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"RAFT model already exists: {model_file.name} ({size_mb:.1f} MB)")
        if not ask_redownload("RAFT model"):
            print("Skipping RAFT model download.")
            return True
    
    # Create RAFT directory
    raft_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    if not download_from_drive(file_url, model_file, is_folder=False):
        return False
    
    print(f"✓ RAFT model download complete!")
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  📄 {model_file.name} ({size_mb:.1f} MB)")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download AIGVDet data, checkpoints, and RAFT model'
    )
    parser.add_argument(
        '--data-id',
        default='1Bgv_TA18CIXwxQ5EQoSybsSC3EcivVv1',
        help='Google Drive file ID for dataset (default: from original script)'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory to save dataset (default: data)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        default='checkpoints',
        help='Directory to save checkpoints (default: checkpoints)'
    )
    parser.add_argument(
        '--checkpoint-folder',
        default='https://drive.google.com/drive/folders/18JO_YxOEqwJYfbVvy308XjoV-N6fE4yP',
        help='Google Drive folder URL for checkpoints'
    )
    parser.add_argument(
        '--raft-dir',
        default='raft_model',
        help='Directory to save RAFT model (default: raft_model)'
    )
    parser.add_argument(
        '--raft-file',
        default='https://drive.google.com/file/d/1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM/view',
        help='Google Drive file URL for RAFT model'
    )
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip dataset download'
    )
    parser.add_argument(
        '--skip-checkpoints',
        action='store_true',
        help='Skip checkpoint download'
    )
    parser.add_argument(
        '--skip-raft',
        action='store_true',
        help='Skip RAFT model download'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("AIGVDet Download Script")
    print("="*60)
    
    # Check and install gdown
    if not check_and_install_gdown():
        print("\n✗ Cannot proceed without gdown. Please install it manually:")
        print("  pip install gdown")
        return 1
    
    success = True
    
    # Download dataset
    if not args.skip_data:
        if not download_dataset(args.data_dir, args.data_id):
            print("\n✗ Dataset download failed!")
            success = False
    else:
        print("\nSkipping dataset download (--skip-data)")
    
    # Download checkpoints
    if not args.skip_checkpoints:
        if not download_checkpoints(args.checkpoint_dir, args.checkpoint_folder):
            print("\n✗ Checkpoint download failed!")
            success = False
    else:
        print("\nSkipping checkpoint download (--skip-checkpoints)")
    
    # Download RAFT model
    if not args.skip_raft:
        if not download_raft_model(args.raft_dir, args.raft_file):
            print("\n✗ RAFT model download failed!")
            success = False
    else:
        print("\nSkipping RAFT model download (--skip-raft)")
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("✓ ALL DOWNLOADS COMPLETE!")
    else:
        print("⚠ SOME DOWNLOADS FAILED - Check errors above")
    print("="*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
