# Data Setup Instructions

## Required Datasets

### Training Data
- Download from Baiduyun Link (extract code: ra95)
- Extract to: `./data/train/`

### Test Data
**Note:** Training/validation data are frame sequences (PNG), but testing typically needs video files.

**Solutions:**
1. **Reconstruct videos from frames:** Convert frame sequences back to videos for testing
2. **Use alternative video datasets:**
   - FaceForensics++: https://github.com/ondyari/FaceForensics (has videos)
   - Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics
   - DFDC: https://dfdc.ai/
3. **Test with frame sequences:** Modify test scripts to work with PNG sequences
4. **Create test split:** Use some frame sequences as test data

### Data Structure
```
data/
├── train/
│   └── trainset_1/
│       ├── 0_real/
│       │   ├── video_00000/
│       │   │   ├── 00000.png
│       │   │   └── ...
│       │   └── ...
│       └── 1_fake/
│           └── ...
├── val/
│   └── val_set_1/
│       └── ... (same structure)
└── test/
    └── testset_1/
        └── ... (same structure)
```

## Getting the Original Data

### Step 1: Download Training Data
**Original Baiduyun Link:** https://pan.baidu.com/s/17xmDyFjtcmNsoxmUeImMTQ?pwd=ra95
- Extract code: `ra95`
- Download the preprocessed training frames  
- Extract to: `./data/train/`

### Step 2: Download Test Videos
**Original Google Drive Link:** https://drive.google.com/drive/folders/1D84SRWEJ8BK8KBpTMuGi3BUM80mW_dKb?usp=sharing
- Download test videos
- Extract to: `./data/test/`

### Step 3: Download Model Weights
**Model Checkpoints:** https://drive.google.com/drive/folders/18JO_YxOEqwJYfbVvy308XjoV-N6fE4yP?usp=share_link
- Download trained model weights
- Move to: `./checkpoints/`

**RAFT Model (for demo.py):** https://drive.google.com/file/d/1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM/view
- Download RAFT model weights
- Move to: `./raft_model/`

### Backup Plan: If Original Links Are Broken
1. Contact paper authors: lyan924@cuc.edu.cn
2. Check paper's GitHub issues for updated links
3. Look for replication studies with alternative datasets

### Step 3: Alternative if Original Data Unavailable
- Use FaceForensics++ dataset (widely used benchmark)
- Download from: https://github.com/ondyari/FaceForensics
- Convert to the same folder structure

## Docker Usage
```bash
# Windows
docker run -it -v C:\path\to\your\data:/app/data sacdalance/thesis-aigvdet:gpu bash

# Linux/Mac  
docker run -it -v /path/to/your/data:/app/data sacdalance/thesis-aigvdet:gpu bash

# With GPU support
docker run --gpus all -it -v /path/to/data:/app/data sacdalance/thesis-aigvdet:gpu bash
```

## Local Development with venv
```bash
# Activate venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\Activate.ps1  # Windows

# Install dependencies
uv pip install -e .

# Set data path
export AIGVDET_DATA_PATH="/path/to/your/data"  # Linux/Mac
$env:AIGVDET_DATA_PATH = "C:\path\to\your\data"  # Windows
```

## Running the Paper

### Training
```bash
# Basic training command
python train.py --gpus 0 --exp_name TRAIN_RGB_BRANCH datasets RGB_TRAINSET datasets_test RGB_TESTSET

# For optical flow branch
python train.py --gpus 0 --exp_name TRAIN_OF_BRANCH datasets OpticalFlow_TRAINSET datasets_test OpticalFlow_TESTSET

# Using the provided script
./train.sh
```

### Testing on Dataset
```bash
python test.py \
  -fop "data/test/hotshot" \
  -mop "checkpoints/optical_aug.pth" \
  -for "data/test/original/hotshot" \
  -mor "checkpoints/original_aug.pth" \
  -e "data/results/T2V/hotshot.csv" \
  -ef "data/results/frame/T2V/hotshot.csv" \
  -t 0.5
```

### Demo on Video
```bash
python demo.py \
  --path "demo_video/fake_sora/video.mp4" \
  --folder_original_path "frame/000000" \
  --folder_optical_flow_path "optical_result/000000" \
  -mop "checkpoints/optical.pth" \
  -mor "checkpoints/original.pth"
```

### Docker Examples
```bash
# Training with Docker
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  sacdalance/thesis-aigvdet:gpu \
  python3.11 train.py --gpus 0 --exp_name TRAIN_RGB datasets RGB_TRAINSET datasets_test RGB_TESTSET

# Testing with Docker
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  sacdalance/thesis-aigvdet:gpu \
  python3.11 test.py \
  -fop "data/test/hotshot" \
  -mop "checkpoints/optical_aug.pth" \
  -for "data/test/original/hotshot" \
  -mor "checkpoints/original_aug.pth" \
  -e "data/results/T2V/hotshot.csv"

# Demo with Docker
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/raft_model:/app/raft_model \
  -v $(pwd)/demo_video:/app/demo_video \
  sacdalance/thesis-aigvdet:gpu \
  python3.11 demo.py \
  --path "demo_video/fake_sora/video.mp4" \
  --folder_original_path "frame/000000" \
  --folder_optical_flow_path "optical_result/000000" \
  -mop "checkpoints/optical.pth" \
  -mor "checkpoints/original.pth"
```

## 🚀 Cross-Device Deployment Guide

### Option 1: Using Pre-built Docker Images (Recommended)

#### Prerequisites
```bash
# Install Docker
# Windows: Download Docker Desktop from docker.com
# Linux: sudo apt install docker.io docker-compose
# Mac: brew install docker docker-compose

# For GPU support (Linux/WSL2)
sudo apt install nvidia-docker2
sudo systemctl restart docker
```

#### Quick Setup on New Device
```bash
# 1. Clone the repository
git clone https://github.com/sacdalance/AIGVDet.git
cd AIGVDet                    # You are now INSIDE the repo folder

# 2. Create data directories INSIDE the repo
mkdir -p data/train data/test data/results checkpoints raft_model demo_video

# 3. Copy your downloaded data INTO the repo folders (REQUIRED BEFORE RUNNING):
#    - Training data → ./data/train/         (inside the AIGVDet repo)
#    - Test videos → ./data/test/            (inside the AIGVDet repo)
#    - Model weights → ./checkpoints/        (inside the AIGVDet repo)
#    - RAFT model → ./raft_model/            (inside the AIGVDet repo)
#    - Demo videos → ./demo_video/           (inside the AIGVDet repo)

# 4. Pull and run (GPU version) - ONLY AFTER DATA IS COPIED
docker pull sacdalance/thesis-aigvdet:gpu
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/raft_model:/app/raft_model \
  -v $(pwd)/demo_video:/app/demo_video \
  sacdalance/thesis-aigvdet:gpu bash

# 5. Inside container, run your experiments
python3.11 train.py --gpus 0 --exp_name my_experiment
```

#### Windows PowerShell Version
```powershell
# Clone and setup on new device
git clone https://github.com/sacdalance/AIGVDet.git
cd AIGVDet
mkdir data\train, data\test, data\results, checkpoints, raft_model, demo_video

# IMPORTANT: Copy your data to these folders FIRST, then:
docker pull sacdalance/thesis-aigvdet:gpu
docker run --gpus all -it --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/checkpoints:/app/checkpoints `
  -v ${PWD}/raft_model:/app/raft_model `
  -v ${PWD}/demo_video:/app/demo_video `
  sacdalance/thesis-aigvdet:gpu bash
```

### Option 2: Local Python Environment

#### Prerequisites
```bash
# Install Python 3.11
# Windows: Download from python.org
# Linux: sudo apt install python3.11 python3.11-venv
# Mac: brew install python@3.11

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
# Windows: iwr https://astral.sh/uv/install.ps1 | iex
```

#### Setup Steps
```bash
# 1. Clone repository on new device
git clone https://github.com/sacdalance/AIGVDet.git
cd AIGVDet

# 2. Create virtual environment and install dependencies
uv venv --python 3.11
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\Activate.ps1  # Windows

# 3. Install project
uv pip install -e .

# 4. Copy your data to appropriate folders:
#    - data/train/ (training frames)
#    - data/test/ (test videos) 
#    - checkpoints/ (model weights)
#    - raft_model/ (RAFT weights)
#    - demo_video/ (demo videos)

# 5. Run experiments
python train.py --gpus 0 --exp_name my_experiment
python test.py -fop "data/test/hotshot" -mop "checkpoints/optical_aug.pth" ...
python demo.py --path "demo_video/video.mp4" ...
```

### 📁 Expected Folder Structure After Setup
```
AIGVDet/                     # ← This is your cloned GitHub repo
├── pyproject.toml           # ← Project files from GitHub
├── train.py                 # ← Python scripts from GitHub  
├── test.py                  # ← Python scripts from GitHub
├── demo.py                  # ← Python scripts from GitHub
├── core/                    # ← Code folders from GitHub
├── networks/                # ← Code folders from GitHub
├── data/                    # ← YOU CREATE & FILL THIS
│   ├── train/               # ← Copy training data here
│   │   └── trainset_1/
│   │       ├── 0_real/
│   │       └── 1_fake/
│   ├── test/                # ← Copy test videos here
│   └── results/             # ← Output results go here
├── checkpoints/             # ← YOU CREATE & FILL THIS
│   ├── optical_aug.pth      # ← Copy model weights here
│   └── original_aug.pth
├── raft_model/              # ← YOU CREATE & FILL THIS  
│   └── raft-things.pth      # ← Copy RAFT model here
└── demo_video/              # ← YOU CREATE & FILL THIS
    ├── fake_sora/           # ← Copy demo videos here
    └── real/
```

**Key Point:** Everything goes INSIDE the AIGVDet folder you cloned from GitHub!

**Note:** The data folders (`data/`, `checkpoints/`, `raft_model/`, `demo_video/`) are automatically ignored by git (see `.gitignore`), so your large data files won't be committed to GitHub.

## 🔄 Updating Docker Images

### If you need to rebuild the Docker image (e.g., after dependency changes):

#### Build locally:
```bash
# Build GPU version
docker build -f Dockerfile.gpu -t sacdalance/thesis-aigvdet:gpu .

# Build CPU version  
docker build -f Dockerfile.cpu -t sacdalance/thesis-aigvdet:cpu .

# Test the build
docker run --rm sacdalance/thesis-aigvdet:gpu python3.11 --version
```

#### Push to Docker Hub (for maintainers):
```bash
# Login to Docker Hub
docker login

# Push updated images
docker push sacdalance/thesis-aigvdet:gpu
docker push sacdalance/thesis-aigvdet:cpu

# Or use the provided scripts
./push-docker.sh  # Linux/Mac
./push-docker.ps1 # Windows
```

#### For users - pull latest updates:
```bash
# Pull latest version
docker pull sacdalance/thesis-aigvdet:gpu

# Force rebuild without cache if needed
docker build --no-cache -f Dockerfile.gpu -t sacdalance/thesis-aigvdet:gpu .
```

### 📦 Data Transfer to New Device

Since you've already downloaded all the data, you need to transfer it to your new device:

#### Option 1: Cloud Storage (Recommended)
```bash
# Upload from current device to cloud (Google Drive, OneDrive, etc.)
# Then download on new device to the appropriate folders

# Or use cloud sync folders:
# 1. Put data in cloud sync folder on current device
# 2. Access from new device once synced
```

#### Option 2: External Drive/USB
```bash
# Copy from current device to external drive:
# - data/ folder (training + test data)
# - checkpoints/ folder (model weights)  
# - raft_model/ folder (RAFT weights)
# - demo_video/ folder (demo videos)

# Then copy to new device after cloning repo
```

#### Option 3: Network Transfer
```bash
# If both devices are on same network:
# Use scp, rsync, or network sharing to transfer data folders
```

### 🔧 Troubleshooting

#### GPU Issues
```bash
# Check GPU availability
nvidia-smi
docker run --gpus all nvidia/cuda:11.7-base nvidia-smi

# If GPU not detected in Docker:
# 1. Install nvidia-docker2
# 2. Restart Docker service
# 3. Use --gpus all flag
```

#### Permission Issues (Linux)
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again
```

#### Memory Issues
```bash
# For large datasets, increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory → 8GB+
```

### 📋 Quick Deployment Checklist (IN ORDER!)
- [ ] Docker installed (with GPU support if needed)
- [ ] Repository cloned: `git clone https://github.com/sacdalance/AIGVDet.git`
- [ ] Data folders created: `mkdir data/train data/test checkpoints raft_model demo_video`
- [ ] **DATA TRANSFER COMPLETE** ⚠️ (Required before Docker run):
  - [ ] Training data copied to `./data/train/`
  - [ ] Test videos copied to `./data/test/`
  - [ ] Model weights copied to `./checkpoints/`
  - [ ] RAFT model copied to `./raft_model/`
  - [ ] Demo videos copied to `./demo_video/`
- [ ] Docker image pulled: `docker pull sacdalance/thesis-aigvdet:gpu`
- [ ] Test run: `docker run --gpus all -it sacdalance/thesis-aigvdet:gpu python3.11 --version`

**⚠️ CRITICAL:** The `-v $(pwd)/data:/app/data` flag mounts your local folders into the container. If the data isn't there, the container will see empty folders!

## Notes
- **Research use only**: The datasets are only allowed for research purposes
- **Citation required**: If you use this work, please cite the PRCV 2024 paper
- **Contact**: For questions, contact lyan924@cuc.edu.cn
- **Docker Hub**: Pre-built images available at `sacdalance/thesis-aigvdet:gpu` and `sacdalance/thesis-aigvdet:cpu`