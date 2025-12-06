# Docker Setup Guide for VM

## Quick Start

### 1. Automated Data Deployment (Recommended for Vast.ai)
Since the dataset is large (27GB+), we use a startup script to download it directly to the VM.

1.  **Start your instance** (Vast.ai, AWS, etc.).
2.  **Open a terminal** in the container.
3.  **Run the download script**:
    ```bash
    # This will download the dataset from Google Drive and unzip it
    chmod +x download_data.sh
    ./download_data.sh
    ```
    *Note: The script defaults to the project's Google Drive ID. You can pass a different ID if needed: `./download_data.sh <YOUR_FILE_ID>`*

### 2. Manual Data Setup (Local Development)
If you are running locally or want to set up manually:

```bash
# Run the setup script
python setup_data_structure.py

# Or manually create:
mkdir -p data/train/trainset_1/0_real/video_00000
mkdir -p data/train/trainset_1/1_fake/video_00000
mkdir -p data/val/val_set_1/0_real/video_00000
mkdir -p data/val/val_set_1/1_fake/video_00000
```

### 2. Add Your Training Data
Place your extracted frames in:
- `data/train/trainset_1/0_real/` - Real video frames
- `data/train/trainset_1/1_fake/` - Fake video frames

Each video should be in its own directory with frames named sequentially:
```
data/train/trainset_1/0_real/
├── video_00000/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── video_00001/
│   └── ...
```

### 3. Ensure .env File Exists
Your `.env` file should contain your wandb API key:
```bash
WANDB="your_api_key_here"
```

### 4. Run with Docker

#### Using docker-compose (Recommended):
```bash
# GPU version
docker-compose up aigvdet-gpu

# CPU version
docker-compose up aigvdet-cpu
```

#### Using docker run:
```bash
# GPU version
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/.env:/app/.env:ro \
  --env-file .env \
  sacdalance/thesis-aigvdet:gpu \
  python train.py --exp_name my_experiment

# CPU version
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/.env:/app/.env:ro \
  --env-file .env \
  sacdalance/thesis-aigvdet:cpu \
  python train.py --exp_name my_experiment
```

#### On Windows PowerShell:
```powershell
# GPU version
docker run --gpus all `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/checkpoints:/app/checkpoints `
  -v ${PWD}/.env:/app/.env:ro `
  --env-file .env `
  sacdalance/thesis-aigvdet:gpu `
  python train.py --exp_name my_experiment

# CPU version
docker run `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/checkpoints:/app/checkpoints `
  -v ${PWD}/.env:/app/.env:ro `
  --env-file .env `
  sacdalance/thesis-aigvdet:cpu `
  python train.py --exp_name my_experiment
```

### 5. Verify Setup Before Training

Run this to check if data is properly mounted:
```bash
docker run -v $(pwd)/data:/app/data sacdalance/thesis-aigvdet:gpu ls -la /app/data/train/trainset_1/
```

## Troubleshooting

### Error: "No such file or directory: '/app/data/train/trainset_1'"

**Solutions:**
1. **Create the directory structure** (see step 1 above)
2. **Verify volume mount** - ensure data directory exists locally
3. **Check permissions** - ensure Docker can read the data directory

### Error: "WANDB API key not found"

**Solution:**
- Create `.env` file with: `WANDB="your_api_key_here"`
- Make sure it's mounted: `-v $(pwd)/.env:/app/.env:ro`
- Or pass directly: `-e WANDB_API_KEY=your_api_key`

### No GPU detected

**Solutions:**
1. Install nvidia-docker2:
   ```bash
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```
2. Use `--gpus all` flag in docker run
3. Or use CPU version instead

### Permission Denied

**Solution:**
```bash
# Fix data directory permissions
chmod -R 755 data/
chmod -R 755 checkpoints/
```

## Data Download Instructions

See `DATA_SETUP.md` for complete instructions on downloading the training dataset from Baiduyun.

Quick summary:
1. Download from: https://pan.baidu.com/s/17xmDyFjtcmNsoxmUeImMTQ?pwd=ra95
2. Extract to `data/train/trainset_1/`
3. Ensure structure matches above format

## Testing Without Full Dataset

For quick testing, you can create a minimal dataset:
1. Create a few sample frames (any images will do)
2. Place in `data/train/trainset_1/0_real/video_00000/`
3. Name them `00000.png`, `00001.png`, etc.
4. Copy to `1_fake/` directory as well
5. Run training to verify setup works

## Training Options

```bash
# Basic training
python train.py --exp_name my_experiment

# With custom settings
python train.py \
  --exp_name my_experiment \
  --batch_size 32 \
  --nepoch 50 \
  --lr 0.0001

# Continue from checkpoint
python train.py \
  --exp_name my_experiment \
  --continue_train \
  --epoch 10
```
