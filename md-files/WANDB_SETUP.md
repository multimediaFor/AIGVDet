# Weights & Biases (W&B) Integration Guide

## What is W&B?
Weights & Biases tracks your training runs, logs metrics, and **stores model checkpoints in the cloud** so you can access them from any device.

## Setup Instructions

### 1. Install wandb
```bash
# Local installation
pip install wandb

# Or with Docker (already included in requirements.txt)
# Just rebuild the Docker image
docker build -f Dockerfile.gpu-alt -t sacdalance/thesis-aigvdet:gpu .
```

### 2. Create W&B Account
1. Go to https://wandb.ai/signup
2. Sign up (free for personal use)
3. Get your API key from https://wandb.ai/authorize

### 3. Login to W&B

**On local machine:**
```bash
wandb login
# Paste your API key when prompted
```

**On Vast.ai or remote server:**
```bash
# Option 1: Interactive login
wandb login

# Option 2: Use API key directly
export WANDB_API_KEY="your-api-key-here"
# Or add to Docker run command:
docker run --gpus all -e WANDB_API_KEY="your-key" ...
```

### 4. Train with W&B Tracking

```bash
# Train normally - W&B will automatically track
python train.py --gpus 0 --exp_name TRAIN_RGB datasets trainset_1_RGB datasets_test val_set_1_RGB

# With Docker on Vast.ai
docker run --gpus all --shm-size=8g -it --rm \
  -e WANDB_API_KEY="your-api-key" \
  -v /workspace/data:/app/data \
  -v /workspace/checkpoints:/app/checkpoints \
  sacdalance/thesis-aigvdet:gpu \
  python3.11 train.py --gpus 0 --exp_name TRAIN_RGB datasets trainset_1_RGB datasets_test val_set_1_RGB
```

## What W&B Tracks

### Metrics Logged:
- **Training loss** - Every batch
- **Validation metrics** - Every epoch
  - Accuracy (ACC)
  - Average Precision (AP)
  - AUC (Area Under Curve)
  - TPR (True Positive Rate)
  - TNR (True Negative Rate)
- **System metrics** - GPU usage, CPU, memory

### Model Checkpoints:
- **Automatically uploads** `model_epoch_best.pth` to W&B cloud
- **Access from anywhere** - Download checkpoints from any device
- **Version control** - All checkpoint versions saved

## Accessing Your Results

### View Training Dashboard:
1. Go to https://wandb.ai/
2. Navigate to your project: `aigvdet-training`
3. Click on your run (e.g., `TRAIN_RGB`)
4. View real-time metrics, charts, and system stats

### Download Checkpoints:
```python
# From any device with Python
import wandb

# Login
wandb.login()

# Download checkpoint
api = wandb.Api()
run = api.run("your-username/aigvdet-training/run-id")
artifact = run.use_artifact('TRAIN_RGB_best_model:latest')
artifact_dir = artifact.download()

# The checkpoint will be in: artifact_dir/model_epoch_best.pth
```

**Or via Web UI:**
1. Go to your run page
2. Click "Artifacts" tab
3. Click on `TRAIN_RGB_best_model`
4. Click "Download" button

## Benefits for Vast.ai Training

### Problem W&B Solves:
- ❌ **Without W&B**: Must manually download checkpoints before stopping Vast.ai instance
- ✅ **With W&B**: Checkpoints automatically saved to cloud, accessible from anywhere

### Workflow:
```bash
# 1. Start training on Vast.ai
docker run --gpus all -e WANDB_API_KEY="your-key" ... python train.py ...

# 2. Training runs (checkpoints auto-upload to W&B cloud)

# 3. Stop Vast.ai instance (no need to manually download!)

# 4. Later, on your local machine:
wandb artifact get your-username/aigvdet-training/TRAIN_RGB_best_model:latest
# Checkpoint downloaded to local machine!
```

## Configuration Options

### Disable W&B (if needed):
```bash
# Set environment variable
export WANDB_MODE=disabled

# Or in code (add to train.py):
os.environ["WANDB_MODE"] = "disabled"
```

### Change Project Name:
Edit `train.py` line 43:
```python
wandb.init(
    project="your-custom-project-name",  # Change this
    name=cfg.exp_name,
    ...
)
```

### Resume Training:
W&B automatically handles resume if you use `continue_train True`:
```bash
python train.py --gpus 0 --exp_name TRAIN_RGB continue_train True epoch latest
```

## Cost
- **Free tier**: Unlimited personal projects, 100GB storage
- **Teams tier**: $50/user/month for collaboration
- For academic research, you can apply for free team accounts

## Troubleshooting

### "wandb: ERROR api_key not configured"
```bash
# Set API key
export WANDB_API_KEY="your-key"
# Or login
wandb login
```

### Checkpoint upload fails
```bash
# Check internet connection
# Check W&B storage quota (free tier = 100GB)
# Verify checkpoint file exists
ls -lh data/exp/TRAIN_RGB/ckpt/model_epoch_best.pth
```

### Disable W&B temporarily
```bash
export WANDB_MODE=offline  # Run offline, sync later
# Or
export WANDB_MODE=disabled  # Completely disable
```

## Example Output

When training starts:
```
Setting up TensorBoard...
✓ Logs will be saved to: data/exp/TRAIN_RGB

Initializing Weights & Biases...
wandb: Currently logged in as: your-username (use `wandb login --relogin` to force relogin)
wandb: Tracking run with wandb version 0.16.0
wandb: Run data is saved locally in data/exp/TRAIN_RGB/wandb
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run TRAIN_RGB
wandb: ⭐️ View project at https://wandb.ai/your-username/aigvdet-training
wandb: 🚀 View run at https://wandb.ai/your-username/aigvdet-training/runs/abc123
✓ W&B tracking enabled: https://wandb.ai/your-username/aigvdet-training/runs/abc123
```

Click the URL to view your training in real-time!

## Summary

**With W&B integration:**
1. ✅ Track training metrics in real-time from any device
2. ✅ Automatically save checkpoints to cloud storage
3. ✅ No need to manually download before stopping Vast.ai
4. ✅ Compare multiple training runs easily
5. ✅ Share results with collaborators

**Setup is simple:**
```bash
pip install wandb
wandb login
python train.py  # W&B automatically tracks!
```
