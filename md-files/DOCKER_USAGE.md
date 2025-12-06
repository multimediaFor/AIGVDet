# Docker Usage Guide for AIGVDet

This guide explains how to build and run AIGVDet using Docker with both CPU and GPU support.

## Prerequisites

### For CPU Version
- Docker installed
- At least 8GB RAM recommended

### For GPU Version
- Docker installed
- NVIDIA Docker runtime (nvidia-docker2)
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed

## Quick Start

### Build Docker Images

**Build GPU version:**
```bash
docker build -f Dockerfile.gpu -t aigvdet:gpu .
```

**Build CPU version:**
```bash
docker build -f Dockerfile.cpu -t aigvdet:cpu .
```

### Run Containers

**Run GPU version:**
```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/raft_model:/app/raft_model \
  -p 6006:6006 \
  aigvdet:gpu
```

**Run CPU version:**
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/raft_model:/app/raft_model \
  -p 6006:6006 \
  aigvdet:cpu
```

## Using Docker Compose

Docker Compose simplifies running the containers with all necessary configurations.

**Start GPU service:**
```bash
docker-compose up aigvdet-gpu
```

**Start CPU service:**
```bash
docker-compose up aigvdet-cpu
```

**Build and start:**
```bash
docker-compose up --build aigvdet-gpu
```

## Training Examples

### GPU Training - RGB Branch
```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  aigvdet:gpu \
  python3.11 train.py --gpus 0 --exp_name TRAIN_RGB_BRANCH \
  datasets RGB_TRAINSET datasets_test RGB_TESTSET
```

### GPU Training - Optical Flow Branch
```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  aigvdet:gpu \
  python3.11 train.py --gpus 0 --exp_name TRAIN_OF_BRANCH \
  datasets OpticalFlow_TRAINSET datasets_test OpticalFlow_TESTSET
```

### CPU Training
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  aigvdet:cpu \
  python train.py --exp_name TRAIN_RGB_BRANCH \
  datasets RGB_TRAINSET datasets_test RGB_TESTSET
```

## Testing

### Test on Dataset
```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  aigvdet:gpu \
  python3.11 test.py \
  -fop "data/test/hotshot" \
  -mop "checkpoints/optical_aug.pth" \
  -for "data/test/original/hotshot" \
  -mor "checkpoints/original_aug.pth" \
  -e "data/results/T2V/hotshot.csv" \
  -ef "data/results/frame/T2V/hotshot.csv" \
  -t 0.5
```

## Demo on Video

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/raft_model:/app/raft_model \
  -v $(pwd)/demo_video:/app/demo_video \
  aigvdet:gpu \
  python3.11 demo.py \
  --path "demo_video/fake_sora/video.mp4" \
  --folder_original_path "frame/000000" \
  --folder_optical_flow_path "optical_result/000000" \
  -mop "checkpoints/optical.pth" \
  -mor "checkpoints/original.pth"
```

## TensorBoard

Access TensorBoard by visiting `http://localhost:6006` in your browser after starting the container.

To explicitly run TensorBoard:
```bash
docker run --gpus all -it --rm \
  -v $(pwd)/logs:/app/logs \
  -p 6006:6006 \
  aigvdet:gpu \
  tensorboard --logdir=/app/logs --host=0.0.0.0 --port=6006
```

## Interactive Shell

**GPU container:**
```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  aigvdet:gpu \
  /bin/bash
```

**CPU container:**
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  aigvdet:cpu \
  /bin/bash
```

## Volume Mounts Explained

- `-v $(pwd)/data:/app/data` - Mount your training/test data
- `-v $(pwd)/checkpoints:/app/checkpoints` - Mount model checkpoints
- `-v $(pwd)/raft_model:/app/raft_model` - Mount RAFT model weights
- `-v $(pwd)/logs:/app/logs` - Mount training logs for TensorBoard

## PowerShell Commands (Windows)

On Windows PowerShell, use `${PWD}` instead of `$(pwd)`:

```powershell
docker run --gpus all -it --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/checkpoints:/app/checkpoints `
  -v ${PWD}/raft_model:/app/raft_model `
  -p 6006:6006 `
  aigvdet:gpu
```

## Troubleshooting

### GPU not detected
Ensure nvidia-docker2 is installed:
```bash
# Install nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu22.04 nvidia-smi
```

### Out of memory
Increase shared memory:
```bash
docker run --gpus all -it --rm --shm-size=8g ...
```

### Permission issues
If you encounter permission issues with mounted volumes:
```bash
docker run --gpus all -it --rm --user $(id -u):$(id -g) ...
```

## Image Sizes

- GPU image: ~8-10GB (includes CUDA runtime)
- CPU image: ~2-3GB (lighter weight)

## Cleaning Up

Remove containers:
```bash
docker-compose down
```

Remove images:
```bash
docker rmi aigvdet:gpu aigvdet:cpu
```

Clean up all stopped containers and unused images:
```bash
docker system prune -a
```
