# AIGVDet Docker Quick Reference

## Pull from Docker Hub

```bash
# Pull GPU version
docker pull sacdalance/thesis-aigvdet:gpu

# Pull CPU version
docker pull sacdalance/thesis-aigvdet:cpu
```

## Build Images

### Windows (PowerShell)
```powershell
# Build both
.\build-docker.ps1 all

# Build GPU only
.\build-docker.ps1 gpu

# Build CPU only
.\build-docker.ps1 cpu
```

### Linux/Mac (Bash)
```bash
# Build both
./build-docker.sh all

# Build GPU only
./build-docker.sh gpu

# Build CPU only
./build-docker.sh cpu
```

## Run with Docker Compose (Recommended)

```bash
# GPU version
docker-compose up aigvdet-gpu

# CPU version
docker-compose up aigvdet-cpu

# Build and run
docker-compose up --build aigvdet-gpu

# Run in background
docker-compose up -d aigvdet-gpu
```

## Run with Docker Run

### GPU Version (Windows PowerShell)
```powershell
docker run --gpus all -it --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/checkpoints:/app/checkpoints `
  -v ${PWD}/raft_model:/app/raft_model `
  -p 6006:6006 `
  sacdalance/thesis-aigvdet:gpu `
  python3.11 train.py --gpus 0 --exp_name my_experiment
```

### CPU Version (Windows PowerShell)
```powershell
docker run -it --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/checkpoints:/app/checkpoints `
  -p 6006:6006 `
  sacdalance/thesis-aigvdet:cpu `
  python train.py --exp_name my_experiment
```

### GPU Version (Linux/Mac)
```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/raft_model:/app/raft_model \
  -p 6006:6006 \
  sacdalance/thesis-aigvdet:gpu \
  python3.11 train.py --gpus 0 --exp_name my_experiment
```

## Common Tasks

### Training RGB Branch
```bash
docker-compose run aigvdet-gpu python3.11 train.py --gpus 0 --exp_name TRAIN_RGB datasets RGB_TRAINSET datasets_test RGB_TESTSET
```

### Training Optical Flow Branch
```bash
docker-compose run aigvdet-gpu python3.11 train.py --gpus 0 --exp_name TRAIN_OF datasets OpticalFlow_TRAINSET datasets_test OpticalFlow_TESTSET
```

### Testing
```bash
docker-compose run aigvdet-gpu python3.11 test.py \
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
docker-compose run aigvdet-gpu python3.11 demo.py \
  --path "demo_video/video.mp4" \
  --folder_original_path "frame/000000" \
  --folder_optical_flow_path "optical_result/000000" \
  -mop "checkpoints/optical.pth" \
  -mor "checkpoints/original.pth"
```

### Interactive Shell
```bash
# GPU
docker-compose run aigvdet-gpu /bin/bash

# CPU
docker-compose run aigvdet-cpu /bin/bash
```

### TensorBoard
```bash
# Access at http://localhost:6006
docker-compose up aigvdet-gpu
```

## Cleanup

```bash
# Stop containers
docker-compose down

# Remove images
docker rmi aigvdet:gpu aigvdet:cpu

# Full cleanup
docker system prune -a
```

## File Structure

```
AIGVDet/
├── Dockerfile.gpu           # GPU version Dockerfile
├── Dockerfile.cpu           # CPU version Dockerfile
├── docker-compose.yml       # Orchestration config
├── .dockerignore           # Files to exclude from build
├── build-docker.sh         # Linux/Mac build script
├── build-docker.ps1        # Windows build script
├── DOCKER_USAGE.md         # Detailed usage guide
└── DOCKER_QUICKREF.md      # This file
```

## Volumes Mounted

- `./data` → `/app/data` - Training/test data
- `./checkpoints` → `/app/checkpoints` - Model weights
- `./raft_model` → `/app/raft_model` - RAFT model
- `./logs` → `/app/logs` - Training logs

## Ports Exposed

- `6006` - TensorBoard (GPU container)
- `6007` - TensorBoard (CPU container, in compose)

## Image Details

| Version | Base Image | Size | Python | PyTorch | CUDA |
|---------|-----------|------|--------|---------|------|
| GPU | nvidia/cuda:11.7.1-cudnn8-runtime | ~8-10GB | 3.11 | 2.0.0+cu117 | 11.7 |
| CPU | python:3.11-slim | ~2-3GB | 3.11 | 2.0.0+cpu | N/A |
