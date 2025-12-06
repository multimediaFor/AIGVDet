# AIGVDet Docker Documentation Index

Welcome to the AIGVDet Docker implementation! This project now supports containerized deployment with both CPU and GPU configurations.

## 📚 Documentation Overview

### Quick Start
- **[DOCKER_QUICKREF.md](DOCKER_QUICKREF.md)** - Quick reference for common Docker commands
  - Build commands
  - Run commands
  - Common tasks
  - One-line examples

### Detailed Guides
- **[DOCKER_USAGE.md](DOCKER_USAGE.md)** - Comprehensive usage guide
  - Prerequisites and setup
  - Detailed build instructions
  - Training and testing examples
  - Troubleshooting
  - Volume management
  - TensorBoard access

- **[DOCKER_COMPARISON.md](DOCKER_COMPARISON.md)** - CPU vs GPU comparison
  - Performance benchmarks
  - Cost analysis
  - Resource requirements
  - Use case recommendations

### Main Documentation
- **[README.md](README.md)** - Project overview and setup (now includes Docker section)
- **[MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md)** - UV migration details

## 🎯 Getting Started in 3 Steps

### 1. Choose Your Version
- **GPU** - For training and fast inference (requires NVIDIA GPU + nvidia-docker)
- **CPU** - For testing and CPU-only environments

### 2. Build the Image

**Windows (PowerShell):**
```powershell
.\build-docker.ps1 gpu   # or 'cpu'
```

**Linux/Mac:**
```bash
./build-docker.sh gpu    # or 'cpu'
```

**Using Make:**
```bash
make build-gpu           # or 'build-cpu'
```

### 3. Run the Container

**With Docker Compose (Recommended):**
```bash
docker-compose up aigvdet-gpu
```

**Direct Docker Run:**
```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  aigvdet:gpu
```

## 📁 File Structure

```
AIGVDet/
├── 🐳 Docker Configuration
│   ├── Dockerfile.gpu           # GPU version
│   ├── Dockerfile.cpu           # CPU version
│   ├── docker-compose.yml       # Compose configuration
│   └── .dockerignore           # Build exclusions
│
├── 🔧 Build Scripts
│   ├── build-docker.ps1        # Windows script
│   ├── build-docker.sh         # Linux/Mac script
│   └── Makefile                # Make targets
│
├── 📖 Documentation
│   ├── DOCKER_INDEX.md         # This file
│   ├── DOCKER_QUICKREF.md      # Quick reference
│   ├── DOCKER_USAGE.md         # Detailed guide
│   ├── DOCKER_COMPARISON.md    # CPU vs GPU
│   └── README.md               # Main README
│
└── 🐍 Project Files
    ├── pyproject.toml          # UV project config
    ├── core/                   # Core modules
    ├── networks/               # Network definitions
    ├── train.py               # Training script
    ├── test.py                # Testing script
    └── demo.py                # Demo script
```

## 🎓 Common Use Cases

### Training a Model
```bash
# GPU training
docker-compose run aigvdet-gpu python3.11 train.py \
  --gpus 0 \
  --exp_name TRAIN_RGB \
  datasets RGB_TRAINSET \
  datasets_test RGB_TESTSET
```

### Running Tests
```bash
# Test on dataset
docker-compose run aigvdet-gpu python3.11 test.py \
  -fop "data/test/hotshot" \
  -mop "checkpoints/optical_aug.pth" \
  -for "data/test/original/hotshot" \
  -mor "checkpoints/original_aug.pth" \
  -e "results/hotshot.csv"
```

### Demo on Video
```bash
# Process a single video
docker-compose run aigvdet-gpu python3.11 demo.py \
  --path "demo_video/video.mp4" \
  --folder_original_path "frame/output" \
  --folder_optical_flow_path "optical_result/output" \
  -mop "checkpoints/optical.pth" \
  -mor "checkpoints/original.pth"
```

### Interactive Development
```bash
# Open bash shell in container
docker-compose run aigvdet-gpu /bin/bash

# Or using Make
make shell-gpu
```

## 🔍 Quick Command Reference

| Task | Command |
|------|---------|
| Build GPU | `.\build-docker.ps1 gpu` or `make build-gpu` |
| Build CPU | `.\build-docker.ps1 cpu` or `make build-cpu` |
| Run GPU | `docker-compose up aigvdet-gpu` |
| Run CPU | `docker-compose up aigvdet-cpu` |
| Shell GPU | `make shell-gpu` |
| Shell CPU | `make shell-cpu` |
| Clean up | `make clean` |
| View logs | `docker-compose logs -f` |

## 🆘 Need Help?

1. **Quick commands?** → [DOCKER_QUICKREF.md](DOCKER_QUICKREF.md)
2. **Detailed setup?** → [DOCKER_USAGE.md](DOCKER_USAGE.md)
3. **CPU or GPU?** → [DOCKER_COMPARISON.md](DOCKER_COMPARISON.md)
4. **Project info?** → [README.md](README.md)

## 💡 Tips

- **First time?** Start with [DOCKER_QUICKREF.md](DOCKER_QUICKREF.md)
- **Troubleshooting?** Check [DOCKER_USAGE.md](DOCKER_USAGE.md) troubleshooting section
- **Performance tuning?** See [DOCKER_COMPARISON.md](DOCKER_COMPARISON.md)
- **Windows user?** All PowerShell commands are in the guides

## 🎯 What's Different from Regular Installation?

| Aspect | Regular Setup | Docker Setup |
|--------|--------------|--------------|
| Installation | Manual dependencies | One `docker build` command |
| Reproducibility | Varies by system | Identical everywhere |
| GPU Setup | Manual CUDA install | Pre-configured in image |
| Isolation | System-wide packages | Containerized environment |
| Portability | Machine-specific | Runs anywhere Docker runs |
| Cleanup | Manual uninstall | `docker rmi` |

## 🚀 Ready to Start?

1. Pick your documentation:
   - **Quick start** → [DOCKER_QUICKREF.md](DOCKER_QUICKREF.md)
   - **Full guide** → [DOCKER_USAGE.md](DOCKER_USAGE.md)

2. Choose your version:
   - **GPU** → For training and production
   - **CPU** → For testing and development

3. Build and run:
   ```bash
   .\build-docker.ps1 gpu
   docker-compose up aigvdet-gpu
   ```

Happy containerizing! 🐳
