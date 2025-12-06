# Docker Configuration Comparison

## Quick Comparison

| Feature | CPU Version | GPU Version |
|---------|------------|-------------|
| **Base Image** | python:3.11-slim | nvidia/cuda:11.7.1-cudnn8-runtime |
| **Image Size** | ~2-3 GB | ~8-10 GB |
| **PyTorch** | 2.0.0+cpu | 2.0.0+cu117 |
| **Build Time** | ~5-10 min | ~10-20 min |
| **Training Speed** | Slow (baseline) | 10-100x faster |
| **Memory Required** | 4-8 GB RAM | 8+ GB RAM + GPU VRAM |
| **Hardware Required** | Any CPU | NVIDIA GPU + nvidia-docker |
| **Use Case** | Testing, inference on small data | Training, production inference |
| **Cost** | Low | High (GPU required) |

## When to Use CPU Version

✅ **Good for:**
- Initial testing and development
- Small-scale inference
- Environments without GPU access
- Budget-constrained deployments
- CI/CD pipelines for testing

❌ **Not recommended for:**
- Training large models
- Processing high-resolution videos
- Batch processing many videos
- Production workloads with time constraints

## When to Use GPU Version

✅ **Good for:**
- Training models
- Large-scale inference
- Video processing pipelines
- Production deployments
- Research and experimentation

❌ **Not recommended for:**
- Simple testing
- Environments without NVIDIA GPU
- Cost-sensitive deployments where speed isn't critical

## Resource Requirements

### CPU Version
```yaml
Minimum:
  RAM: 4 GB
  Storage: 10 GB
  CPU: 2 cores

Recommended:
  RAM: 8 GB
  Storage: 20 GB
  CPU: 4+ cores
```

### GPU Version
```yaml
Minimum:
  RAM: 8 GB
  GPU VRAM: 6 GB
  Storage: 20 GB
  GPU: NVIDIA GPU with CUDA 11.7 support

Recommended:
  RAM: 16 GB
  GPU VRAM: 12+ GB
  Storage: 50 GB
  GPU: NVIDIA RTX 3080 or better
```

## Performance Comparison

### Training (1000 iterations)
| Hardware | Time | Relative Speed |
|----------|------|----------------|
| CPU (Intel i7) | ~4-6 hours | 1x |
| GPU (GTX 1080 Ti) | ~20-30 min | 10-15x |
| GPU (RTX 3090) | ~10-15 min | 20-30x |
| GPU (A100) | ~5-8 min | 40-60x |

### Inference (Single Video)
| Hardware | Time | Relative Speed |
|----------|------|----------------|
| CPU (Intel i7) | ~5-10 min | 1x |
| GPU (GTX 1080 Ti) | ~30-60 sec | 5-10x |
| GPU (RTX 3090) | ~15-30 sec | 10-20x |
| GPU (A100) | ~10-20 sec | 15-30x |

*Note: Times vary based on video resolution, length, and model complexity*

## Build Time Comparison

### Initial Build
| Version | Download Size | Build Time |
|---------|--------------|------------|
| CPU | ~500 MB | 5-10 min |
| GPU | ~2-3 GB | 10-20 min |

### Rebuild (after changes)
| Version | Time |
|---------|------|
| CPU | 1-2 min |
| GPU | 2-5 min |

## Docker Compose Configuration

The included `docker-compose.yml` is configured with optimal defaults:

### CPU Container
```yaml
Resources:
  - 4 GB shared memory
  - All CPU cores available
  - Port 6007 for TensorBoard
```

### GPU Container
```yaml
Resources:
  - 8 GB shared memory
  - All NVIDIA GPUs available
  - Port 6006 for TensorBoard
  - CUDA environment configured
```

## Cost Comparison (Cloud Deployment)

### AWS EC2 Approximate Hourly Costs
| Instance Type | Specs | Cost/Hour | Best For |
|--------------|-------|-----------|----------|
| t3.2xlarge | 8 vCPU, 32 GB RAM | $0.33 | CPU Testing |
| p3.2xlarge | V100 GPU, 16 GB VRAM | $3.06 | GPU Training |
| p4d.24xlarge | 8x A100 GPUs | $32.77 | Large-scale Training |

*Prices as of 2024, may vary by region*

## Recommendations by Use Case

### Research & Development
- **Start with**: GPU version
- **Why**: Faster iteration, better for experimentation
- **Fallback**: CPU for quick tests

### Production Inference
- **Small scale**: CPU version
- **Large scale**: GPU version
- **Why**: Cost vs. speed tradeoff

### CI/CD Testing
- **Use**: CPU version
- **Why**: No GPU required, faster builds, lower cost

### Training
- **Always use**: GPU version
- **Why**: CPU training is impractically slow

## Environment Variables

Both versions support these environment variables:

```bash
# PyTorch settings
CUDA_VISIBLE_DEVICES=0,1  # GPU only: Select GPUs
OMP_NUM_THREADS=4         # CPU only: Thread count

# Application settings
PYTHONUNBUFFERED=1        # Real-time logging
PYTHONDONTWRITEBYTECODE=1 # Smaller image size

# UV package manager
UV_SYSTEM_PYTHON=1        # Use system Python
```

## Summary

Choose your version based on:

1. **Hardware Available**
   - No GPU? → CPU version
   - Have GPU? → GPU version

2. **Use Case**
   - Testing/Development? → Start with CPU, move to GPU
   - Training? → GPU version mandatory
   - Production inference? → Depends on scale

3. **Budget**
   - Limited? → CPU version
   - Performance critical? → GPU version

4. **Time Constraints**
   - Quick results needed? → GPU version
   - Can wait? → CPU version acceptable

For most users working on AI-Generated Video Detection, the **GPU version is recommended** for any serious work beyond initial exploration.
