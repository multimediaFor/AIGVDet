## AIGVDet
An official implementation code for paper "AI-Generated Video Detection via Spatial-Temporal Anomaly Learning", PRCV 2024. This repo will provide <B>codes, trained weights, and our training datasets</B>. 

## 🐳 Docker Support
Now supports Docker with both CPU and GPU versions! See [DOCKER_QUICKREF.md](DOCKER_QUICKREF.md) for quick commands or [DOCKER_USAGE.md](DOCKER_USAGE.md) for detailed instructions.

**Quick Start with Docker:**
```bash
# Build and run GPU version
docker-compose up --build aigvdet-gpu

# Or build manually
./build-docker.ps1 gpu  # Windows
./build-docker.sh gpu   # Linux/Mac
```

## Network Architecture
<center> <img src="fig/NetworkArchitecture.png" alt="architecture"/> </center>

## Dataset 
- Download the preprocessed training frames from
[Baiduyun Link](https://pan.baidu.com/s/17xmDyFjtcmNsoxmUeImMTQ?pwd=ra95) (extract code: ra95).
- Download the test videos from [Google Drive](https://drive.google.com/drive/folders/1D84SRWEJ8BK8KBpTMuGi3BUM80mW_dKb?usp=sharing).

**You are allowed to use the datasets for <B>research purpose only</B>.**

## Training
- Prepare for the training datasets.
```
└─data
   ├── train
   │   └── trainset_1
   │       ├── 0_real
   │       │   ├── video_00000
   │       │   │    ├── 00000.png
   │       │   │    └── ...
   │       │   └── ...
   │       └── 1_fake
   │           ├── video_00000
   │           │    ├── 00000.png
   │           │    └── ...
   │           └── ...
   ├── val
   │   └── val_set_1
   │       ├── 0_real
   │       │   ├── video_00000
   │       │   │    ├── 00000.png
   │       │   │    └── ...
   │       │   └── ...
   │       └── 1_fake
   │           ├── video_00000
   │           │    ├── 00000.png
   │           │    └── ...
   │           └── ...
   └── test
       └── testset_1
           ├── 0_real
           │   ├── video_00000
           │   │    ├── 00000.png
           │   │    └── ...
           │   └── ...
           └── 1_fake
               ├── video_00000
               │    ├── 00000.png
               │    └── ...
               └── ...

```
- Modify configuration file in `core/utils1/config.py`.
- Train the Spatial Domain Detector with the RGB frames.
```
python train.py --gpus 0 --exp_name TRAIN_RGB_BRANCH datasets RGB_TRAINSET datasets_test RGB_TESTSET
```
- Train the Optical Flow Detector with the optical flow frames.
```
python train.py --gpus 0 --exp_name TRAIN_OF_BRANCH datasets OpticalFlow_TRAINSET datasets_test OpticalFlow_TESTSET
```
## Testing
Download the weights from [Google Drive Link](https://drive.google.com/drive/folders/18JO_YxOEqwJYfbVvy308XjoV-N6fE4yP?usp=share_link) and move it into the `checkpoints/`.

- Run on a dataset.
Prepare the RGB frames and the optical flow maps.
```
python test.py -fop "data/test/hotshot" -mop "checkpoints/optical_aug.pth" -for "data/test/original/hotshot" -mor "checkpoints/original_aug.pth" -e "data/results/T2V/hotshot.csv" -ef "data/results/frame/T2V/hotshot.csv" -t 0.5
```
- Run on a video.
Download the RAFT model weights from [Google Drive Link](https://drive.google.com/file/d/1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM/view) and move it into the `raft_model/`.
```
python demo.py --use_cpu --path "video/000000.mp4" --folder_original_path "frame/000000" --folder_optical_flow_path "optical_result/000000" -mop "checkpoints/optical.pth" -mor "checkpoints/original.pth"
```

## License 
The code and dataset is released only for academic research. Commercial usage is strictly prohibited.

## Citation
 ```
@article{AIGVDet24,
author = {Jianfa Bai and Man Lin and Gang Cao and Zijie Lou},
title = {{AI-generated video detection via spatial-temporal anomaly learning}},
conference = {The 7th Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
year = {2024},}
```

## Contact
If you have any questions, please contact us(lyan924@cuc.edu.cn).


