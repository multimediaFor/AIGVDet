import argparse
import os
import sys
from abc import ABC
from typing import Type


class DefaultConfigs(ABC):
    ####### base setting ######
    gpus = [0]
    seed = 3407
    arch = "resnet50"
    datasets = ["trainset_1"]  # Changed from trainset_1 to match test/train structure
    datasets_test = ["val_set_1"]  # Changed from val_set_1 to match test/val structure
    mode = "binary"
    class_bal = False
    batch_size = 64 # RTX 3090 24GB can handle original batch size
    loadSize = 512 # Resizes the image, this is used as it is standard for HIGH RES CNNs and must have margin before cropping
    cropSize = 448
    epoch = "latest"
    num_workers = 16  # Increased for faster data loading on 24GB GPU (choose 8 or 16) - doesnt affect accuracy but only throughput
    serial_batches = False
    isTrain = True

    # data augmentation - to match the paper's augmentation rate (10%), set blur and jpg to 0.5
    rz_interp = ["bilinear"]
    # blur_prob = 0.1 - resnet50 template
    blur_prob = 0.1 # optical flow
    blur_sig = [0.5]
    # jpg_prob = 0.1 - resnet50 template
    jpg_prob = 0.1 # optical flow
    # P(augmented) = 1-(1-0.5)(1-0.05) = 0.975
    jpg_method = ["cv2"]
    jpg_qual = list(range(70, 91))
    gray_prob = 0.0
    aug_resize = True
    aug_crop = True 
    aug_flip = True # optical flow
    aug_norm = True # optical flow

    ####### train setting ######
    warmup = False
    # warmup = True
    warmup_epoch = 3
    # earlystop = True - resnet50 template, training ends only when lr reaches 1e-6 which trainer already supports (Trainer.adjust_learning_rate(min_lr=1e-6))
    earlystop = True
    earlystop_epoch = 5
    optim = "adam"
    new_optim = False
    loss_freq = 400
    save_latest_freq = 2000
    save_epoch_freq = 5
    continue_train = False
    epoch_count = 1
    last_epoch = -1
    # nepoch = 100, try
    nepoch = 50
    beta1 = 0.9
    lr = 0.0001
    init_type = "normal"
    init_gain = 0.02
    pretrained = True

    # paths information
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_root = os.path.join(root_dir, "data")
    exp_root = os.path.join(root_dir, "data", "exp")
    _exp_name = ""
    exp_dir = ""
    ckpt_dir = ""
    logs_path = ""
    ckpt_path = ""

    @property
    def exp_name(self):
        return self._exp_name

    @exp_name.setter
    def exp_name(self, value: str):
        self._exp_name = value
        self.exp_dir: str = os.path.join(self.exp_root, self.exp_name)
        self.ckpt_dir: str = os.path.join(self.exp_dir, "ckpt")
        self.logs_path: str = os.path.join(self.exp_dir, "logs.txt")

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def to_dict(self):
        dic = {}
        for fieldkey in dir(self):
            fieldvalue = getattr(self, fieldkey)
            if not fieldkey.startswith("__") and not callable(fieldvalue) and not fieldkey.startswith("_"):
                dic[fieldkey] = fieldvalue
        return dic


def args_list2dict(arg_list: list):
    assert len(arg_list) % 2 == 0, f"Override list has odd length: {arg_list}; it must be a list of pairs"
    return dict(zip(arg_list[::2], arg_list[1::2]))


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    elif v.lower() in ("true", "yes", "on", "y", "t", "1"):
        return True
    elif v.lower() in ("false", "no", "off", "n", "f", "0"):
        return False
    else:
        return bool(v)


def str2list(v: str, element_type=None) -> list:
    if not isinstance(v, (list, tuple, set)):
        v = v.lstrip("[").rstrip("]")
        v = v.split(",")
        v = list(map(str.strip, v))
        if element_type is not None:
            v = list(map(element_type, v))
    return v


CONFIGCLASS = Type[DefaultConfigs]

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default=[0], type=int, nargs="+")
parser.add_argument("--exp_name", default="", type=str)
parser.add_argument("--ckpt", default="model_epoch_latest.pth", type=str)
parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
args = parser.parse_args()

if os.path.exists(os.path.join(DefaultConfigs.exp_root, args.exp_name, "config.py")):
    sys.path.insert(0, os.path.join(DefaultConfigs.exp_root, args.exp_name))
    from config import cfg

    cfg: CONFIGCLASS
else:
    cfg = DefaultConfigs()

if args.opts:
    opts = args_list2dict(args.opts)
    for k, v in opts.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unrecognized option: {k}")
        original_type = type(getattr(cfg, k))
        if original_type == bool:
            setattr(cfg, k, str2bool(v))
        elif original_type in (list, tuple, set):
            setattr(cfg, k, str2list(v, type(getattr(cfg, k)[0])))
        else:
            setattr(cfg, k, original_type(v))

cfg.gpus: list = args.gpus
os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(gpu) for gpu in cfg.gpus])
cfg.exp_name = args.exp_name
cfg.ckpt_path: str = os.path.join(cfg.ckpt_dir, args.ckpt)

if isinstance(cfg.datasets, str):
    cfg.datasets = cfg.datasets.split(",")
