import argparse
import contextlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
import torch

from models.experimental import attempt_load
from models.yolo import ClassificationModel, Detect, DetectionModel, SegmentationModel
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, check_requirements, check_version,
                           check_yaml, colorstr, file_size, get_default_args, print_args, url2file, yaml_save)
from utils.torch_utils import select_device, smart_inference_mode


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to 

def option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", 
                        type=str,
                        help="model file (.pt, .onnx, .engine")
    parser.add_argument('--imgsz', '--img', '--img-size', 
                        nargs='+', 
                        type=int, 
                        default=[640, 640],
                        help='image (h, w)')
    parser.add_argument('--batch-size', 
                        type=int, 
                        default=1, 
                        help='batch size')
    parser.add_argument('--device', 
                        default='0', 
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', 
                        action='store_true', 
                        help='FP16 half-precision export')
    parser.add_argument('--dynamic', 
                        action='store_true', 
                        help='ONNX/TF/TensorRT: dynamic axes')
    parser.add_argument('--simplify', 
                        action='store_true', 
                        help='ONNX: simplify model')
    parser.add_argument('--opset', 
                        type=int, 
                        default=17, 
                        help='ONNX: opset version')
    parser.add_argument('--verbose', 
                        action='store_true', 
                        help='TensorRT: verbose log')
    parser.add_argument('--workspace', 
                        type=int, 
                        default=4, 
                        help='TensorRT: workspace size (GB)')
    parser.add_argument('--include',
                        nargs='+',
                        default=['onnx'],
                        help="onnx, engine")
    
    parser.add_argument('--inplace', 
                        action='store_true', 
                        help='set YOLOv5 Detect() inplace=True')
    
    opt = parser.parse_args()

    return opt

def run(
        weights=ROOT / 'yolov5s.pt',
        imgsz=(640, 640),
        batch_size=1,
        device='0',
        half=False,
        dynamic=False,
        simplify=False,
        opset=12,
        verbose=False,
        workspace=4,
        include = ('onnx, engine')
):
    t = time.time()
    include = [x.lower for x in include]
    
    file = weights
    device = select_device(device)
    if half:
        assert device.type != 'cpu', '--half only compatible with GPU export, i.e. use --device 0'
        assert not dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both'
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model

    #check image size
    imgsz *= 2 if len(imgsz) == 1 else 1

    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True

    for _ in range(2):
        y = model(im)  # dry runs


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = option()
    main(opt)

