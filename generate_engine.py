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



def try_export(inner_func):
    # YOLOv5 export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args['prefix']
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f'{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)')
            return f, model
        except Exception as e:
            LOGGER.info(f'{prefix} export failure ❌ {dt.t:.1f}s: {e}')
            return None, None

    return outer_func



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
        inplace=False,
        include = ('onnx', 'engine')
):
    t = time.time()
    include = [x.lower() for x in include]
    
    file = Path(weights)
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

    #warm up
    for _ in range(10):
        y = model(im)   

    if half:
        im, model = im.half(), model.half() #to FP 16
    
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    metadata = {'stride': int(max(model.stride)), 'names': model.names}  # model metadata
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

    output = {}

    print("include: {}".format(include))
    # if 'engine' in include:
        # output['engine'], _ = export_engine(model, im, file, half, dynamic, simplify, workspace, verbose)
    if 'onnx' in include:
        output['onnx'], _ = export_onnx(model, im, file, opset, dynamic, simplify)

    duration = time.time() - t
    LOGGER.info(f'\nExport complete ({duration:.1f}s)')



@try_export
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr('ONNX:')):

    check_requirements('onnx>=1.12.0')
    import onnx
    
    LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
    onnx_file_name = str(file.with_suffix('.onnx'))

    output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output0']
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
        if isinstance(model, SegmentationModel):
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
            dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
        elif isinstance(model, DetectionModel):
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)

    torch.onnx.export(
        model.cpu() if dynamic else model,
        im.cpu() if dynamic else im,
        onnx_file_name,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic or None
    )
    
    
    model_onnx = onnx.load(onnx_file_name)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, onnx_file_name)

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
            import onnxsim

            LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, onnx_file_name)
        except Exception as e:
            LOGGER.info(f'{prefix} simplifier failure: {e}')
    return onnx_file_name, model_onnx
def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = option()
    main(opt)

