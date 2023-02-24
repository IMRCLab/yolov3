 # YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov3.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import shutil
import numpy as np
import yaml
from functions import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
# python3 inference.py PATH-TO-MAIN-FOLDER --weights PATH-TO-WEIGHTS
@torch.no_grad()
def run(foldername,
        weights,  
        imgsz,  # inference size (pixels)
        conf_thres=0.45,  # confidence threshold           0.25
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = "{}yolov3/images/test/**/*.jpg".format(foldername)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    
     # save predicted images with bb
    detection_dir = "{}yolov3/det/".format(foldername)

    if os.path.exists(detection_dir): shutil.rmtree(detection_dir)
    os.mkdir(detection_dir)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    bs = 1  # batch_size
    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    predictions, images = {},{}
    for path, im, im0s, vid_cap, s in dataset:
        print(path)
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # read mapping
        with open(Path(foldername) / "yolov3" / "filename_to_dataset_mapping.yaml", 'r') as stream:
            filename_to_dataset_key = yaml.safe_load(stream)
        
        # read original data
        with open(Path(foldername) / "dataset_filtered.yaml", 'r') as stream:
            filtered_dataset = yaml.safe_load(stream)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            
            pred_neighbors, per_image = [], {}
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(detection_dir + p.name)  
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # get the calibration file, camera matrix for each image
                calibration_key = filtered_dataset['images'][filename_to_dataset_key[p.name]]['calibration']
                camera_matrix = np.array(filtered_dataset['calibration'][calibration_key]['camera_matrix'])
                dist_vec = np.array(filtered_dataset['calibration'][calibration_key]['dist_coeff'])
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                    # line = (det[0][0], det[0][1], det[0][2], det[0][3]) 
                    # inference_file.write('{},{},{},{},{} \n'.format(p.name,det[0][0],det[0][1],det[0][2],det[0][3]))
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

                det = det.cpu()
                for j in range(det.shape[0]): # for each robot
                    xyz = xyz_from_bb(det[j],camera_matrix,dist_vec) # det -> xmin,ymin,xmax,ymax
                    pred_neighbors.append(np.array([xyz[0],xyz[1],xyz[2]]))

                    # xyz_yolo.append(np.array((det[j][0], det[j][1], det[j][2], det[j][3])).tolist())
                all_robots = []
                for h in range(len(pred_neighbors)):
                    per_robot = {}
                    per_robot['pos'] = pred_neighbors[h].tolist() 
                    all_robots.append(per_robot)
                per_image['visible_neighbors'] = all_robots
                # images[p.name] = per_image
                images[filename_to_dataset_key[p.name]] = per_image

            else:
                per_image['visible_neighbors'] = []
                # images[p.name] = per_image
                images[filename_to_dataset_key[p.name]] = per_image
            # Stream results
            im0 = annotator.result()
            cv2.imwrite(save_path, im0)
    # Print results
    # inference_file.close()
    predictions['images'] = images
    with open("{}yolov3/inference_yolo.yaml".format(foldername), 'w') as outfile:
        yaml.dump(predictions, outfile)
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    LOGGER.info(f"Results saved to {colorstr('bold', detection_dir)}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('foldername', help="Path to the Main folder")
    parser.add_argument('--weights', nargs='+', type=str, default='/home/akmaral/tubCloud/Shared/cvmrs/trained-models/yolov3/synth-1k-best.pt', help='weights path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[320], help='image size h,w')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    # run(**vars(opt))
    run(opt.foldername, opt.weights, opt.imgsz)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
