#!/usr/bin/python3
# coding: utf-8


import matplotlib, gi
matplotlib.use('Agg') #'TkAgg'
#gi.require_version('Gtk','2.0')

import numpy as np
import os, sys, cv2
from pathlib import Path
from random import randint
from time import time

import torch, json, yaml
from torch.backends import cudnn

FILE = Path(__file__).absolute() # add path
sys.path.append(FILE.parents[0].as_posix())

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.plots import Annotator, colors
#from utils.plots import plot_one_box


##########################################################################################
@torch.no_grad()
class yolov5(object):
    def __init__(self, wt, conf=0.25, iou=0.45, cls=None, size=640, augment=False, agnostic_nms=False):
        if type(wt)!=str or not os.path.isfile(wt):
            for wt in ['yolov5x.pt', 'yolov5l.pt', 'yolov5m.pt', 'yolov5s.pt']:
                if os.path.isfile(wt): break
        self.weight = wt; assert os.path.isfile(wt)
        print(f'YOLOv5: {torch.cuda.get_device_name()}')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.half = self.device.type!='cpu' # only CUDA supports half-precision
        self.model = model = attempt_load(wt, map_location=self.device) # FP32
        if self.half: self.model.half() # convert to FP16

        self.names = model.module.names if hasattr(model,'module') else model.names
        self.colors = [[randint(0,255) for _ in range(3)] for _ in self.names]

        self.iou_thres = iou; self.conf_thres = conf
        self.classes = cls; self.agnostic_nms = agnostic_nms
        self.augment = augment; s = int(model.stride.max())
        self.imsz = int(size if size%s==0 else (size//s)*s)

        cudnn.benchmark = True # speedup for constant img_size
        #img = torch.zeros((1, 3, self.imsz, self.imsz), device=self.device)
        #model(img.half() if self.half else img) # run once for test


    def plot(self, im, det):
        '''names = self.names; colors = self.colors
        for *xyxy, conf, c in reversed(det): # (x1,y1,x2,y2,conf,cls)
            c = int(c); label = f'{names[c]} {conf:.2f}' # Add bbox to image
            plot_one_box(xyxy, im, label=label, color=colors[c], line_width=2)#'''

        names = self.names; from utils.plots import colors
        annotator = Annotator(im, line_width=2, example=str(names))
        for *xyxy, conf, c in reversed(det): # (x1,y1,x2,y2,conf,cls)
            c = int(c); label = f'{names[c]} {conf:.2f}' # Add bbox to image
            annotator.box_label(xyxy, label, color=colors(c, True))
        im[:] = annotator.result()#'''

        return {names[int(c)]:int((det[:,-1]==c).sum()) for c in det[:,-1].unique()}


    def infer1(self, im, prepare=True, post=True):
        if type(im)==str: im = cv2.imread(im)
        assert type(im)==np.ndarray
        if prepare:
            img = letterbox(im, new_shape=self.imsz)[0] # resize
            img = img[...,::-1].transpose(2,0,1) # BGR->RGB->(C,H,W)
            img = np.ascontiguousarray(img)
        else: img = im

        img = torch.from_numpy(img).to(self.device)
        # convert uint8 to fp16/fp32, [0,255] to [0,1.0]
        img = (img.half() if self.half else img.float())/255.0
        if img.ndimension()==3: img = img.unsqueeze(0)

        t0 = time_sync() # Inference
        pred = self.model(img, augment=self.augment)[0]
        # Apply NMS. pred=[N,(n,6)]: list of batch_size=N tensors (n,6)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                        classes=self.classes, agnostic=self.agnostic_nms)
        dt = (time_sync()-t0)*1000
        if post:
            for det in pred: # Rescale boxes from img_shape to im_shape, det=(n,6)
                det[:,:4] = scale_coords(img.shape[2:], det[:,:4], im.shape).round()
            return im, pred, dt, self.plot(im, pred[0]) # det=pred[-1]
        return im, pred, dt # det[i]=(x1,y1,x2,y2,conf,cls)


    def infer(self, src, dst=None, show='det'):
        src = str(src); save = False
        if type(dst)==str:
            os.makedirs(dst, exist_ok=True)
            dst = Path(dst); save = True
        webcam = src.isdigit() or src.startswith(('rtsp://','http://'))

        if webcam: # Set Dataloader
            data = LoadStreams(src, img_size=self.imsz)
        else: data = LoadImages(src, img_size=self.imsz)

        vid_path, vid_writer = None, None
        for path, img, imgs, vid_cap in data:
            im, pred, dt = self.infer1(img, prepare=False, post=False)

            # Process detections per batch of images, img=Tensor(N,c,h,w)
            for i, det in enumerate(pred): # webcam: N>=1 for multi-stream
                p, im = (Path(path[i]+'.mp4'), imgs[i].copy()) if webcam else (Path(path), imgs)
                det[:,:4] = scale_coords(img.shape[2:], det[:,:4], im.shape).round()
                print('%.2fms:'%dt, self.plot(im, det))

                if show: # show results
                    cv2.imshow(str(show), im)
                    if cv2.waitKey(1)==27: return

                if save: # save results
                    save_path = str(dst/p.name)
                    if data.mode=='images' or not webcam:
                        if os.path.isfile(save_path):
                            save_path = save_path[:-4]+'_det.jpg'
                        cv2.imwrite(save_path, im)
                    else: # video or stream
                        if vid_path != save_path:
                            vid_path = save_path # new video
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release() # release previous video writer
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # video codec
                            if vid_cap: # for video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else: h, w = im.shape[:2]; fps = 30 # for stream
                            vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (w,h))
                        vid_writer.write(im)


##########################################################################################
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    '''with open('yolov5/data/cooc.yaml') as f:
        cls = yaml.safe_load(f)['names'] # det.names'''
    det = yolov5('yolov5l.pt', cls=[0,63]) # 0=person
    #det.infer(src=0) # batch_infer
    cap = cv2.VideoCapture(-1)
    while cv2.waitKey(5)!=27: # single_infer
        im, res, dt = det.infer1(cap.read()[1])
        cv2.imshow('yolo', im)#'''

