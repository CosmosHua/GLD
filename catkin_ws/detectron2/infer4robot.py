#!/usr/bin/python3
# coding: utf-8

import os, cv2, random
from detectron2.config import get_cfg
from detectron2.structures import Instances, Boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from glob import glob; from time import time, sleep
from torch import empty, cuda; import numpy as np


##########################################################################################
def Setup4Infer(data_dir, thd=0.6):
    data_name = 'doors'; json = data_dir+'/annotations.json'
    register_coco_instances(data_name, {}, json, data_dir)
    dicts = DatasetCatalog.get(data_name) # dicts[2]
    metadata = MetadataCatalog.get(data_name) # metadata
    metadata.thing_colors = [tuple(np.random.randint(0,256,3)) #list
        for c in metadata.thing_classes] #random.choices(range(256),k=3) 

    model = sorted(glob(data_dir+'/models/*.pth'))[-1]
    ym = glob(os.path.dirname(model)+'/*x.yaml')[0]
    assert os.path.isfile(model) and os.path.isfile(ym)
    print('Loading model: %s' % model)

    cfg = get_cfg(); cfg.merge_from_file(ym)
    cfg.DATASETS.TEST = (data_name,)
    cfg.MODEL.WEIGHTS = model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thd # test threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
    return DefaultPredictor(cfg), metadata


RES = lambda x,s: cv2.resize(x, NewSZ(x,s))
INT = lambda x,f=1: tuple(int(i*f) for i in x) # round
RSZ = lambda x,s: cv2.resize(x,s) if type(s)==tuple else cv2.resize(x,None,fx=s,fy=s)
##########################################################################################
def NewSZ(wh, sz=1.0):
    if type(wh)==cv2.UMat: wh = wh.get()
    if type(wh)==np.ndarray: wh = wh.shape[1::-1]
    else: assert type(wh) in (tuple, list)
    if type(sz)==float: sz = INT(wh, sz)
    elif type(sz)==int: sz = INT(wh, sz/wh[1])
    return sz if type(sz)==tuple else tuple(wh)


def Infer2Img(im, pred, meta, sz=1.0, info=''): # BGR
    t = time(); cuda.current_stream().synchronize()
    out = old = pred(im)['instances'].to('cpu')
    #print(id_cls(meta.thing_classes)); print(meta); print(out)
    out = Sift_CLS(out, meta, [1,2,'door_unknown'], 'include')
    #out = Sift_BOX(out, meta, wt=weight); sc = max(1,720/im.shape[0])
    if len(old)!=len(out): print(old.pred_classes,'->',out.pred_classes)
    vis = Visualizer(im[:,:,::-1], metadata=meta, scale=1.0,
            instance_mode=ColorMode.SEGMENTATION); sz = NewSZ(im,sz)
    im = vis.draw_instance_predictions(out).get_image()[:,:,::-1]
    #sleep(0.9+0.1*np.random.rand())
    h,w = im.shape[:2]; im = cv2.UMat(im) # for OpenCV 4.2+
    cuda.current_stream().synchronize(); t = (time()-t)*1000
    cv2.putText(im, '%.1fms'%t, (w-75,h-8), 4, 0.5, (0,255,255), 1)
    cv2.putText(im, info, (5,h-8), 4, 0.5, (0,255,255), 1)
    return cv2.resize(im.get(),sz), out


weight = dict(door_open=1, door_close=1)
id_cls = lambda CLS: {i:c for i,c in enumerate(CLS)}
cls_id = lambda CLS: {c:i for i,c in enumerate(CLS)}
##########################################################################################
def Sift_CLS(out, meta, cls=[], mod='ex'):
    if type(cls) not in (list,tuple): cls = [cls]
    cid = meta.thing_classes; id = cid.index; idx = []
    for c in cls: # filter and map->id
        if type(c)==str and c in cid: idx += [id(c)]
        if type(c)==int and 0<=c<len(cid): idx += [c]
    if len(out)<0 or len(idx)<len(cls): return out

    pred = out.pred_classes; h,w = out.image_size
    if 'in' in mod: idx = [i for i,d in enumerate(pred) if d in idx]
    else: idx = [i for i,d in enumerate(pred) if d not in idx]
    if len(idx)>0: return out[idx] # out.__class__.cat()
    new = dict(pred_boxes=Boxes(empty(0,4)), scores=empty(0),
        pred_classes=empty(0,dtype=int), pred_masks=empty(0,h,w))
    return Instances((h,w), **new) # OR: out.cat()


def Sift_BOX(out, meta, thd=10, wt=None):
    if len(out)<0: return out # unnecessary
    cid = meta.thing_classes; id = cid.index
    pred = out.pred_classes; prob = out.scores
    ct = out.pred_boxes.get_centers(); idx = []
    if type(wt)==dict:
        wt = {id(k):wt[k] for k in wt if k in cid} # map->id
        prob = prob + sum((pred==k)*wt[k] for k in wt) # adjust
        '''wt = [wt[i] if cid[i] in wt else 0 for i in pred]
        prob = prob + np.array(wt) # adjust prob'''
    for c in ct: #(N,2)
        # TODO: using IOU and class-aware/specific
        dis = np.linalg.norm(ct-c, axis=1) # distance
        p = np.where(dis<thd)[0] # np.nonzero(dis<thd)[0]
        k = p[prob[p].sort(0,True).indices[0]] # sort
        if k not in idx: idx.append(k)
    return out[idx]


##########################################################################################
if __name__ == '__main__':
    root = os.path.expanduser('~/GLD_Git/Data_Door/coco_door3')
    pred, meta = Setup4Infer(root, 0.6)

    im = cv2.imread('xxx.jpg')
    im, out = Infer2Img(im, pred, meta, 720)
    cv2.imshow('im',im); print(meta, out)
    cv2.waitKey(0); cv2.destroyAllWindows()
    #cv2.imwrite('xxx.png', im)

