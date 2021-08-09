#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, cv2, torch
import infer4robot as Seg
from detectron2.structures import boxes


CID = lambda C,L: [C.index(i) for i in C if i in L]
'''{0: 'door_way', 1: 'door_open', 2: 'door_close', 3: 'door_unknown',
4: 'door_hydrant', 5: 'wall_glass', 6: 'door_leaf_wood', 7: 'door_leaf_glass',
8: 'door_leaf_metal', 9: 'door_lateral_plank', 10: 'door_window', 11: 'doorplate',
12: 'room_number', 13: 'sign_exit', 14: 'light_on', 15: 'light_off', 16: 'person'}'''
##########################################################################################
# Ref: detectron2.structures.boxes.pairwise_iou
def PairwiseIOU(A, B): # A:(N,4), B:(M,4)
    assert type(A)==type(B)==boxes.Boxes
    uni_area = A.area()[:,None] + B.area()

    A, B = A.tensor[:,None], B.tensor # wh:(N,M,2)
    wh = torch.min(A[...,2:],B[:,2:]) - torch.max(A[...,:2],B[:,:2])
    wh.clamp_(min=0); inter = wh.prod(dim=2) #(N,M)

    iou = torch.where(inter>0, # handle empty box
        inter/(uni_area-inter), torch.zeros(1) )
    return iou #(N,M)


# Ref: detectron2.structures.boxes.inside_box
def InsideBox(A, B, thd=0): # A:(N,4), B:(M,4)
    if type(A)==boxes.Boxes: A = A.tensor
    if type(B)==boxes.Boxes: B = B.tensor
    A = A[:,None] #(N,1,4): broadcast on inserted dim
    idx = [(A[...,0]>=B[...,0]-thd) & (A[...,1]>=B[...,1]-thd) &
        (A[...,2]<=B[...,2]+thd) & (A[...,3]<=B[...,3]+thd)] #(N,M)
    return idx #(N,M)


# H:[0,180], S:[0,255], V:[0,255]
def lumin(im, rt=1/3, roi=None):
    assert type(im)==np.ndarray # BGR
    bin = round(256*rt); rag = 256-bin
    if type(roi)==boxes.Boxes: roi = roi.tensor
    if type(roi) in (np.ndarray,torch.Tensor):
        x1,y1,x2,y2 = roi; im = im[y1:y2+1,x1:x2+1]
    v = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[2]
    return v[np.where(v>=rag)].mean()
    # roi = np.zeros(im.shape,im.dtype); roi[y1:y2+1,x1:x2+1] = 255
    # hist = cv2.calcHist([v],[0],roi,[bin],[rag,256])


def Lumin(im, out, meta, thd=188):
    CLS = meta.thing_classes; obj = ['light_on','person']
    glass = ['wall_glass','door_leaf_glass','door_window']
    door = ['door_way','door_open','door_close','door_unknown']
    door,glass,obj = [CID(CLS,i) for i in (door,glass,obj)]
    # door, glass, obj = [0,1,2,3], [5,7,10], [14,16]
    cls = out.pred_classes; obj = CID(cls,obj); idx = []
    box = out.pred_boxes; ISB = InsideBox(box,box)
    for i,c in enumerate(cls):
        if c not in door+glass: continue
        if any(ISB[i][obj]): # person/light_on?
            idx.append(i); continue
        if c not in glass: continue # glass?
        lumin(im,box[i])
        # something to do


##########################################################################################
if __name__ == '__main__':
    root = os.path.expanduser('~/GLD_Git/Data_Door/coco_door')
    seg = ros_seg(root, 'BIM.json', 1.0)

    try: rospy.spin()
    finally: log_output(seg.log)

