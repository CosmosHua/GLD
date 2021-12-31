#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, sys, cv2
from glob import glob


INT = lambda x,f=1: tuple(int(i*f) for i in x)
RSZ = lambda x,s: cv2.resize(x, None, fx=s, fy=s)
##########################################################################################
def canny_handle(img, th=(50,150), k=3):
    if type(img)==str: img = cv2.imread(img)
    r = 720/img.shape[0]; img = RSZ(img, r)
    im = cv2.GaussianBlur(img, (k,k), 0)
    im = cv2.Canny(im, *th)

    lines = cv2.HoughLines(im, 1, np.pi/180, int(250*r), min_theta=-0.5, max_theta=0.5)
    #lines = cv2.HoughLinesP(im, 1, np.pi/180, int(150*r), minLineLength=100, maxLineGap=20)
    cnt, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = [c.squeeze() for c in cnt if c.shape[0]>2 and cv2.contourArea(c)>300]
    ct = np.array([list(cv2.moments(c).values())[:3] for c in cnt]).transpose()
    if len(ct): ct = np.int32(ct[1:]/ct[0]).transpose().tolist(); #print(ct)

    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for i in lines.squeeze():
        r,t = i # for cv2.HoughLines
        p = np.array([np.cos(t), np.sin(t)])*r
        t = np.array([-np.sin(t), np.cos(t)])*1000
        p1 = tuple(np.int32(p-t))
        p2 = tuple(np.int32(p+t))
        cv2.line(im, p1, p2, (0,255,0), 2)#'''
        #x1, y1, x2, y2 = i # for cv2.HoughLinesP
        #cv2.line(im, (x1,y1), (x2,y2), (0,255,0), 2)
    #im = cv2.drawContours(im, cnt, -1, (0,0,255), 2)
    im = cv2.fillPoly(im, cnt, (0,0,255))
    #im = cv2.addWeighted(im, 0.7, img, 0.3, 0) # black
    im = np.where(im>0, im, img) # color
    cv2.imshow('seg',im); return im


########################################################
def test_canny(src='test', th=(50,150)):
    for i in glob(f'{src}/*.jpg'):
        im = canny_handle(i,th); cv2.waitKey(5)
        cv2.imwrite(i[:-4]+f'_c{th}.png', im)
    cv2.destroyAllWindows()


import json
##########################################################################################
def mask_show(js, alpha=0.5): # for labelme
    with open(js,'r') as f: sh = json.load(f)
    im = os.path.dirname(js)+'/'+sh['imagePath']
    im = cv2.imread(im); fc = (255,255,255)
    for s in sh['shapes']:
        s.pop('group_id'); s.pop('flags')
        p = s['points'] = np.int32(s['points'])
        color = np.random.randint(0,256,3).tolist()
        pt0, pt1 = tuple(p[0]), tuple(p[1])
        if s['shape_type']=='circle':
            r = np.linalg.norm(p[0]-p[1]).astype(int)
            m = cv2.circle(im.copy(), pt0, r, color, -1)
        elif s['shape_type']=='rectangle':
            m = cv2.rectangle(im.copy(), pt0, pt1, color, -1)
        elif s['shape_type']=='polygon':
            m = cv2.drawContours(im.copy(), [p], 0, color, -1)
        im = cv2.addWeighted(im, 1-alpha, m, alpha, 0)
        cv2.putText(im, s['label'], pt0, 7, 1, fc)
    cv2.imshow('seg',im); return im, sh['shapes']


def mask_seg(js, mod=0): # for labelme
    with open(js,'r') as f: sh = json.load(f)
    im = os.path.dirname(js)+'/'+sh['imagePath']
    im = cv2.imread(im); mask = []
    for s in sh['shapes']:
        m = np.zeros(im.shape[:2], im.dtype)
        s.pop('group_id'); s.pop('flags')
        p = s['points'] = np.int32(s['points'])
        pt0, pt1 = tuple(p[0]), tuple(p[1])
        if s['shape_type']=='circle':
            r = np.linalg.norm(p[0]-p[1]).astype(int)
            cv2.circle(m, pt0, r, (255,)*3, -1)
        elif s['shape_type']=='rectangle':
            cv2.rectangle(m, pt0, pt1, 255, -1)
        elif s['shape_type']=='polygon':
            cv2.drawContours(m, [p], 0, 255, -1)
        mask.append(m.astype(bool))
    merge = np.any(mask, axis=0) # merge instances
    split = np.stack(mask, axis=0) # instance: (n,h,w)
    cv2.imshow('seg', im*merge[...,None])
    return im, (split if mod else merge)


########################################################
def test_mask(src='test'):
    for i in glob(f'{src}/*.json'):
        im, mk = mask_seg(i); cv2.waitKey(5)
        cv2.imwrite(i[:-4]+f'_m.png', im*mk[...,None])
        im, sh = mask_show(i); cv2.waitKey(0)
        cv2.imwrite(i[:-4]+f'_l.png', im)
    cv2.destroyAllWindows()


##########################################################################################
def get_masks_color(src, v=30, c=2):
    HSV = []; SET = {}
    for i in glob(f'{src}/*.json'):
        im, mk = mask_seg(i)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        HSV.append(im[mk.nonzero()])
    HSV = np.concatenate(HSV) # (N,3)
    if v: HSV = HSV[np.where(HSV[:,2]>v)]

    '''h = int(np.ceil(len(HSV)**0.5))
    im = np.resize(HSV, (h,h,3)) # auto fill
    im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
    cv2.imshow('set',im); cv2.waitKey(500)#'''

    #his = [np.bincount(HSV[:,i]) for i in range(3)]
    '''import matplotlib.pyplot as plt
    plt.hist(HSV, 256, label=list('hsv'), stacked=True)
    plt.legend(); plt.show()#'''

    #SET = np.array(list(set([tuple(i[:c]) for i in HSV])))
    for i in HSV: i = tuple(i[:c]); SET[i] = SET.get(i,0)+1
    SET = dict(sorted(SET.items(), key=lambda x: -x[1]))
    #HSV = sorted(HSV, key=lambda x: -SET[tuple(x)])
    return np.array(list(SET.keys()))


##########################################################################################
def get_colors_mask(im, cs=10, c=1):
    if type(cs) in (int,float): # silver
        #b,g,r = im.transpose(2,0,1) # futile
        #mask = np.abs([b-g,g-r,r-b]).max(axis=0)<cs
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h,s,v = im.transpose(2,0,1) # (c,h,w)
        mask = np.all([s<2*cs,v>cs], axis=0)
        return np.stack([mask]*c, axis=-1)
    else: # cs is colors_set
        if type(cs)!=np.ndarray: # ineffective
            cs = get_masks_color(src)[:50]
        k = cs.shape[-1]; cs = cs[:,None,None]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[...,:k]
        mask = (im==cs).all(axis=-1).any(axis=0)
        return np.stack([mask]*c, axis=-1)


########################################################
def test_colors(src='test', d=50):
    cs = 10 #get_masks_color(src)[:d]
    d = cs if type(cs)==int else d
    for i in glob(f'{src}/*.jpg'):
        im = cv2.imread(i)
        im = RSZ(im, 720/im.shape[0])
        mk = get_colors_mask(im, cs)
        im = cv2.addWeighted(im*mk, 0.6, im, 0.4, 0)
        cv2.imshow('seg',im); cv2.waitKey(5)
        cv2.imwrite(i[:-4]+f'_s{d}.png', im)
    cv2.destroyAllWindows()


from yolov5.infer import yolov5
from cv_bridge import CvBridge; CVB = CvBridge()
from sensor_msgs.msg import CompressedImage, Image
def get_rgb(x): global rgb; rgb = x # callback
def get_dpt(x): global dpt; dpt = x # callback
##########################################################################################
def det_handle(model, cls=None):
    det = yolov5(model, cls=cls)
    name = {i:k for i,k in enumerate(det.names)}
    while cv2.waitKey(5)!=27:
        if type(rgb)==cv2.VideoCapture: im = rgb.read()[1]
        elif hasattr(rgb, 'header'):
            im = CVB.compressed_imgmsg_to_cv2(rgb)
        elif type(rgb)==np.ndarray: im = rgb.copy()
        elif type(rgb)==str: im = cv2.imread(rgb)
        else: print(type(rgb)); continue
        im, pred, dt, _ = det.infer1(im)
        pred = pred[0].cpu().numpy() # [n,6]
        for x1,y1,x2,y2,p,c in pred:
            x1,y1,x2,y2,c = INT([x1,y1,x2,y2,c])
            if c==1: # seg handle
                hd = im[y1:y2, x1:x2]
                mk = get_colors_mask(hd,15); hd *= mk
                mk = (mk*255).astype('uint8')[...,0]
                #mk = cv2.Canny(cv2.GaussianBlur(mk,(3,3),0), 50, 150)
                cnt, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt = [c.squeeze() for c in cnt if c.shape[0]>2 and cv2.contourArea(c)>50]
                mk = cv2.fillPoly(np.stack([mk]*3,axis=-1), cnt, (0,0,255))
                cv2.imshow(name[c], mk)#'''
        pub_det(pred, name); cv2.imshow('det', im)
    cv2.destroyAllWindows()


from sensor_msgs.msg import RegionOfInterest
from geometry_msgs.msg import Point32, Polygon
from glodon_msgs.msg import Object; import rospy
########################################################
def pub_det(Pd, name): # Pd=[n,6]
    c = np.where(Pd[:,-1]==1)[0]
    if len(c)<1: return # swap row
    c = c[0]; Pd[(0,c),:] = Pd[(c,0),:]

    x1,y1,x2,y2, prob, cls = Pd.transpose()
    res = dict(cid=np.int32(cls), prob=prob)
    res['box'] = np.int32([x1,y1,x2-x1,y2-y1]).transpose()
    res['cls'] = [name[int(i)] for i in cls]
    res['cnt'] = np.random.rand(len(cls), 0, 2)

    obj = Object(header=rgb.header)
    if hasattr(obj,'scores'): obj.scores = res['prob']
    if hasattr(obj,'class_ids'): obj.class_ids = res['cid']
    if hasattr(obj,'class_names'): obj.class_names = res['cls']
    if hasattr(obj,'boxes'): obj.boxes = [RegionOfInterest(x_offset=x,
        y_offset=y, width=w, height=h) for (x,y,w,h) in res['box']]
    if hasattr(obj,'contours'): obj.contours = [Polygon([Point32(
        x=x, y=y) for (x,y) in c]) for c in res['cnt']]
    pub.publish(obj)


from detectron2 import geometry as Geo
##########################################################################################
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    #test_canny(src='test', th=(50,150))
    #test_colors(src='test', d=50)
    #test_mask(src='test')

    rospy.init_node('handle', anonymous=True); #rgb = cv2.VideoCapture(-1)
    rospy.Subscriber('aligned_depth_to_color/image_raw', Image, get_rgb, queue_size=1)
    rospy.Subscriber('/ee/color/image_raw/compressed', CompressedImage, get_rgb, queue_size=1)
    pub = rospy.Publisher('result/handle', Object, queue_size=1)
    det_handle('yolov5/door_mid_b32_e500/weights/last.pt', [0,1])

