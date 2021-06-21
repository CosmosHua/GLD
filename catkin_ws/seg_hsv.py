#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, cv2, sys


HSV_Set = {'red': [(-24, 55, 55), (11, 255, 255)],
        'yellow': [(26, 55, 55), (34, 255, 255)],
        'blue': [(100, 55, 55), (124, 255, 255)],
        'green': [(35, 55, 55), (77, 255, 255)]}
COLORS = [(244,  67,  54), (233,  30,  99), ( 96, 125, 139),
          (103,  58, 183), ( 63,  81, 181), ( 33, 150, 243),
          (  3, 169, 244), (  0, 188, 212), (  0, 150, 136),
          ( 76, 175,  80), (139, 195,  74), (205, 220,  57),
          (255, 235,  59), (255, 193,   7), (255, 152,   0),
          (255,  87,  34), (121,  85,  72), (158, 158, 158)]


##########################################################################################
def hsv_contours(im, low, high, thd=625, thd_bi=10):
    # 筛选出位于两个数组之间的元素
    assert low <= high, f'{low} not <= {high}'
    img_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    color = cv2.inRange(img_hsv, low, high) # (h,w)
    if low[0]<0: # for Hue: its periodicity
        low, high = (low[0]+180, *low[1:]), (180, *high[1:])
        color = cv2.add(color, cv2.inRange(img_hsv, low, high))
    # 设置非掩码检测部分全为黑色
    mask = im.copy(); mask[color==0] = 0 # (h,w,c)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) # (h,w)
    # 获取不同形状的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # 形态学开/闭操作: 形态学变换消除细小黑点/连通细小空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 图像二值化操作：# mask = (mask>thd_bi).astype('uint8')*255
    _, mask = cv2.threshold(mask, thd_bi, 255, cv2.THRESH_BINARY)
    # 获取轮廓点集(坐标)：python2和python3在此处略有不同
    method = cv2.CHAIN_APPROX_NONE # 存储所有的轮廓点，相邻两个点的像素位置差不超过1
    # cv2.CHAIN_APPROX_SIMPLE # 压缩水平+垂直+对角线方向的元素，只保留该方向的终点坐标
    # cv2.CHAIN_APPROX_TC89_L1和cv2.CHAIN_APPROX_TC89_KCOS：使用teh-Chinl chain 近似算法
    contours, heriachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, method) # python3
    #_, contours, heriachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, method) # python2
    # contours = [cnt.squeeze() for cnt in contours if cv2.contourArea(cnt)>thd]

    masks, result = [], [] # filter by area
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    bg = np.zeros((*mask.shape[:2],3), mask.dtype)
    for cnt in contours:
        if cv2.contourArea(cnt)>thd: # area_threshold
            #cv2.drawContours(bg.copy(), [cnt], 0, (5,)*3, -1)
            m = cv2.fillPoly(bg.copy(), [cnt], (5,)*3)[...,0]>0
            mk = bg[...,0].copy(); mk[m] = mask[...,0][m]
            masks.append(mk>0) # instantiate mask: bool
            result.append(cnt.squeeze()) # squeeze
        else: cv2.fillPoly(mask, [cnt], (0,0,0)) # erase
        # cv2.drawContours(mask, [cnt], 0, (0,0,0), -1)
    return result, masks, mask[...,0]


def fit_contours(contours, mod='R', thd=625):
    result = []
    for cnt in contours:
        if len(cnt.shape)<2: continue # 1-vertex
        mm = cv2.moments(cnt) # 多边形的矩
        if mm['m00']==0: continue # 多边形中心
        ct = mm['m10']/mm['m00'], mm['m01']/mm['m00']
        area = cv2.contourArea(cnt)
        if area > thd: # area_threshold
            if type(mod)==str and mod in 'Cc': # 最小包围圆形
                fit = cv2.minEnclosingCircle(cnt) # (center,radius)
            elif type(mod)==str and mod in 'Ee': # 最优拟合椭圆
                fit = cv2.fitEllipse(cnt) # (center,axes,theta)
            elif type(mod)==str and mod in 'Rr': # 最小包围矩形
                fit = cv2.minAreaRect(cnt) # (center,(w,h),theta)
                #fit = cv2.boxPoints(fit) # convert: [4-vertex]
            elif type(mod)==str and mod in 'Tt': # 最优外包三角形
                _, fit = cv2.minEnclosingTriangle(cnt) # (area,3-vertex)
            elif type(mod) in (int,float) and mod>0: # 逼近多边形
                s = area/cv2.arcLength(cnt, closed=True)
                fit = cv2.approxPolyDP(cnt, mod*s, True).squeeze()
            result.append([ct,fit]) # [center,fit]
    return result


A, B = np.random.randint(0, 33, 2)
ADW = lambda x,c,s,g=0: (x*s+(1-s)*c+g).clip(0,255).astype('uint8')
########################################################
def mask_instance(im, mask, color, box, txt, s=0.65):
    im[mask>0] = ADW(im[mask>0], np.asarray(color), s)
    x,y,w,h = box; cv2.rectangle(im, (x,y), (x+w,y+h), color, 1)

    fc, fs, ftk, lnt = 2, 0.6, 1, cv2.LINE_AA
    w,h = cv2.getTextSize(txt, fc, fs, ftk)[0]
    cv2.rectangle(im, (x,y), (x+w,y-h-4), color, -1)
    #im[y-h-4:y,x:x+w] = (im[y-h-4:y,x:x+w]*s).astype('uint8')
    cv2.putText(im, txt, (x,y-3), fc, fs, (255,)*3, ftk, lnt)
    return im # alter im


def mask_instances(im, masks):
    if type(masks)==list: # list of dicts
        for m in masks: # [cls, cid, box, cnt, mask]
            key, cid, box, cnt, mk = list(m.values())
            color = COLORS[(A*cid+B) % len(COLORS)]
            im = mask_instance(im, mk, color, box, key)
    elif type(masks)==dict: # dict of lists
        for key, cid, box, cnt, mk in zip(*masks.values()):
            color = COLORS[(A*cid+B) % len(COLORS)]
            im = mask_instance(im, mk, color, box, key)
    return im # alter im


INT = lambda x,f=1: tuple(int(i*f) for i in x)
RSZ = lambda x,s: cv2.resize(x, None, fx=s, fy=s)
##########################################################################################
def get_fits(im, cls, mod='R', ht=480):
    result = []; #im = RSZ(im, ht/im.shape[0])
    mask = np.zeros(im.shape[:2], im.dtype)
    for i, (key,val) in enumerate(cls.items()):
        contours, masks, mk = hsv_contours(im, *val)
        approx = fit_contours(contours, mod) # thd accord
        for (ct, fit), mb in zip(approx, masks):
            result.append(dict(cls=key, cid=i, center=ct, fit=fit, mask=mb))
        if approx: mask = cv2.add(mask, mk); #cv2.imshow(key, mk)
    for i in result: # list of dicts
        key, cid, ct, fit, mk = list(i.values())
        color = COLORS[(A*cid+B) % len(COLORS)]
        cv2.circle(im, INT(ct), 5, color, -1)
        if type(mod)==str and mod in 'Cc':
            ct, sh = INT(fit[0]), int(fit[1])
            cv2.circle(im, ct, sh, color, 2)
            tc = tuple(np.int0(ct)-12)
        elif type(mod)==str and mod in 'Ee':
            cv2.ellipse(im, fit, color, 2)
            tc = tuple(np.int0(ct)-12)
        elif type(mod)==str and mod in 'Rr':
            sh = np.int0(cv2.boxPoints(fit)) # [4-vertex]
            cv2.drawContours(im, [sh], 0, color, 2)
            #cv2.polylines(im, [sh], True, color, 2)
            tc = tuple(min(sh, key=lambda x: sum(x))-5)
        else:
            sh = np.int0(fit.squeeze()) # [n-vertex]
            cv2.drawContours(im, [sh], 0, color, 2)
            #cv2.polylines(im, [sh], True, color, 2)
            tc = tuple(min(sh, key=lambda x: sum(x))-5)
        im = mask_instance(im, mk, color, (*tc,0,0), key)
    return im, result, mask


def test_fits(src, t=300):
    from glob import glob
    for i in sorted(glob(f'{src}/*.jpg'))\
        if os.path.isdir(src) else [src]:
        for x in ['C', 'E', 'R', 0.5, 1.0]:
            im = cv2.imread(i)
            im, res, mk = get_fits(im, HSV_Set, x)
            if t<1: print(f'{i}:\n{res}\n'); cv2.imshow('mk',mk)
            cv2.imwrite(i[:-4]+f'_{x}'+'.png', im)
            cv2.imshow('hsv', im); cv2.waitKey(t)
        cv2.destroyAllWindows()


import rospy; from cv_bridge import CvBridge; CVB = CvBridge()
D2L = lambda x: [{k:v for k,v in zip(x.keys(),i)} for i in zip(*x.values())]
L2D = lambda x: {k:v for k,*v in zip(x[0].keys(),*[i.values() for i in x])} if x else {}
##########################################################################################
def get_instances(im, cls, mod=[], s=1):
    result = []; #im = RSZ(im, ht/im.shape[0])
    for i, (key,val) in enumerate(cls.items()):
        contours, masks, mk = hsv_contours(im, *val, 625/s)
        for cnt, mk in zip(contours, masks):
            cnt, box = cnt.transpose(), cv2.boundingRect(cnt) # (x,y,w,h)
            result.append(dict(cls=key, cid=i, box=box, cnt=cnt, mask=mk))
    if type(mod)==dict: result = L2D(result) # list->dict
    return mask_instances(im,result), result # image


def format_msg(src, Obj, s=2):
    assert type(Obj)==type and 'msg' in str(Obj)
    from sensor_msgs.msg import RegionOfInterest
    from geometry_msgs.msg import Point32, Polygon

    hd = hasattr(src,'header')
    if type(src)==str: im = cv2.imread(src)
    elif type(src)==np.ndarray: im = src.copy()
    elif hd: im = CVB.compressed_imgmsg_to_cv2(src)

    im, res = get_instances(im, HSV_Set, {}, s)
    ob = Obj(header=src.header) if hd else Obj()
    if not res: return ob, im, res

    if hasattr(Obj,'class_ids'): ob.class_ids = res['cid']
    if hasattr(Obj,'class_names'): ob.class_names = res['cls']
    if hasattr(Obj,'scores'): ob.scores = [1.0]*len(res['cls'])
    if hasattr(Obj,'boxes'): ob.boxes = [RegionOfInterest(x_offset=x,
        y_offset=y, width=w, height=h) for (x,y,w,h) in res['box']]
    if hasattr(Obj,'contours'): ob.contours = [Polygon([Point32(
        x=x, y=y) for (x,y) in ct.transpose()]) for ct in res['cnt']]
    elif hasattr(Obj,'masks'): # for masks
        for mk in res['mask']: # get masks
            m = CVB.cv2_to_imgmsg(mk.astype('uint8')*255, 'mono8')
            m.header = ob.header; ob.masks.append(m)
    return ob, im, res


'''std_msgs/Header header # Obj header
int32[] class_ids # Integer class IDs for each bounding box
string[] class_names # String class IDs for each bouding box
float32[] scores # Float probability scores of the class_id
# http://docs.ros.org/en/api/sensor_msgs/html/msg/RegionOfInterest.html
sensor_msgs/RegionOfInterest[] boxes # Bounding boxes in pixels
# https://docs.ros.org/en/api/geometry_msgs/html/msg/Polygon.html
geometry_msgs/Polygon[] contours # Instance contours as Polygon
# http://docs.ros.org/en/api/sensor_msgs/html/msg/Image.html
#sensor_msgs/Image[] masks # Instance masks as Image'''
########################################################
def publish_msg(Obj, out='/glodon', sub=''):
    #from glodon_msgs.msg import Object as Obj
    #from mask_rcnn_ros.msg import Result as Obj
    from sensor_msgs.msg import CompressedImage, Image

    def get(x): global src; src = x # callback
    rospy.init_node(out[1:], anonymous=True) # init
    sub = sub if sub else '/camera/color/image_raw/compressed'
    sub = rospy.Subscriber(sub, CompressedImage, get, queue_size=1)
    pim = rospy.Publisher(f'{out}/compressed', CompressedImage, queue_size=1)
    pub = rospy.Publisher(f'{out}/result', Obj, queue_size=1)

    while cv2.waitKey(5)!=27:
        if not hasattr(src,'header'): continue
        ob, im, *_ = format_msg(src,Obj); pub.publish(ob)
        pim.publish(CVB.cv2_to_compressed_imgmsg(im))
        cv2.imshow('hsv', im) #'''
    '''from glob import glob
    for i in sorted(glob(f'{src}/*.jpg')):
        ob, im, *_ = format_msg(i,Obj); pub.publish(ob)
        pim.publish(CVB.cv2_to_compressed_imgmsg(im))
        cv2.imshow('hsv', im); cv2.waitKey() #'''
    cv2.destroyAllWindows()


##########################################################################################
if __name__ == '__main__':
    src = 'test'; #test_fits(src)
    from glodon_msgs.msg import Object
    from mask_rcnn_ros.msg import Result
    #publish_msg(Result, '/mask_rcnn')
    publish_msg(Object, '/glodon')

