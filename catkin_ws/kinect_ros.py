#!/usr/bin/python3
# coding: utf-8

import numpy as np
import scipy.linalg as la
import os, sys, cv2, json
sys.path.append('openpose')
from openpose import openpose
from time import time

import rospy, roslib
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
# cv_bridge does not support CompressedImage in python
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, LaserScan
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer


KeyPoints = ['nose','chest',
            'R_shoulder','R_elbow','R_wrist', 'L_shoulder','L_elbow','L_wrist',
            'R_hip','R_knee','R_ankle', 'L_hip','L_knee','L_ankle',
            'R_eye','R_ear','L_eye','L_ear']
KeyPoints = {i:k for i,k in enumerate(KeyPoints)}
##########################################################################################
class ros_det:
    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        rospy.init_node('ros_det', anonymous=True)

        self.rgbd = None; self.pose = openpose('openpose')
        cam_info = TPC(['/rgb/camera_info','/zed_node/left/camera_info'])
        img_depth = TPC(['/depth_to_rgb/image_raw','/zed_node/depth/depth_registered'])
        img_rgb = TPC(['/rgb/image_raw/compressed','/zed_node/left/image_rect_color/compressed'])
 
        img_depth = Subscriber(img_depth, Image, queue_size=1)
        img_rgb = Subscriber(img_rgb, CompressedImage, queue_size=1)

        rospy.Subscriber(cam_info, CameraInfo, self.callback_info)
        sync = ApproximateTimeSynchronizer([img_rgb, img_depth],
                    queue_size=5, slop=1, allow_headerless=True)
        sync.registerCallback(self.callback_multi)#'''


    def callback_info(self, cam):
        self.K = cam.K


    def callback_multi(self, rgb, depth):
        self.rgbd = rgb, depth


    def process_img(self, rgb, depth):
        rgb = np.frombuffer(rgb.data, 'uint8')
        rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)

        depth = CvBridge().imgmsg_to_cv2(depth, '32FC1')
        depth = np.where(abs(depth)<np.inf, depth, 0)

        '''dp = (depth/depth.max()*255).astype('uint8')
        dp = cv2.cvtColor(dp, cv2.COLOR_GRAY2BGR)
        fuse = cv2.addWeighted(dp, 0.5, rgb, 0.5, 0)
        fuse = cv2.resize(fuse, None, fx=0.5, fy=0.5)
        cv2.imshow('fuse', fuse); cv2.waitKey(1)#'''
        
        '''v,u = np.array([i//2 for i in depth.shape[:2]])
        pt = Camera_2Dto3D(u, v, depth, self.K)
        print('2D: [%d,%d,%.3f] -> 3D:'%(u,v,depth[v,u]), pt)#'''

        img = crop(rgb, mg=1/4, s=2, dim=1)
        body, ids, hands = self.pose.process_image(img)
        if body is None: return rgb

        uv = crop(rgb, mg=1/4, s=2, dim=1, uv=get_uv(body,ids))
        body[...,:2] = crop(rgb, mg=1/4, s=2, dim=1, uv=body[...,:2])

        pt = []; depth[0,0] = 0
        for x in uv: # uv->3d: pt(N,18,3)
            pt += [Camera_2Dto3D(*x.T, depth, self.K).T]
        pt = np.asarray(pt)*100 # meter->centimeter
        if len(pt)<1: return rgb

        z = np.array(pt)[...,2]
        nz = np.count_nonzero(z, axis=1)
        z = z.sum(axis=1)/np.where(nz>0,nz,1)
        if min(z)<1: return rgb
        pid = z.argmin(); ids = ids[pid:pid+1]
        img = self.pose.draw_result(rgb, body, ids, hands)

        for n,x in enumerate(pt):
            if n!=pid: continue
            print('person_%d:'%n, get_height(x))
            '''for i,p in enumerate(x):
                p = '[%6.1f, %6.1f, %6.1f]' % tuple(p)
                print('%2d %10s:'%(i,KeyPoints[i]), p)#'''
        return img


##########################################################################################
def TPC(tp): # pick existed topic
    while True:
        tps = rospy.get_published_topics()
        tps = {t[0]:t[1] for t in tps}
        x = [i for i in tp if i in tps]
        if len(x)>0: return x[0]


def parse_depth(depth, mod='F'):
    # Filter nan/inf: (-inf, inf, nan)->0
    #np.seterr(divide='ignore', invalid='ignore')
    # right = CvBridge().imgmsg_to_cv2(rgb, 'bgr8')
    depth = CvBridge().imgmsg_to_cv2(depth, '32FC1')
    depth = np.where(abs(depth)<np.inf, depth, 0)
    if 'F' in mod: return depth # float, meters
    # np.astype('uint8') can also filter nan/inf
    return (depth/depth.max()*255).astype('uint8')


def Camera_2Dto3D(u, v, dp, CK):
    u,v = [np.asarray(i,int) for i in (u,v)]
    dp = np.asarray(dp); shp = dp.shape
    fx,_,cx,_,fy,cy = CK[:6] # cam intrinsics
    z = dp[v,u] if len(shp)>1 else dp # meter
    z = z * (1E-3 if z.dtype==np.uint16 else 1)
    x = z * (u-cx)/fx; y = z * (v-cy)/fy
    return np.asarray([x,y,z]) #(3,N)


def crop(im, mg=0, s=1, dim=1, uv=None):
    assert dim in (0,1) # mg=margin
    w = im.shape[dim]; m = int(w*mg)
    if uv is None: # crop->resize
        return cv2.resize(im[:,m:w-m], None, fx=1/s, fy=1/s)
    elif type(uv)==np.ndarray: #(N,18,2)
        if uv.shape[-1]!=2: return uv
        sh = [1]*(uv.ndim-1)+[uv.shape[-1]]
        m = np.ones(sh)*m; m[...,dim] = 0
        return (uv*s + m).astype(int)


def get_uv(body, ids):
    uv = np.zeros((len(ids),18,2))
    for n in range(len(ids)):
        for i in range(18):
            index = int(ids[n][i])
            if index==-1: continue
            uv[n,i] = body[index][0:2]
    return uv #(N,18,2)


def get_height(pt, fs=166/145.92465): #(18,3)
    isz = np.abs(pt[...,2])<np.finfo(float).eps
    chest = isz[1] # need: (1) nonzero
    eye = isz[14], isz[15] # need: either (14,15) nonzero
    # need: all (8,9,10) nonzero, or all (11,12,13) nonzero
    leg = sum(isz[8:11])>0, sum(isz[11:14])>0
    # (1) + (8,9,10)|(11,12,13) + (14)|(15)
    if chest or sum(eye)>1 or sum(leg)>1: return

    seq = pt[8:11], pt[11:14], pt[14], pt[15]
    seq = [distance(np.vstack([pt[1],i])) for i in seq]
    eye = seq[3] if eye[0] else seq[2] if eye[1] else (seq[2]+seq[3])/2
    leg = seq[1] if leg[0] else seq[0] if leg[1] else (seq[0]+seq[1])/2
    return (eye+leg)*fs, seq


dis = lambda x: np.linalg.norm(x)
def distance(p): #(n,3)
    L = 0
    if p.ndim<2 or p.shape[-2]<2:
        return dis(p)
    for i in range(len(p)-1):
        L += dis(p[i+1,:]-p[i,:])
    return L


def match(pt): # pt:(N,dim)
    sol, r, rank, s = la.lstsq(pt[:,:-1], pt[:,-1])
    return sol # res = pt[:,:-1].dot(sol)



##########################################################################################
if __name__ == '__main__':
    det = ros_det(); k = True
    while k!=27:
        t = time()
        if det.rgbd is not None:
            rgb, dp = det.rgbd
            rgb = det.process_img(rgb, dp)
            cv2.imshow('pose', rgb)
            k = cv2.waitKey(1)
            print('total: %.2f ms\n'%((time()-t)*1000))

