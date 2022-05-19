#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, cv2, json


##########################################################################################
class Camera:
    def __init__(self, src, idx=0):
        with open(src) as f: info = json.load(f)
        idx = max(0, min(idx,len(info)))
        info = list(info.values())[idx]
        self.width = info.get('width', 0)
        self.height = info.get('height', 0)
        self.size = max(self.width, self.height)
        self.project = info.get('projection_type', '')
        if 'focal' not in info:
            self.focal_x = info.get('focal_x', 0)
            self.focal_y = info.get('focal_y', 0)
        else: self.focal_x = self.focal_y = info['focal']
        self.c_x = info.get('c_x', 0)
        self.c_y = info.get('c_y', 0)
        self.k1 = info.get('k1', 0)
        self.k2 = info.get('k2', 0)
        self.k3 = info.get('k3', 0)
        self.p1 = info.get('p1', 0)
        self.p2 = info.get('p2', 0)

    # project the normalized coordinates onto Z=1 plane
    def norm(self, x, y): return x/self.focal_x, y/self.focal_y

    def undistort(self, x, y): # undistort/correct
        xn, yn = self.norm(x,y); r2 = xn*xn + yn*yn
        dr = 1 + self.k1*r2 + self.k2*r2**2 + self.k3*r2**3
        dx = 2*self.p1*xn*yn + self.p2*(r2 + 2*xn)
        dy = 2*self.p2*xn*yn + self.p1*(r2 + 2*yn)
        u = self.focal_x*(dr*xn + dx) + self.c_x
        v = self.focal_y*(dr*yn + dy) + self.c_y
        u = 2*x - u; v = 2*y - v # reverse calibrate
        return u, v

    def K(self): # K-matrix
        sz = self.size; fx, fy = sz*self.focal_x, sz*self.focal_y
        cx, cy = self.width/2+sz*self.c_x, self.height/2+sz*self.c_y
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def DistCoeffs(self):
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])


##########################################################################################
def load_feature(im, cam=0, un=0):
    xy = np.load(im+'.features.npz')['points'].T[:2]
    if type(cam)==str: cam = Camera(cam)
    if type(cam)==Camera:
        if un: xy = cam.undistort(*xy)
        xy = cam.norm(*xy) # (2,N)
    return np.array(xy).T # (N,2)


def calc_axis_cam(translation, rotation):
    R,J = cv2.Rodrigues(-rotation)
    O = R.dot(-translation) # cam origin
    A = np.diag([1.0]*3) # cam axis
    for i in range(len(A)): # X,Y,Z
        A[i] = R.dot(A[i]-translation)-O
    return (O,*A) # O,X,Y,Z


import logging as log; INFO = log.getLogger('Joshua').info
log.basicConfig(level=log.INFO, format='%(asctime)s %(levelname)s: %(message)s')
##########################################################################################
def filter_reconstruct(src, thd=0.3):
    src = os.path.abspath(src); res = {}
    if os.path.isdir(src+'/opensfm'): # for odm
        cam = src+'/cameras.json'; src += '/opensfm'
        rec = src+'/reconstruction.topocentric.json'
    elif os.path.isfile(src+'/../cameras.json'):
        cam = src+'/../cameras.json' # for odm
        rec = src+'/reconstruction.topocentric.json'
    elif os.path.isfile(src+'/camera_models.json'):
        cam = src+'/camera_models.json' # for sfm
        rec = src+'/reconstruction.json'
    cam = Camera(cam); bak = rec[:-4]+'bak'
    if os.path.isfile(bak):
        if os.path.isfile(rec): os.remove(rec)
        os.rename(bak, rec) # for win
    with open(rec) as f: data = json.load(f)[0]
    os.rename(rec, bak); INFO(f'Filter: {rec}')

    from opensfm.dataset import DataSet
    T = DataSet(src).load_tracks_manager()
    for im in T.get_shot_ids():
        v = data['shots'][im]
        rotation =  np.array(v['rotation'])
        translation = np.array(v['translation'])
        O, X, Y, Z = calc_axis_cam(translation, rotation)
        feat = load_feature(f'{src}/features/{im}', cam, 1)
        for tid,x in T.get_shot_observations(im).items():
            if not tid in data['points']: continue
            dp = data['points'][tid]['coordinates']-O
            ddp = np.linalg.norm(dp)
            u,v = feat[x.id][:2] # fid
            qt = u*X + v*Y + Z
            qt /= np.linalg.norm(qt)
            delta = np.dot(dp, qt)
            dis = np.sqrt(ddp**2 - delta**2)
            if tid not in res: res[tid] = dis
            elif dis>res[tid]: res[tid] = dis # meters
            #print(f'{im} %6s %6s %.3f'%(tid,x.id,dis))
    dis = [*res.values()]; md = np.mean(dis); thd = min(thd,md)
    out = {k:v for k,v in res.items() if v>thd}; #print(out)
    #plt.hist(dis, [0.01,0.05,0.1,0.5,1,2]); plt.show()
    for tid in out: data['points'].pop(tid)
    with open(rec,'w') as f: json.dump([data], f, indent=4)
    INFO('Out=%d/%d, Thd=%.3f, Max=%.3f'%(len(out),len(res),thd,max(dis)))


##########################################################################################
if __name__ == '__main__':
    RTK = 'data_part/out-sift/odm-RTK-20211108'
    #filter_reconstruct(RTK, 0.3)

