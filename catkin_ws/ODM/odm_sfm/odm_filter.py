#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, cv2, json
from odm_sfm import INFO


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


##########################################################################################
def load_feature(img, cam=0, fix=0):
    xy = np.load(f'{img}.features.npz')['points'].T[:2]
    if type(cam)==str: cam = Camera(cam)
    if type(cam)==Camera:
        if fix: xy = cam.undistort(*xy)
        xy = cam.norm(*xy) # (2,N)
    return np.array(xy).T # (N,2)


##########################################################################################
def cal_cam_axis(translation, rotation):
    R, J = cv2.Rodrigues(-rotation)
    O = R.dot(-translation) # cam origin
    A = np.diag([1.0]*3) # cam axis
    for i in range(len(A)): # X,Y,Z
        A[i] = R.dot(A[i]-translation)-O
    return (O,*A) # O,X,Y,Z


def filter_reconstruct(odm, thd=0.5):
    cam = Camera(odm+'/cameras.json')
    rec = '/opensfm/reconstruction.topocentric.json'
    if os.path.isfile(odm+rec[:-4]+'bak'):
        os.rename(odm+rec[:-4]+'bak', odm+rec)
    with open(odm+rec) as f: data = json.load(f)[0]
    points = data['points']; shots = data['shots']

    res = {}; last = ''; INFO(odm+rec)
    with open(odm+'/opensfm/tracks.csv') as f:
        tracks = f.readlines()[1:]
    for tk in tracks:
        tk = tk.split(); img, tid, fid = tk[:3]
        #u,v = cam.norm(*np.float64(tk[3:5]))
        if not tid in points: continue
        # some tid not in the points-cloud
        pt = points[tid]['coordinates']

        if img != last:
            last = img; vt = shots[img]
            rotation =  np.array(vt['rotation'])
            translation = np.array(vt['translation'])
            gps_position = np.array(vt['gps_position'])
            O, X, Y, Z = cal_cam_axis(translation, rotation)
            feat = load_feature(f'{odm}/opensfm/features/{img}', cam, 1)

        dp = pt - O
        d1 = np.linalg.norm(dp)
        u, v = feat[int(fid)][:2]
        qt = u*X + v*Y + Z
        qt /= np.linalg.norm(qt)
        delta = np.dot(dp, qt)
        dis = np.sqrt(d1*d1 - delta*delta)
        if tid not in res: res[tid] = dis
        elif dis>res[tid]: res[tid] = dis # meters
        #if dis>thd: print(f'{img} %6s %6s %.3f'%(tid,fid,dis))
    dis = [*res.values()]; md = np.mean(dis); thd = min(thd,md)
    out = {k:v for k,v in res.items() if v>thd}; #print(out)
    INFO('Out=%d/%d, Thd=%.3f, Max=%.3f'%(len(out),len(res),thd,max(dis)))

    for tid in out: data['points'].pop(tid)
    os.rename(odm+rec, odm+rec[:-4]+'bak')
    with open(odm+rec,'w') as f: json.dump([data], f, indent=4)
    INFO(f'Filter: {odm+rec}'); return out


##########################################################################################
if __name__ == '__main__':
    RTK = 'data_part/out-sift/odm-RTK-20211108'
    #filter_reconstruct(RTK)

