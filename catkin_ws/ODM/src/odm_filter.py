#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, cv2, json
from matplotlib import pyplot as plt
from matplotlib import use; use('TkAgg')


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

    '''with open(src+'/tracks.csv') as f:
        tracks = f.readlines()[1:]
    for tk in tracks: # skip 1st-row
        im, tid, fid, *x = tk.split()
        #u,v = cam.norm(*np.float64(x[:2]))
        if im != old: # update feat+OXYZ
            v = data['shots'][im]; old = im
            rotation =  np.array(v['rotation'])
            translation = np.array(v['translation'])
            O, X, Y, Z = calc_axis_cam(translation, rotation)
            feat = load_feature(f'{src}/features/{im}', cam, 1)
        if not tid in data['points']: continue
        dp = data['points'][tid]['coordinates']-O
        ddp = np.linalg.norm(dp)
        u,v = feat[int(fid)][:2]
        qt = u*X + v*Y + Z
        qt /= np.linalg.norm(qt)
        delta = np.dot(dp, qt)
        dis = np.sqrt(ddp**2 - delta**2)
        if tid not in res: res[tid] = dis
        elif dis>res[tid]: res[tid] = dis # meters
        #print(f'{im} %6s %6s %.3f'%(tid,fid,dis))'''

    dis = [*res.values()]; md = np.mean(dis); thd = min(thd,md)
    out = {k:v for k,v in res.items() if v>thd}; #print(out)
    #plt.hist(dis, [0.01,0.05,0.1,0.5,1,2]); plt.show()
    for tid in out: data['points'].pop(tid)
    with open(rec,'w') as f: json.dump([data], f, indent=4)
    INFO('Out=%d/%d, Thd=%.3f, Max=%.3f'%(len(out),len(res),thd,max(dis)))


##########################################################################################
def check_gcp(gcp, cam, org=0, n=0):
    res = {}; K = Camera(cam).K()
    if os.path.isdir(org):
        from opensfm.dataset import DataSet
        ref = DataSet(org).load_reference()
    with open(gcp) as f: data = f.readlines()[n:]
    for v in data: # skip first n-rows
        v = v.split(); im = v[-1]
        v = v[:5] + [np.inf]*2
        if os.path.isdir(org): # lat,lon.alt->xyz
            lon,lat,alt = [float(i) for i in v[:3]]
            v[:3] = ref.to_topocentric(lat,lon,alt)
        if im not in res: res[im] = [v]
        else: res[im].append(v)

    for k,v in res.items():
        v = res[k] = np.float64(v)
        if len(v)<5: continue # skip
        pt, uv = v[:,:3].copy(), v[:,3:5] # copy()->new mem-block
        _, Rvec, Tvec, Ins = cv2.solvePnPRansac(pt, uv, K, None)
        xy, Jacob = cv2.projectPoints(pt, Rvec, Tvec, K, None)
        err = v[:,5] = np.linalg.norm(xy.squeeze()-uv, axis=1)

        his = np.histogram(err, bins=[*range(11),np.inf])[0]
        for c in range(len(his)-1,0,-1): # len(v)=sum(his)
            if sum(his[c:])>=len(v)*0.2: break
        idx = np.where(err<=c)[0]; #print(c, his)
        if len(idx)<7: continue # skip
        _, Rvec, Tvec = cv2.solvePnP(pt[idx], uv[idx], K, None)
        xy, Jacob = cv2.projectPoints(pt, Rvec, Tvec, K, None)
        v[:,-1] = np.linalg.norm(xy.squeeze()-uv, axis=1) # err2

    out = os.path.abspath(gcp+'.err'); print(out)
    with open(out,'w') as f:
        for k,v in zip(data, np.vstack([*res.values()])):
            f.write(k[:-1]+'%11.3f%11.3f\n'%(*v[-2:],))


import gzip, pickle
##########################################################################################
def export_gz_check(src, check, dst, thd=1, src2=''):
    if os.path.isfile(src+'/camera_models.json'): # sfm
        cam1 = Camera(src+'/camera_models.json', 0) # 0=GPS
        cam2 = Camera(src+'/camera_models.json', 1) # 1=RTK
    elif os.path.isfile(src+'/cameras.json'): # odm
        cam1 = cam2 = Camera(src+'/cameras.json'); src += '/opensfm'
        if os.path.isfile(src2+'/cameras.json'): # RTK
            cam2 = Camera(src2+'/cameras.json')
    mt_dir = src+'/matches'+('' if os.path.isdir(src+'/matches') else '_gcp')
    info = 4*'%9.6f '%(cam1.focal_x, cam1.focal_y, cam2.focal_x, cam2.focal_y)+'\n'
    info += 4*'%9d '%(cam1.width, cam1.height, cam2.width, cam2.height)+'\n'

    os.makedirs(dst, exist_ok=True); res = {}
    for gz in os.scandir(mt_dir):
        im1 = gz.name[:-15]
        ft1 = load_feature(f'{src}/features/{im1}', cam1)
        with gzip.open(gz.path) as f: MT = pickle.load(f)
        for im2, fid in MT.items():
            if len(fid)<1: continue # skip empty
            ft2 = load_feature(f'{src}/features/{im2}', cam2)
            txt = f'{dst}/{im1}-{im2}.txt'; js = txt[:-3]+'json'
            with open(txt, 'w') as f: # x0, y0, x1, y1
                f.write(f'{len(fid)}\n'+info)
                for i,j in fid: f.write(4*'%9.6f '%(*ft1[i],*ft2[j])+'\n')
            os.system(f'./{os.path.relpath(check)} {txt} {thd} {js}')
            #os.system(f'./{os.path.relpath(check)} {txt} {thd}')
            if not os.path.isfile(js): print(f'No: {js}'); continue
            with open(js,'r') as f: x = json.load(f)
            res[im1,im2] = [int(k[1:]) for k in x['errMatch']]
            if src2: show_err_match(src, txt, src2) # need txt+js
    return res # idx of invalid fid of gz


##########################################################################################
def show_err_match(src, file, src2='', sh=5):
    if file.endswith('.txt'):
        txt, js = file, file[:-3]+'json'
    elif file.endswith('.json'):
        txt, js = file[:-4]+'txt', file
    if not os.path.isdir(src2): src2 = src
    if os.path.isdir(src+'/../images'): src += '/..'
    if os.path.isdir(src2+'/../images'): src2 += '/..'
    img0, img1 = os.path.basename(txt)[:-4].split('-')
    img0 = cv2.imread(f'{src}/images/{img0}')
    img1 = cv2.imread(f'{src2}/images/{img1}')
    sz0, sz1 = max(img0.shape), max(img1.shape)

    with open(txt,'r') as f:
        v = f.readline(); pts = []
        fx0, fy0, fx1, fy1 = np.float64(f.readline().split()[:4])
        w0, h0, w1, h1 = np.int32(f.readline().split()[:4])
        for v in f.readlines(): pts += [np.float64(v.split()[:4])]
        pts = np.array(pts); print(txt, js)
    assert (*img0.shape[:2],*img1.shape[:2])==(h0,w0,h1,w1)

    with open(js,'r') as f: x = json.load(f)
    em = x['errMatch']; err = float(x['err'])
    #mat0 = np.float64(x['Matrix0'].split()).reshape(3,-1)
    #mat1 = np.float64(x['Matrix1'].split()).reshape(3,-1)

    from odm_sfm import roi_ct2
    for k,v in em.items():
        pt = pts[int(k[1:])]
        x0 = int(pt[0]*fx0*sz0 + w0/2)
        y0 = int(pt[1]*fy0*sz0 + h0/2)
        x1 = int(pt[2]*fx1*sz1 + w1/2)
        y1 = int(pt[3]*fy1*sz1 + h1/2) # im: (sz+bt, sz*2, 3)
        im = roi_ct2(img0, img1, (x0,y0), (x1,y1), sz=501, bt=30)
        draw_center_cross2(im, (0,0,255), 6, f'{k}: {v}')
        cv2.imshow(os.path.basename(js), im)
        if cv2.waitKey(sh) in (27,32): break
    cv2.destroyAllWindows()


def draw_center_cross2(im, color, r, txt=''):
    # im = np.zeros([sz+bt, sz*2, 3], 'uint8')
    h,w = im.shape[:2]; sz = w//2; hf, bt = sz//2, h-sz
    cv2.line(im, (0, hf), (hf-r, hf), color, 1)
    cv2.line(im, (hf+r, hf), (sz, hf), color, 1)
    cv2.line(im, (hf, 0), (hf, hf-r), color, 1)
    cv2.line(im, (hf, hf+r), (hf, sz), color, 1)
    cv2.circle(im, (hf, hf), r, color, lineType=cv2.LINE_AA)
    cv2.line(im, (sz, hf), (sz+hf-r, hf), color, 1)
    cv2.line(im, (sz+hf+r, hf), (sz+sz, hf), color, 1)
    cv2.line(im, (sz+hf, 0), (sz+hf, hf-r), color, 1)
    cv2.line(im, (sz+hf, hf+r), (sz+hf, sz), color, 1)
    cv2.circle(im, (sz+hf, hf), r, color, lineType=cv2.LINE_AA)
    cv2.line(im, (sz, 0), (sz, sz), (0,0,0), 5)
    if txt: cv2.putText(im, txt, (0, sz+bt-5), 4, 0.8, color, 1)


##########################################################################################
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # os.chdir('data_part/out-sift'); GPS = 'odm-GPS-20211109'
    # RTK = 'odm-RTK-20211108'; filter_reconstruct(RTK, 0.3)

    '''exe = 'checker'; GCP = 'sfm-GCP-20211108-20211109'
    os.system(f'g++ check_gz.cpp -o {exe}')
    #export_gz_check(GPS, exe, 'tmp', src2=1)
    export_gz_check(GCP, exe, 'tmp', src2=RTK)#'''

    os.chdir('/media/hua/20643C51643C2BC2/odm/iter_none')
    src = 'odm-GPS-20211108-20211109/opensfm'
    gcp = src+'/gcp_list.txt'; cam = src+'/camera_models.json'
    check_gcp(gcp, cam, 'odm-RTK-20211108/opensfm')

