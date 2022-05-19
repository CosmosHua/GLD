#!/usr/bin/python3
# coding: utf-8

import os, numpy as np


MAX = 0; RES = []; SIGN = []
##########################################################################################
def CalHash(sign):
    idx = 0; k = 1
    P = 97; Q = 133
    for ch in sign:
        m = ord(ch)*k; k += 1
        idx += (P*m) % MAX
        idx = (idx*Q) % MAX
    return idx


def FindSign(sign):
    idx = CalHash(sign)
    while SIGN[idx] != sign:
        if SIGN[idx] == '': return 0, ''
        idx += 1
        if idx == MAX: idx = 0
    return 1, RES[idx]


def AddSign(sign, x):
    idx = CalHash(sign)
    while SIGN[idx] != '':
        idx += 1
        if idx == MAX: idx = 0
    SIGN[idx] = x


Dis = [[0]]; V = [np.zeros(3)]; VT = [np.zeros(2)]; VN = [np.zeros(3)]
##########################################################################################
def Intersect(fq, r1, r2):
    if MAX > 0:
        sign = r1 + '_' + r2
        tag, res = FindSign(sign)
        if tag: return res

    global Dis, V, VT, VN
    p1 = r1.split('/'); p2 = r2.split('/')
    n1 = int(p1[0]); n2 = int(p2[0])
    t = (0 - Dis[n1])/(Dis[n2] - Dis[n1])
    
    pt = V[n1] + (V[n2] - V[n1])*t
    V += [np.array(pt)]
    fq.write('v %7.3f %7.3f %7.3f\n'%(*pt[:3],))
    res = str(len(V) - 1)

    if (len(p1) > 1 and len(p2) > 1):
        n1 = int(p1[1]); n2 = int(p2[1])
        pt = VT[n1] + (VT[n2] - VT[n1])*t
        VT += [np.array(pt)]
        fq.write('vt %7.3f %7.3f\n'%(*pt[:2],))
        res += '/'+str(len(VT) - 1)

    if (len(p1) > 2 and len(p2) > 2):
        n1 = int(p1[2]); n2 = int(p2[2])
        pt = VN[n1] + (VN[n2] - VN[n1])*t
        pt = pt/np.sqrt(pt.dot(pt))
        VN += [np.array(pt)]
        fq.write('vn %7.3f %7.3f %7.3f\n'%(*pt[:3],))
        res += '/'+str(len(VN) - 1)
    if MAX > 0: AddSign(sign, res)
    return res


########################################################
def Split(fq, r, EPS=1E-3):
    neg = 0; pos = 0
    num = len(r); idx = [-1]
    for i in range(num):
        if i == 0: continue
        p = r[i].split('/')
        n = int(p[0]); idx += [n]
        if Dis[n] < EPS: neg += 1
        if Dis[n] > -EPS: pos += 1
    if neg == num-1: return 0, ''
    if pos == num-1: return 2, ''

    tag = 0; n = idx[num-1]
    if Dis[n] < 0: tag = -1
    elif Dis[n] > 0: tag = 1
    last = r[num - 1]; rlist = []
    for i in range(num):
        if i == 0: continue
        if tag < 0 and Dis[idx[i]] < 0:
            last = r[i]; continue
        if tag > 0 and Dis[idx[i]] > 0:
            rlist += [r[i]]; last = r[i]; continue
        if Dis[idx[i]] == 0:
            rlist += [r[i]]; last = r[i]; tag = 0; continue
        # order: from outside to inside
        if tag < 0: rlist += [Intersect(fq, last, r[i])]
        elif tag > 0: rlist += [Intersect(fq, r[i], last)]

        if Dis[idx[i]] > 0:
            rlist += [r[i]]; tag = 1
        else: tag = -1
        last = r[i]
    if len(rlist) < 3: return 0, ''
    face = 'f ' + ' '.join(rlist) + '\n'
    return 1, face


##########################################################################################
def SplitObj(src, plane=[1], suf=''): # plane=[a,b,c,d]
    global Dis, V, VT, VN
    Dis = [[0]]; VT = [np.zeros(2)]
    V = [np.zeros(3)]; VN = [np.zeros(3)]

    global MAX, SIGN, RES
    with open(src) as f: data = f.readlines()
    facelist = ['']*len(data); #MAX = len(data)
    RES, SIGN = ['']*MAX, ['']*MAX

    if not hasattr(plane,'__len__'): plane = [plane]
    if len(plane)<4: plane = list(plane)+[0]*4
    plane = np.asarray(plane[:4])
    dst = f'{src[:-4]}_{list(plane)}{suf}{src[-4:]}'
    # must append, lest alter the order of previous
    with open(dst, 'w', encoding='utf-8') as f:
        for i,line in enumerate(data):
            res = line.split()
            if len(res) == 0: continue
            #cmd = res[0].lower()
            if res[0] == 'v':
                f.write(line); facelist[i] = 0
                x = np.float64(res[1:4]); V += [x]
                d = x.dot(plane[:3])+plane[3]; Dis += [d]
            elif res[0] == 'vt': # 2D texture
                f.write(line); facelist[i] = 0
                VT += [np.float64(res[1:3])]
            elif res[0] == 'vn':
                f.write(line); facelist[i] = 0
                VN += [np.float64(res[1:4])]
            elif res[0] == 'f': facelist[i] = 1

        for i,line in enumerate(data):
            if facelist[i] != 1: continue
            res = line.split()
            if len(res) >= 4:
                tag, face = Split(f, res)
                if tag == 0: facelist[i] = 0
                elif tag == 1: facelist[i] = face
                else: facelist[i] = ''
            else: facelist[i] = 0

        for i,line in enumerate(data):
            if facelist[i] == '': f.write(line)
            elif facelist[i] != 0: f.write(facelist[i])
    print(f'Split: {dst}')


########################################################
def odm_split(src, plane=[1]):
    tex = '/odm_texturing_25d'
    obj = '/odm_textured_model_geo.obj'
    objT = '/odm_textured_model_geo_T.obj'
    if os.path.isdir(src+tex): src += tex
    if os.path.isfile(src+objT): src += objT
    elif os.path.isfile(src+obj): src += obj
    assert os.path.isfile(src) and src[-4:]=='.obj'
    if type(plane) in (list,tuple): plane = np.array(plane)
    SplitObj(src, plane); SplitObj(src, -plane)


import os, json, shutil
from pyproj import Transformer
##########################################################################################
def calc_axis_lla(lat, lon, alt=0, dt=1E-7): # origin & axis
    # WGS84='EPSG:4326', BeiJing(BJ54_CM_6_20)='EPSG:21460'
    ECEF = '+proj=geocent +ellps=WGS84 +datum=WGS84'
    T = Transformer.from_crs('EPSG:4326', ECEF, always_xy=True)

    lla = np.array([lon,lat,alt]); delta = [dt,0,0]
    o = np.array(T.transform(*lla, radians=False))

    x = np.array(T.transform(*(lla+delta), radians=False))
    x -= np.array(T.transform(*(lla-delta), radians=False))
    x[2] = 0; x /= np.linalg.norm(x); delta = [0,dt,0]

    y = np.array(T.transform(*(lla+delta), radians=False))
    y -= np.array(T.transform(*(lla-delta), radians=False))
    z = np.cross(x,y); z /= np.linalg.norm(z)
    y = np.cross(z,x); return np.array([o,x,y,z]) # (4,3)


# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/geo.py
def ecef_from_lla(lat, lon, alt=0):
    '''ECEF = '+proj=geocent +ellps=WGS84 +datum=WGS84'
    T = Transformer.from_crs('EPSG:4326', ECEF, always_xy=True)
    return T.transform(lon, lat, alt, radians=False) #'''
    WGS84_a = 6378137.0; WGS84_b = 6356752.314245
    a2 = WGS84_a ** 2; b2 = WGS84_b ** 2
    lat = np.radians(lat); lon = np.radians(lon)
    L = 1.0 / np.sqrt(a2 * np.cos(lat) ** 2 + b2 * np.sin(lat) ** 2)
    x = (a2 * L + alt) * np.cos(lat) * np.cos(lon)
    y = (a2 * L + alt) * np.cos(lat) * np.sin(lon)
    z = (b2 * L + alt) * np.sin(lat); return x,y,z


##########################################################################################
# src: (odm_texturing_25d) folder include geo.obj
# new_lla: (folder include)/(path to) new lla.json
# src_lla: (folder include)/(path to) src lla.json
# dst: path of new geo.obj to rename/overwrite/save
def align_obj(src, new_lla, src_lla='', dst='T'):
    lla = '/reference_lla.json'
    if os.path.isdir(new_lla): new_lla += lla
    if os.path.isdir(src_lla): src_lla += lla
    if not src_lla: src_lla = src+lla

    with open(src_lla) as f: s = json.load(f)
    s = calc_axis_lla(*s.values()) # lat,lon,alt
    with open(new_lla) as f: t = json.load(f)
    t = calc_axis_lla(*t.values()) # lat,lon,alt

    obj = '/odm_textured_model_geo.obj'
    with open(src+obj) as f: data = f.readlines()
    for i,v in enumerate(data):
        v = v.split()
        if len(v) and v[0]=='v':
            v = np.float64(v[1:4]).dot(s[1:4])
            v = t[1:4].dot(v+s[0]-t[0]) # x,y,z
            data[i] = 'v %9.6f %9.6f %9.6f\n'%(*v,)

    if os.path.isdir(dst): dst += obj
    elif dst.endswith('.obj'): dst = src+obj
    elif dst: dst = src+obj[:-4]+f'_{dst}'+obj[-4:]
    with open(dst, 'w', encoding='utf-8') as f:
        f.writelines(data); print('Align:', dst)


#Auth = lambda x: os.system(f'sudo chown -R {os.getenv("USER")} {x}')
Auth = lambda x: os.system(f'sudo chown -R {os.environ["USER"]} {x}')
########################################################
# src: odm results folder
def odm_align(src, ref='', suf='T', plane=[1]):
    lla = '/reference_lla.json'
    dst = src+'/odm_texturing_25d'; Auth(dst)
    if not os.path.isfile(dst+lla):
        assert os.path.isfile(src+'/opensfm'+lla)
        shutil.copyfile(src+'/opensfm'+lla, dst+lla)
    if os.path.isfile(ref+'/opensfm'+lla):
        align_obj(dst, ref+'/opensfm', dst=suf)
    elif os.path.isdir(ref) or os.path.isfile(ref):
        align_obj(dst, ref, dst=suf)
    if len(plane): odm_split(src, plane)


##########################################################################################
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('data_part/out-sift')
    #odm_split('odm-RTK-20211108', [1,0,0,0])
    
    #odm_align('odm-RTK-20211108')
    #odm_align('odm-GPS-20211109', 'odm-RTK-20211108')
    #odm_align('odm-GCP-20211108-20211109', 'odm-RTK-20211108')

