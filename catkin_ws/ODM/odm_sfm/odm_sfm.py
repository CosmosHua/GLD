#!/usr/bin/python3
# coding: utf-8

import os, sys, cv2, argparse
import gzip, pickle, json, yaml
import numpy as np, logging, shutil
from itertools import combinations


Auth = lambda x: os.system(f'sudo chown -R {os.environ["USER"]} {x}')
##########################################################################################
def merge_dir(src, dst, pre='', sep='@'):
    if sys.platform=='linux': Auth(src)
    os.makedirs(dst, exist_ok=True)
    for i in os.scandir(src): # USE os.link
        if i.is_symlink(): # NOT os.symlink
            i = os.readlink(i.path)
            if not os.path.isfile(i): continue
            x = f'{dst}/{pre}{os.path.basename(i)}'
            if not os.path.isfile(x): os.link(i, x)
        elif i.is_file() and not i.is_symlink():
            x = f'{dst}/{pre}{i.name}'
            if not os.path.isfile(x): os.link(i.path, x)
        elif i.is_dir(): # recursion
            pre += i.name+sep if sep else ''
            merge_dir(i.path, dst, pre, sep)


def merge_json(src, dst, js, key=None):
    with open(f'{src}/{js}') as f: A = json.load(f)
    with open(f'{dst}/{js}') as f: B = json.load(f)
    for k,v in A.items(): # merge A to B
        if k not in B: B[k] = v
        #elif type(v)==list: B[k] += v
        elif type(v)==list: # type(B[k])==list
            B[k] += [i for i in v if i not in B[k]]
    with open(f'{dst}/{js}','w') as f: json.dump(B, f, indent=4)


########################################################
def load_sort_save(src, a=0, b=None):
    with open(src) as f: d = f.readlines()
    if type(a) in (list,tuple): d = sorted(a)
    elif type(a)==int: d[a:b] = sorted(d[a:b])
    with open(src,'w') as f: f.writelines(d); return d


########################################################
def feat_size(src, cfg=0.5, n=5): # update cfg
    k = 'feature_process_size'; mx = 0
    if type(cfg)==dict and k not in cfg: return
    v = cfg[k] if type(cfg)==dict else cfg
    for i in list(os.scandir(src+'/images'))[:n]:
        mx = max(mx, *cv2.imread(i.path).shape)
    if v<=0 or v>=mx: v = mx # for latest SfM
    elif type(v)==float: v = round(mx*v)
    if type(cfg)==dict: cfg[k] = min(v,mx)
    return min(v,mx) # cfg=dict/int/float


########################################################
def filter_match(pt1, pt2, mod=cv2.RANSAC, thd=1, prob=0.99):
    assert pt1.shape==pt2.shape and len(pt1)>6 and mod in (1,2,4,8)
    M, idx = cv2.findFundamentalMat(pt1, pt2, mod, thd, confidence=prob)
    idx = np.where(idx>0)[0]; return M, idx # inliers


##########################################################################################
# Ref: https://github.com/OpenDroneMap/ODM/blob/master/opendm/osfm.py
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/config.py
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/features_processing.py
# osfm.update_config(), config.load_config(), config.default_config()
def SfM_config(src, args):
    file = src+'/config.yaml'
    from opensfm.config import default_config
    if os.path.isfile(file):
        with open(file) as f: cfg = yaml.safe_load(f)
    else: cfg = default_config() # cfg=dict

    if type(args)==str and os.path.isfile(args):
        with open(args) as f: args = yaml.safe_load(f)
    if type(args)==dict: cfg.update(args)
    if cfg['feature_type']=='ORB': cfg['matcher_type']='BRUTEFORCE'
    feat_size(src, cfg) # cfg['feature_process_size']
    with open(file, 'w') as f: # update config.yaml
        f.write(yaml.dump(cfg, default_flow_style=False))


def CMD(cmd): INFO(cmd); os.system(cmd) # return os.popen(cmd).readlines()
SfM_CMD = ['extract_metadata', 'detect_features', 'match_features', 'create_tracks',
    'reconstruct', 'bundle', 'mesh', 'undistort', 'compute_depthmaps', 'compute_statistics',
    'export_ply', 'export_openmvs', 'export_visualsfm', 'export_pmvs', 'export_bundler',
    'export_colmap', 'export_geocoords', 'export_report', 'extend_reconstruction',
    'create_submodels', 'align_submodels']
##########################################################################################
# Ref: https://github.com/mapillary/OpenSfM/blob/main/bin/opensfm_run_all
# Ref: https://github.com/mapillary/OpenSfM/blob/main/bin/opensfm_main.py
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/commands/__init__.py
def SfM_cmd(src, cmd): # cmd=int,str,range,list,tuple
    for c in [cmd] if type(cmd) in (int,str) else cmd:
        c = SfM_CMD[c] if type(c)==int else c # ->str
        assert type(c)==str and c.split()[0] in SfM_CMD
        c = f'{SfM_DIR}/bin/opensfm {c} {src}'; CMD(c)
        #p = subprocess.Popen(c, shell=True)


##########################################################################################
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/exif.py
# Ref: https://github.com/OpenDroneMap/ODM/blob/master/opendm/photo.py
# photo.get_gps_dop(), photo.parse_exif_values(), photo.get_xmp(), exif.get_xmp()
def SfM_exif_dop(src, L=0, dop=10): # add DOP to EXIF
    if not os.path.isdir(src+'/exif'):
        SfM_cmd(src, 'extract_metadata')
    for i in os.scandir(src+'/images'):
        e = f'{src}/exif/{i.name}.exif'
        with open(e,'r') as f: x = json.load(f)
        d = SfM_xmp_dop(i.path, L); gps = x['gps']
        gps['dop'] = d if d else gps.get('dop',dop)
        with open(e,'w') as f: json.dump(x, f, indent=4)


def SfM_xmp_dop(im, L=0):
    from opensfm.exif import get_xmp
    #from exifread import process_file
    with open(im, 'rb') as f:
        xmp = get_xmp(f)[0] # get xmp info
        #xmp = process_file(f, details=False)
    x = float(xmp.get('@drone-dji:RtkStdLat', -1))
    y = float(xmp.get('@drone-dji:RtkStdLon', -1))
    z = float(xmp.get('@drone-dji:RtkStdHgt', -1))
    #gps_xy_stddev = max(x,y); gps_z_stddev = z
    if max(x,y,z)<0: return None # use default
    dp = np.array([i for i in (x,y,z) if i>0])
    return np.mean(dp**L)**(1/L) if L else max(dp)


load_im = lambda s,i: cv2.imread(f'{s}/images/{i}')
load_ft = lambda s,i: np.load(f'{s}/features/{i}.features.npz')
##########################################################################################
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/features.py
# features.normalize_features(), features.denormalized_image_coordinates()
# features.load_features(), features.save_features()
def SfM_feat_denorm(pt, hw): # pt=[x,y,sz,angle]
    if type(hw)==str: hw = cv2.imread(hw).shape[:2]
    elif type(hw)==np.ndarray: hw = hw.shape[:2]
    assert type(hw) in (list,tuple) and len(hw)>1
    h,w = hw[:2]; sz = max(hw); p = np.asarray(pt).T
    p[0] = p[0] * sz - 0.5 + w / 2.0 # x
    p[1] = p[1] * sz - 0.5 + h / 2.0 # y
    if p.shape[0]>2: p[2:3] *= sz # size
    return np.int32(np.round(p[:3].T))


# ft.files; ft['points']=[x,y,size,angle]
def SfM_feat_uv(im, src=0, pt=0, idx=''):
    if type(src)==type(im)==str:
        if type(pt)!=np.ndarray: # first
            pt = load_ft(src,im)['points']
        im = load_im(src, im) # then
    assert type(im)==np.ndarray, type(im)
    assert type(pt)==np.ndarray, type(pt)
    if 'float' in str(pt.dtype):
        pt = SfM_feat_denorm(pt[:,:2], im)
    pt = pt[idx] if type(idx)!=str else pt
    return im, pt[:,:2] # norm->pixel


##########################################################################################
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/dataset.py
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/matching.py
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/actions/match_features.py
# match_features.run_dataset(), matching.match_images(), matching.save_matches()
def SfM_match(src, pre, mix=0): # match_features
    from opensfm.actions.match_features import timer, matching, write_report
    from opensfm.dataset import DataSet; data = DataSet(src); t = timer()
    INFO(f'{SfM_DIR}/bin/opensfm match_features: {src}'); GPS,RTK = [],[]
    if os.path.isdir(pre):
        merge_dir(pre+'/exif', src+'/exif')
        merge_dir(pre+'/features', src+'/features')
        merge_json(pre, src, 'camera_models.json')
        #merge_json(pre, src, 'reports/features.json')
        #merge_dir(pre+'/reports/features', src+'/reports/features')
        GPS, RTK = data.images(), DataSet(pre).images()
    else: # split data->(GPS,RTK)
        for i in data.images(): (RTK if i.startswith(pre) else GPS).append(i)
    if mix in (1,3): GPS += RTK # 1: match (GPS+RTK, RTK)
    if mix in (2,3): RTK += GPS # 2: match (GPS, RTK+GPS)
    pairs, preport = matching.match_images(data, {}, GPS, RTK)
    matching.save_matches(data, GPS, pairs)
    write_report(data, preport, list(pairs.keys()), timer()-t)


##########################################################################################
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/tracking.py
# tracking.create_tracks_manager(), load_features(), load_matches()
def SfM_parse_track(src):
    from opensfm.dataset import DataSet
    #from opensfm.pymap import Observation
    TM = DataSet(src).load_tracks_manager(); T = {}
    for im in TM.get_shot_ids():
        T.setdefault(im, [[],[],[],[]]) # tid=str
        for tid,v in TM.get_shot_observations(im).items():
            T[im][0] += [tid]; T[im][1] += [v.id] # xys
            T[im][2] += [np.hstack([v.point, v.scale])]
            T[im][3] += [v.color] # RGB
    return T # {im: [tid,fid,xys,RGB]}


#from pyproj import Proj, transform
LLA = '+proj=lonlat +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
ECEF = '+proj=geocent +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
##########################################################################################
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/geo.py
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/actions/export_geocoords.py
# export_geocoords._transform_image_positions(), geo.to_topocentric(), geo.to_lla()
# transform(Proj(LLA), Proj(ECEF), *v, radians=False); Proj(LLA)(x,y,inverse=True)
def SfM_export_pos(src, dop=0.1, tsv='image_geocoords.tsv'):
    rename_rec(src); dif = []; gjs = {} # for sfm
    SfM_cmd(src, f'export_geocoords --image-positions --proj="{LLA}"')
    rename_rec(src); geo = load_sort_save(f'{src}/{tsv}', a=1) # sort
    with open(src+'/geo.txt','w') as f: f.writelines([LLA+'\n']+geo[1:])

    from opensfm.dataset import DataSet
    data = DataSet(src); ref = data.load_reference()
    for v in geo[1:]: # skip 1st-row
        im, *v = v.split(); v = np.float64(v)[[1,0,2]]
        o = [*data.load_exif(im)['gps'].values()][:3] # lat,lon,alt
        ov = ref.to_topocentric(*v)-np.array(ref.to_topocentric(*o))
        gjs[im] = {'gps': dict(latitude=v[0], longitude=v[1], altitude=v[2],
        dop=dop)}; dif += [f'{im} lla={v} exif={o}\tdif={ov.tolist()}\n']
    with open(f'{src}/{tsv[:-4]}.dif.txt','w') as f: f.writelines(dif)
    with open(src+'/geo.json','w') as f: json.dump(gjs, f, indent=4)


# Ref: https://github.com/OpenDroneMap/ODM/blob/master/stages/run_opensfm.py
def rename_rec(src): # for odm: rename|recover topocentric.json
    if sys.platform=='linux': Auth(src)
    #rc = src+'/reconstruction.json'; rt = rc[:-4]+'topocentric.json'
    rt = src+'/reconstruction.topocentric.json'; rc = rt[:-16]+'json'
    if os.path.isfile(rc+'='): os.rename(rc, rt); os.rename(rc+'=', rc)
    elif os.path.isfile(rt): os.rename(rc, rc+'='); os.rename(rt, rc)


########################################################
def svd_fit(A, B):
    a = np.mean(A, axis=0); A = A-a
    b = np.mean(B, axis=0); B = B-a
    # calculate rotation matrix
    W = B.T.dot(A); U,S,V = np.linalg.svd(W)
    R = U.dot(V); TF = np.identity(4)
    # special reflection case
    if np.linalg.det(R)<0: V[2,:] *= -1; R = U.dot(V)
    TF[:3,3] = b.T-R.dot(a.T); TF[:3,:3] = R; return TF


########################################################
# Ref: features.normalized_image_coordinates()
def SfM_gcp2xyz(GPS, RTK):
    from opensfm.dataset import DataSet
    RRF = DataSet(RTK).load_reference()
    a = 3 if os.path.isdir(GPS+'/matches') else 2
    if not os.path.isfile(GPS+'/reconstruction.json'):
        SfM_cmd(GPS, range(a,5))
    GTM = DataSet(GPS).load_tracks_manager(); res = []
    GRC = GPS+'/reconstruction.topocentric.json'
    if not os.path.isfile(GRC): GRC = GRC[:-16]+'json'
    with open(GRC) as f: GRC = json.load(f)[0]['points']
    GIM = GPS if os.path.isdir(GPS+'/images') else GPS+'/..'
    with open(GPS+'/gcp_list.txt') as f: gcp = f.readlines()
    #dtp = [(k,float) for k in ('lon','lat','alt','x','y')]
    for v in gcp[1:]: # skip 1st-row
        *v,im = v.split(); v = np.float64(v); x = []
        h,w,c = cv2.imread(f'{GIM}/images/{im}').shape
        denm = lambda x: (x*max(w,h)*2+(w,h)-1)/2
        for tid,ob in GTM.get_shot_observations(im).items():
            d = np.linalg.norm(denm(ob.point)-v[3:5])
            if d<1: x.append([d,tid]) # pixel
        d, tid = min(x) if len(x) else [np.inf,'']
        if tid not in GRC: INFO(f'skip {tid}: {[*v,im]}')
        else: res += [(*GRC[tid]['coordinates'], *v[[1,0,2]])]
    v = np.array(res).T; v[3:] = RRF.to_topocentric(*v[3:]); return v.T


########################################################
def show_gcp_svd(GPS, RTK):
    gr = SfM_gcp2xyz(GPS, RTK) # gr=(gps,rtk)
    TF = svd_fit(gr[:,:3], gr[:,3:])
    INFO(f'SVD Rigid Transformation:\n{TF}')


IDX = lambda x,y: np.where(x==np.asarray(y)[:,None])[-1]
GCF = lambda x: (3*'%.15f '+2*'%5s '+'%s\n')%(*x,) # lla,uv,im
##########################################################################################
# Ref: https://opensfm.org/docs/using.html#ground-control-points
# gz->fid->track2->tid2->reconstruction2->tid2->xyz2->lla2->gcp
def SfM_gcp_gz(GPS, RTK='', thd=1, dst=0):
    from opensfm.dataset import DataSet
    if not os.path.isdir(RTK): RTK = GPS
    R = RTK+'/reconstruction.topocentric.json'
    if not os.path.isfile(R): R = R[:-16]+'json'
    with open(R) as f: R = json.load(f)[0]['points']
    T = SfM_parse_track(RTK) # {im2:[tid2,fid2,xys2,rgb2]}
    ref = DataSet(RTK).load_reference(); gcp = [LLA+'\n']

    for gz in os.scandir(GPS+'/matches_gcp'):
        im1 = gz.name[:-15] # parse *_matches.pkl.gz
        with gzip.open(gz.path,'rb') as f: gz = pickle.load(f)
        for im2, fid in gz.items(): # (im1,im2):[fid1,fid2]
            if len(fid)<7: INFO(f'skip: {im1} {im2}'); continue # filter
            _, uv1 = SfM_feat_uv(im1, src=GPS, idx=fid[:,0]) # norm->pixel
            _, uv2 = SfM_feat_uv(im2, src=RTK, idx=fid[:,1]) # norm->pixel
            _, idx = filter_match(uv1, uv2, thd=0.5); fid = fid[idx]

            idx = IDX(T[im2][1], fid[:,1]) # filter: track+fid2->fid2
            tid2, fid2, xys2, rgb2 = [np.array(k)[idx] for k in T[im2]]
            idx = IDX(tid2, list(R)) # filter: reconstruct+tid2->tid2
            INFO(f'gz_gcp: {im1} {im2} {len(uv1)}->{len(fid)}->{len(idx)}')
            if len(idx)<1: continue # skip ref.to_lla() when idx=[]
            tid2, fid2, xys2, rgb2 = tid2[idx], fid2[idx], xys2[idx], rgb2[idx]
            xyz2 = np.array([R[k]['coordinates'] for k in tid2]) # local xyz
            lla2 = np.array(ref.to_lla(*xyz2.T)).T # xyz->lat,lon,alt

            idx = IDX(fid[:,1], fid2); fid = fid[idx]
            _, uv1 = SfM_feat_uv(im1, src=GPS, idx=fid[:,0]) # fid1
            _, uv2 = SfM_feat_uv(im2, src=RTK, pt=xys2) # norm->pixel
            for pt,uv in zip(lla2,uv1): gcp += [GCF([*pt[[1,0,2]],*uv,im1])]
    with open(GPS+'/gcp_list.txt','w') as f: f.writelines(gcp)
    gcp = dedup_gcp(GPS); gcp = filter_gcp(GPS, RTK, thd=thd)
    INFO(f'Created {len(gcp)-1} GCPs: {GPS}/gcp_list.txt\n')
    cv2.destroyAllWindows(); return gcp # list


def dedup_gcp(GPS, EPS=0.01):
    from opensfm.geo import ecef_from_lla; res = {}
    with open(GPS+'/gcp_list.txt') as f: gcp = f.readlines()
    for i in gcp[1:]: # skip 1st-row
        *lla,u,v,im = i.split(); k = (u,v,im)
        lla = np.float64(lla)[[1,0,2]] # lat,lon,alt
        res.setdefault(k,[]); res[k].append(lla)
    new = [gcp[0]] # retain 1st-row
    for k in sorted(res, key=lambda x:x[-1]):
        m = np.mean(res[k], axis=0) # first mean()
        d = np.float64([ecef_from_lla(*i) for i in res[k]])
        d = np.linalg.norm(d-ecef_from_lla(*m), axis=1)
        #d = res[k]-m; d[:,:2] *= 1E5 # degree->meter
        if np.all(d<EPS): new += [GCF([*m[[1,0,2]],*k])]
    with open(GPS+'/gcp_list.txt','w') as f: f.writelines(new)
    INFO(f'Dedup_GCPs: {len(gcp)} -> {len(new)}'); return new


########################################################
def filter_gcp(GPS, RTK, thd=1): # reproject
    from odm_filter import Camera
    from opensfm.dataset import DataSet
    K = Camera(GPS+'/camera_models.json').K()
    ref = DataSet(RTK).load_reference(); res = {}
    PM = 8 if hasattr(cv2,'SOLVEPNP_SQPNP') else 1
    # cv2.SOLVEPNP_ITERATIVE=0: need n>=6 non-planar
    # cv2.SOLVEPNP_EPNP=1, cv2.SOLVEPNP_SQPNP=8: n>=4
    out, err = GPS+'/gcp_list.txt', GPS+'/gcp_err.txt'
    if os.path.isfile(out):
        with open(out) as f: gcp = f.readlines()
    elif os.path.isfile(err):
        with open(err) as f: gcp = f.readlines()
        for i,v in enumerate(gcp):
            v = v.split(); x = np.float64(v[1:4])
            gcp[i] = GCF([*x,*v[4:6],v[0]])
        gcp.insert(0, LLA+'\n')
    for v in gcp[1:]: # skip 1st-row
        *v, im = v.split(); v = np.float64(v+2*[np.inf])
        res.setdefault(im,[]); res[im].append(v)

    for im,v in res.items():
        P = 0 if len(v)>5 else PM
        if len(v)<(4 if P>0 else 6): continue
        v = res[im] = np.float64(v); uv = v[:,3:5] # lon,lat,alt
        pt = np.array(ref.to_topocentric(*v[:,:3].T[[1,0,2]])).T
        # for coplanar points; cv2.Rodrigues(Rv): RotVector->RotMatrix
        try: _, Rv, Tv, _ = cv2.solvePnPRansac(pt, uv, K, None, flags=P)
        except: _, Rv, Tv, _ = cv2.solvePnPRansac(pt, uv, K, None, flags=PM)
        # cv2.projectPoints: np.array/np.ascontiguousarray->mem-block
        xy, Jacob = cv2.projectPoints(pt, Rv, Tv, K, None)
        dis = v[:,5] = np.linalg.norm(xy.squeeze()-uv, axis=1)

        his = np.histogram(dis, bins=[*range(11),np.inf])[0]
        for c in range(len(his)-1,-1,-1): # len(v)=sum(his)
            if sum(his[c:])>=len(v)*0.2: break
        idx = np.where(dis<=c)[0]; P = 0 if len(idx)>5 else PM
        if len(idx)<(4 if P>0 else 6): continue # for cv2.solvePnP
        try: _, Rv, Tv = cv2.solvePnP(pt[idx], uv[idx], K, None, flags=P)
        except: _, Rv, Tv = cv2.solvePnP(pt[idx], uv[idx], K, None, flags=PM)
        xy, Jacob = cv2.projectPoints(pt, Rv, Tv, K, None)
        v[:,6] = np.linalg.norm(xy.squeeze()-uv, axis=1) # dis2
    with open(GPS+'/gcp_err.txt','w') as f: # save
        F = lambda x: ('%s'+3*' %.15f'+2*' %5d'+2*' %9.3f'+'\n')%x
        for k,v in res.items(): f.writelines([F((k,*e)) for e in v])

    F = lambda x: np.where(x.max(axis=1)<np.inf, x.mean(axis=1),
        x.min(axis=1)) # np.mean(v) if max(v)<np.inf else min(v)
    F = lambda x: np.where(x[:,1]<np.inf, x[:,1], x[:,0])
    dis = F(np.vstack([*res.values()])[:,5:])<thd
    new = [gcp[0]]+[gcp[i] for i in np.where(dis)[0]+1]
    with open(GPS+'/gcp_list.txt','w') as f: f.writelines(new)
    INFO(f'Filter_GCPs: {len(gcp)} -> {len(new)}'); return new


########################################################
def SfM_GCP(GPS, RTK, thd=3, mix=0):
    if sys.platform=='linux': Auth(GPS)
    out, err = GPS+'/gcp_list.txt', GPS+'/gcp_err.txt'
    if os.path.isfile(out) and Args.bak: # backup
        new = out[:-4]+'_%d.txt'%os.stat(out).st_mtime
        os.rename(out, new); INFO(f'Backup: {new}')
    if os.path.isfile(err): #os.remove(err)
        return filter_gcp(GPS, RTK, thd=thd)
    if not os.path.isdir(GPS+'/features'): # extract
        SfM_exif_dop(GPS); SfM_cmd(GPS,'detect_features')
    if not os.path.isdir(GPS+'/matches_gcp'):
        if os.path.isdir(GPS+'/matches'): # backup
            os.rename(GPS+'/matches', GPS+'/matches=')
        SfM_match(GPS, RTK, mix) # match (GPS,RTK)
        os.rename(GPS+'/matches', GPS+'/matches_gcp')
        if os.path.isdir(GPS+'/matches='): # recover
            os.rename(GPS+'/matches=', GPS+'/matches')
    return SfM_gcp_gz(GPS, RTK, thd=thd, dst=0)


PS = lambda x,k=1: int(np.clip(x, 1, os.cpu_count()//k))
GF = lambda x: f'grep {x}' if sys.platform=='linux' else f'find /i "{x}"'
SfM_CFG = dict(use_exif_size=True, use_altitude_tag=True, feature_type='SIFT',
    sift_peak_threshold=0.066, feature_min_frames=10000, feature_process_size=0.5,
    matcher_type='FLANN', flann_algorithm='KDTREE', triangulation_type='ROBUST',
    matching_gps_neighbors=8, matching_gps_distance=0, matching_graph_rounds=50,
    align_orientation_prior='vertical', bundle_use_gcp=False, bundle_use_gps=True,
    align_method='auto', retriangulation_ratio=2, processes=PS(8,6), # concurrency
    bundle_outlier_filtering_type='AUTO', optimize_camera_parameters=True) # odm'''
##########################################################################################
# Ref: https://github.com/OpenDroneMap/ODM/blob/master/opendm/osfm.py
# Ref: https://github.com/OpenDroneMap/ODM/blob/master/opendm/config.py
def ODM_cmd(src, cfg, ref=0, sfm=SfM_CFG):
    for k,v in sfm.items(): # config from sfm
        if k=='processes': cfg['max-concurrency'] = v
        elif k=='matcher_type': cfg['matcher-type'] = v.lower()
        elif k=='feature_type': cfg['feature-type'] = v.lower()
        elif k=='feature_min_frames': cfg['min-num-features'] = v
        elif k=='feature_process_size': cfg['resize-to'] = feat_size(src, v)
    for k,v in cfg.items(): # config for odm
        if k=='resize-to': cfg[k] = feat_size(src, v)
        if k.split('-')[-1] in ('type','quality'): cfg[k] = v.lower()
        if cfg[k]=='orb': cfg['matcher-type']='bruteforce'
    cfg = ' '.join(['--'+(f'{k}={v}' if v!='' else k) for k,v in cfg.items()])
    src = os.path.abspath(src); GPU = Args.gpu if Args.mod=='odm' else ''
    if EXE: # use windows installer
        assert VER>='2.8.5'; gpu = '--no-gpu' if GPU=='' else ''
        CMD(f'{ODM_DIR}/run.bat {src} {cfg} {gpu} --time')
    else: # use linux/windows docker
        proj = f'--project-path=/root {os.path.basename(src)}'
        img = 'opendronemap/odm'; root = f'{os.path.dirname(src)}:/root'
        if sys.platform=='win32': # conform docker with installer
            if VER not in os.popen('docker image ls|'+GF(img)).read():
                os.system(f'docker image pull {img}:{VER}')
            os.system(f'docker tag {img}:{VER} {img}:latest')
        img = f'--gpus={GPU} {img}:gpu' if GPU!='' else img
        CMD(f'docker run -ti --rm -v={root} {img} {proj} {cfg} --time')


##########################################################################################
def ODM_img_lla2(GPS, RTK, dst, mesh=0, dop=0.1):
    from odm_filter import filter_reconstruct
    gps, rtk = os.path.basename(GPS), os.path.basename(RTK)
    tmp = f'{dst}/odm-RTK-{rtk}'; cfg = {'end-with':'opensfm'}
    merge_dir(RTK, tmp+'/images'); merge_dir(RTK, tmp+'/opensfm/images')
    RTK = tmp; ODM_cmd(RTK, cfg); filter_reconstruct(RTK, 0.3)

    tmp = f'{dst}/sfm-GPS-{rtk}-{gps}'; merge_dir(GPS, tmp+'/images')
    GPS = tmp; SfM_config(GPS, SfM_CFG); SfM_GCP(GPS, RTK+'/opensfm')
    if Args.svd: return show_gcp_svd(GPS, RTK+'/opensfm')

    cfg['gcp'] = (os.path.abspath(GPS) if EXE else '/root/'
        +os.path.basename(GPS))+'/gcp_list.txt'
    GCP = f'{dst}/odm-GCP-{rtk}-{gps}'; merge_dir(GPS+'/images', GCP+'/images')
    ODM_cmd(GCP, cfg, RTK); SfM_export_pos(GCP+'/opensfm'); INFO('ALL DONE!')


def ODM_img_lla3(GPS, RTK, dst, mesh=0, dop=0.1):
    from odm_filter import filter_reconstruct
    gps, rtk = os.path.basename(GPS), os.path.basename(RTK)
    tmp = f'{dst}/odm-RTK-{rtk}'; cfg = {'end-with':'opensfm'}
    merge_dir(RTK, tmp+'/images'); merge_dir(RTK, tmp+'/opensfm/images')
    RTK = tmp; ODM_cmd(RTK, cfg); filter_reconstruct(RTK, 0.3)

    tmp = f'{dst}/odm-GPS-{rtk}-{gps}'; merge_dir(GPS, tmp+'/images')
    merge_dir(GPS, tmp+'/opensfm/images'); GPS = tmp
    ODM_cmd(GPS, cfg, RTK); SfM_GCP(GPS+'/opensfm', RTK+'/opensfm')
    if Args.svd: return show_gcp_svd(GPS+'/opensfm', RTK+'/opensfm')

    cfg['gcp'] = (os.path.abspath(GPS) if EXE else '/root/'
        +os.path.basename(GPS))+'/opensfm/gcp_list.txt'
    GCP = f'{dst}/odm-GCP-{rtk}-{gps}'; merge_dir(GPS+'/images', GCP+'/images')
    merge_dir(GPS+'/opensfm/features', GCP+'/opensfm/features')
    merge_dir(GPS+'/opensfm/matches', GCP+'/opensfm/matches')
    ODM_cmd(GCP, cfg, RTK); SfM_export_pos(GCP+'/opensfm'); INFO('ALL DONE!')


SP = lambda x: os.path.relpath(x, os.getcwd()) # abspath()
##########################################################################################
def Initial(odm='.'):
    import cpuinfo; global SfM_DIR, ODM_DIR, INFO, EXE
    INFO = logging.getLogger('Joshua').info; logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    CPU = f'{os.cpu_count()} * '+cpuinfo.get_cpu_info()['brand_raw']
    if sys.platform=='linux':
        #CPU = os.popen('lscpu|grep name|cut -f2 -d:').read()[9:-1]
        bin = os.popen('sudo updatedb; locate bin/opensfm').readline()
        SfM_DIR = bin[:bin.find('/bin')]; #sys.path.append(SfM_DIR)
    elif sys.platform=='win32':
        while not os.path.isfile(os.path.join(odm,'run.bat')):
             odm = input('Invalid ODM folder! Please reset: ')
        ODM_DIR = os.path.abspath(odm); global VER
        with open(ODM_DIR+'/VERSION') as f: VER = f.read().strip()
        SfM_DIR = ODM_DIR+'/SuperBuild/install/bin/opensfm'
    else: assert False, 'Unsupported Operating System!'
    EXE = sys.platform=='win32' and not Args.docker; INFO(CPU)
    sys.path.append(SfM_DIR); INFO(' '.join(sys.argv)); INFO(Args)


def parse_args(k=5):
    parser = argparse.ArgumentParser(description='ODM-SFM')
    parser.add_argument('--log', default=None, type=str, help='log file name')
    parser.add_argument('--dst', default=None, type=str, help='results folder')
    parser.add_argument('--gps', default=None, type=str, help='GPS images folder')
    parser.add_argument('--rtk', default=None, type=str, help='RTK images folder')
    parser.add_argument('--min', default=4000, type=int, help='min feature number')
    parser.add_argument('--mod', default='odm', type=str, help='method of matching')
    parser.add_argument('--dop', default=0.1, type=float, help='gps-accuracy for POS')
    parser.add_argument('--cpu', default=os.cpu_count(), type=int, help='concurrency')
    parser.add_argument('--gpu', default='', type=str, help='""|all|GPU_IDs(0,1,2)')
    parser.add_argument('--svd', default=False, action='store_true', help='skip pos')
    parser.add_argument('--bak', default=True, type=bool, help='backup|over gcp_list')
    parser.add_argument('--docker', default=False, action='store_true', help='docker')
    parser.add_argument('--root', default='D:/ODM', type=str, help='ODM_DIR for Win')
    parser.add_argument('--mesh', default=0, type=int, help='odm_texturing_25d')
    parser.add_argument('--type', default='sift', type=str, help='feature type')
    parser.add_argument('--size', default=0.5, type=float, help='feature quality')
    global Args; Args = parser.parse_args(); os.makedirs(Args.dst, exist_ok=True)
    Args.gps, Args.rtk, Args.dst = [SP(i) for i in (Args.gps, Args.rtk, Args.dst)]
    if Args.log: sys.stderr = sys.stdout = open(f'{Args.dst}/{Args.log}','a+')
    SfM_CFG['feature_process_size'] = Args.size; Args.mod = Args.mod.lower()
    SfM_CFG['processes'] = Args.cpu = PS(Args.cpu, k); Initial(Args.root)
    SfM_CFG['feature_type'] = Args.type = Args.type.upper()
    SfM_CFG['feature_min_frames'] = Args.min; return Args


# python3 odm_sfm.py --gps=GPS --rtk=RTK --dst=xxx > xxx.log 2>&1
##########################################################################################
if __name__ == '__main__':
    x = parse_args()
    if x.mod=='sfm': ODM_img_lla2(x.gps, x.rtk, x.dst, x.mesh, x.dop)
    elif x.mod=='odm': ODM_img_lla3(x.gps, x.rtk, x.dst, x.mesh, x.dop)

