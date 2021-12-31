#!/usr/bin/python3
# coding: utf-8

import numpy as np, logging
import os, sys, cv2, argparse
import gzip, pickle, csv, yaml, json
from itertools import combinations


Auth = lambda x: os.system(f'sudo chown -R {os.environ["USER"]} {x}')
##########################################################################################
def filter_match(pt1, pt2, mod=cv2.RANSAC, thd=1, prob=0.99):
    assert pt1.shape==pt2.shape and len(pt1)>6 and mod in (1,2,4,8)
    M, idx = cv2.findFundamentalMat(pt1, pt2, mod, thd, confidence=prob)
    idx = np.where(idx>0)[0]; return M, idx # inliers


########################################################
def merge_dir(src, dst, pre='', sep='%'):
    Auth(src); os.makedirs(dst, exist_ok=True)
    for i in os.scandir(src): # USE os.link
        if i.is_symlink(): # NOT os.symlink
            i = os.readlink(i.path)
            if not os.path.isfile(i): continue
            x = f'{dst}/{pre}{os.path.basename(i)}'
            if not os.path.isfile(x): os.link(i, x)
        elif i.is_file() and not i.is_symlink():
            x = f'{dst}/{pre}{i.name}'
            if not os.path.isfile(x): os.link(i.path, x)
        else: merge_dir(i.path, dst, pre+i.name+sep)


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


x = os.popen('locate bin/opensfm').readline().strip()
SfM_DIR = x[:x.find('/bin')]; sys.path.append(SfM_DIR)
##########################################################################################
# Ref: https://github.com/OpenDroneMap/ODM/blob/master/opendm/osfm.py
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/config.py
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/features_processing.py
# osfm.update_config(), config.load_config(), config.default_config()
def SfM_config(src, args):
    file = f'{src}/config.yaml'
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


SfM_CDM = ['extract_metadata', 'detect_features', 'match_features', 'create_tracks',
    'reconstruct', 'bundle', 'mesh', 'undistort', 'compute_depthmaps', 'compute_statistics',
    'export_ply', 'export_openmvs', 'export_visualsfm', 'export_pmvs', 'export_bundler',
    'export_colmap', 'export_geocoords', 'export_report', 'extend_reconstruction',
    'create_submodels', 'align_submodels']; INFO = logging.getLogger('Hua').info
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
##########################################################################################
# Ref: https://github.com/mapillary/OpenSfM/blob/main/bin/opensfm_run_all
# Ref: https://github.com/mapillary/OpenSfM/blob/main/bin/opensfm_main.py
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/commands/__init__.py
def SfM_cmd(src, cmd): # cmd=int,str,range,list,tuple
    for c in [cmd] if type(cmd) in (int,str) else cmd:
        c = SfM_CDM[c] if type(c)==int else c
        assert type(c)==str and c.split()[0] in SfM_CDM
        c = f'{SfM_DIR}/bin/opensfm {c} {src}'; INFO(c)
        os.system(c) #p = os.popen(c).readline()
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
        if 'dop' in x['gps'] and x['gps']['dop']<dop: continue
        x['gps']['dop'] = SfM_xmp_dop(i.path, L, dop)
        with open(e,'w') as f: json.dump(x, f, indent=4)


def SfM_xmp_dop(im, L=0, dop=10):
    from opensfm.exif import get_xmp
    #from exifread import process_file
    with open(im, 'rb') as f:
        xmp = get_xmp(f)[0] # get xmp info
        #xmp = process_file(f, details=False)
    x = float(xmp.get('@drone-dji:RtkStdLat', -1))
    y = float(xmp.get('@drone-dji:RtkStdLon', -1))
    z = float(xmp.get('@drone-dji:RtkStdHgt', -1))
    #gps_xy_stddev = max(x,y); gps_z_stddev = z
    if max(x,y,z)<0: return dop # use default
    dp = np.array([i for i in (x,y,z) if i>0])
    return np.mean(dp**L)**(1/L) if L else max(dp)


load_im = lambda dir,i: cv2.imread(f'{dir}/images/{i}')
load_ft = lambda dir,i: np.load(f'{dir}/features/{i}.features.npz')
##########################################################################################
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/features.py
# features.normalize_features(), features.denormalized_image_coordinates()
# features.load_features(), features.save_features()
def SfM_feat_denorm(pt, hw): # pt=[x,y,size,angle]
    if type(hw)==str: hw = cv2.imread(hw).shape[:2]
    elif type(hw)==np.ndarray: hw = hw.shape[:2]
    assert type(hw) in (list,tuple) and len(hw)>1
    h,w = hw[:2]; p = pt.copy(); size = max(w,h)
    p[:,0] = p[:,0] * size - 0.5 + w / 2.0
    p[:,1] = p[:,1] * size - 0.5 + h / 2.0
    if p.shape[1]>2: p[:,2:3] *= size
    return np.int32(np.round(p[:,:3]))


# ft.files; ft['points']=[x,y,size,angle]
def SfM_feat_uv(im, src=0, pt=0, idx=0):
    if type(src)==type(im)==str:
        if type(pt)!=np.ndarray: # first
            pt = load_ft(src,im)['points']
        im = load_im(src, im) # then
    assert type(im)==np.ndarray, type(im)
    assert type(pt)==np.ndarray, type(pt)
    if 'float' in str(pt.dtype):
        pt = SfM_feat_denorm(pt[:,:2], im)
    pt = pt[idx] if type(idx)!=int else pt
    return im, pt[:,:2] # norm->pixel


##########################################################################################
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/dataset.py
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/matching.py
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/actions/match_features.py
# match_features.run_dataset(), matching.match_images(), matching.save_matches()
def SfM_match(src, dst, mix=0): # SfM_cmd(src, 'match_features')
    from opensfm.dataset import DataSet; INFO(f'match_features: {src}')
    from opensfm.actions.match_features import timer, matching, write_report
    assert type(dst)==str; data = DataSet(src); t = timer()
    if not os.path.isdir(dst): # split src to (GPS,RTK)
        GPS, RTK = [],[]; key = lambda x: x.startswith(dst)
        for i in data.images(): (RTK if key(i) else GPS).append(i)
    else: GPS, RTK = data.images(), DataSet(dst).images()
    if mix in (1,3): GPS += RTK # 1: match (GPS+RTK, RTK)
    if mix in (2,3): RTK += GPS # 2: match (GPS, RTK+GPS)
    pairs, preport = matching.match_images(data, {}, GPS, RTK)
    matching.save_matches(data, GPS, pairs)
    write_report(data, preport, list(pairs.keys()), timer()-t)


##########################################################################################
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/dataset.py
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/matching.py
# dataset.load_matches(), matching.match_images(), matching.match()
def SfM_parse_gz(src, sub='matches'):
    M = {} # *_matches.pkl.gz
    for i in os.scandir(f'{src}/{sub}'):
        with gzip.open(i.path, 'rb') as f:
            mt = pickle.load(f); im1 = i.name[:-15]
        for im2, fid in mt.items(): # dict
            if len(fid): M[im1, im2] = fid
    #for k,v in M.items(): print(k,len(v))
    return M # {(im1,im2): [fid1,fid2]}


IDX = lambda x,y: np.where(x==np.asarray(y)[:,None])[-1]
##########################################################################################
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/tracking.py
# tracking.create_tracks_manager(), load_features(), load_matches()
# csv: [img_name, track_id, feature_id, x, y, size, R, G, B, ...]
def SfM_parse_csv(src):
    T = {} # {im: {tid,fid,xys,rgb}}
    with open(f'{src}/tracks.csv') as f:
        track = csv.reader(f, delimiter='\t')
        for i in track: # i=list of items
            if len(i)<9: continue # skip 1st-row
            im, tid, fid = i[0], int(i[1]), int(i[2])
            xys = np.float64(i[3:6]).tolist() # [x,y,size]
            rgb = np.int32(i[6:9]).tolist() # RGB
            if im in T: # {im: {tid,fid,xys,rgb}}
                T[im]['tid'].append(tid); T[im]['fid'].append(fid)
                T[im]['xys'].append(xys); T[im]['rgb'].append(rgb)
            else: T[im] = dict(tid=[tid], fid=[fid], xys=[xys], rgb=[rgb])
            #print(i, '\n', im, tid, fid, xys, rgb)
    return T # {im: {tid,fid,xys,rgb}}


#from pyproj import Proj, transform
LLA = '+proj=lonlat +ellps=WGS84 +datum=WGS84'
ECEF = '+proj=geocent +ellps=WGS84 +datum=WGS84'
##########################################################################################
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/geo.py
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/actions/export_geocoords.py
# export_geocoords._transform_image_positions(), geo.to_topocentric(), geo.to_lla()
# transform(Proj(LLA), Proj(ECEF), *v, radians=False); Proj(LLA)(x,y,inverse=True)
def export_geocoords_img(src): # image_geocoords.tsv
    rename_odm_rec(src) # rename topocentric
    from opensfm.dataset import DataSet; Auth(src)
    data = DataSet(src); ref = data.load_reference()
    SfM_cmd(src, f'export_geocoords --image-positions --proj "{LLA}"')
    with open(f'{src}/image_geocoords.tsv') as f:
        geo = csv.reader(f, delimiter='\t'); res = []
        for i,*v in geo: # i=img, v=lla(degree)
            if i=='Image': continue # if radians: v[:2] *= 180/np.pi
            v = np.float64(v[:3]); i = i.replace('.tif','') # lon,lat,alt
            o = list(data.load_exif(i)['gps'].values())[:3] # lat,lon,alt
            xyz = ref.to_topocentric(v[1], v[0], v[2])
            xyz = ref.to_topocentric(*o)-np.array(xyz)
            i = f'{i} lla={v} xyz_dif={xyz.tolist()}'
            res.append(i); print(i); #INFO(i)
    with open(f'{src}/image_geocoords_dif.txt','w') as f:
        f.write('\n'.join(res)); rename_odm_rec(src) # recover


# Ref: https://github.com/OpenDroneMap/ODM/blob/master/stages/run_opensfm.py
def rename_odm_rec(src, mod=1): # for odm: rename/recover topocentric,json
    rec = src+'/reconstruction.'; tpc = rec+'topocentric.json'
    tmp = rec+'geo.json'; rec += 'json'; Auth(src)
    if mod and os.path.isfile(tpc): os.rename(rec, tmp); os.rename(tpc, rec)        
    elif mod and os.path.isfile(tmp): os.rename(rec, tpc); os.rename(tmp, rec)


##########################################################################################
# Ref: https://opensfm.org/docs/using.html#ground-control-points
# gz->fid->track2->tid2->reconstruction2->tid2->xyz2->lla2->gcp
def SfM_gcp_gz(GPS, RTK='', thd=0.5, dst=0):
    from opensfm.dataset import DataSet
    if not os.path.isdir(RTK): RTK = GPS
    rec = RTK+'/reconstruction.topocentric.json'
    if not os.path.isfile(rec): rec = rec[:-16]+'json'
    with open(rec) as f: R = json.load(f)[0]['points']
        #for r in json.load(f): R.update(r['points'])
    R = {int(k):v['coordinates'] for k,v in R.items()}
    M = SfM_parse_gz(GPS) # {(im1,im2): [fid1,fid2]}
    T = SfM_parse_csv(RTK) # {im2: {tid2,fid2,xys2,rgb2}}
    #set.union(*[set(v['tid']) for v in T.values()]).issuperset(R)
    if dst: dst = GPS+'/mt_gcp'; os.makedirs(dst, exist_ok=True)
    ref = DataSet(RTK).load_reference(); gcp = [LLA]; npz = {}
    for (im1,im2),fid in M.items(): # fid=[fid1,fid2]
        if len(fid)<7: INFO(f'skip: {im1} {im2}'); continue # filter
        _, xy1 = SfM_feat_uv(im1, src=GPS, idx=fid[:,0]) # norm->pixel
        _, xy2 = SfM_feat_uv(im2, src=RTK, idx=fid[:,1]) # norm->pixel
        M, idx = filter_match(xy1, xy2, thd=thd); fid = fid[idx]

        idx = IDX(T[im2]['fid'], fid[:,1]) # filter: track+fid2->tid2
        tid2, fid2, xys2, rgb2 = [np.array(T[im2][k])[idx] for k in T[im2]]
        idx = IDX(tid2, list(R)) # filter: reconstruction+tid2->tid2
        tid2, fid2, xys2, rgb2 = tid2[idx], fid2[idx], xys2[idx], rgb2[idx]
        xyz2 = np.array([R[i] for i in tid2]) # reconstruction->xyz
        lla2 = np.array([ref.to_lla(*i) for i in xyz2]) # xyz->lla
        num = dict(org=len(xy1), ransac=len(fid), track=len(lla2))
        INFO(f'{im1} {im2} gz_gcp: {num}')

        idx = IDX(fid[:,1], fid2); fid = fid[idx]
        _, xy1 = SfM_feat_uv(im1, src=GPS, idx=fid[:,0]) # fid1
        _, xy2 = SfM_feat_uv(im2, src=RTK, pt=xys2) # norm->pixel
        for (lat,lon,alt),xy in zip(lla2,xy1):
            gcp += [(3*'%.15f '+2*'%4s '+'%s')%(lon,lat,alt,*xy,im1)]
        npz[f'{im1}-{im2}'] = np.hstack([xy1, xy2])
    if npz: np.savez(GPS+'/gcp_gz.npz', **npz)
    with open(GPS+'/gcp_list.txt','w') as f:
        gcp = dedup_gcp(gcp); f.write('\n'.join(gcp))
    cv2.destroyAllWindows(); return gcp # list


def dedup_gcp(gcp, eps=0.01):
    from opensfm.geo import ecef_from_lla
    if type(gcp)==str and os.path.isdir(gcp):
        with open(gcp+'/gcp_list.txt') as f:
            gcp = f.readlines()
    elif type(gcp)==str and os.path.isfile(gcp):
        with open(gcp) as f: gcp = f.readlines()
    assert type(gcp)==list; res = {}
    for i in gcp[1:]: # skip 1st-line
        lon, lat, alt, x, y, im = i.split()
        lla = np.float64([lat,lon,alt])
        k = f'{x}:{y}:{im}'
        if k in res: res[k] += [lla]
        else: res[k] = [lla]
    gcp = [gcp[0].strip()] # clear gcp
    for k,v in res.items():
        m = np.mean(v, axis=0)
        v = [ecef_from_lla(*i) for i in v]
        v -= np.asarray(ecef_from_lla(*m))
        v = np.linalg.norm(v, axis=1); #print(v)
        if np.all(v<eps): # meters
            m = (m[1], m[0], m[2], *k.split(':'))
            gcp += [(3*'%.15f '+2*'%4s '+'%s')%m]
    return gcp # list: without '\n'


########################################################
def SfM_create_gcp(GPS, RTK, mix=0):
    out = f'{GPS}/gcp_list.txt'
    if not os.path.isdir(f'{GPS}/features'): # extract
        SfM_exif_dop(GPS); SfM_cmd(GPS, 'detect_features')

    if not os.path.isfile(out):
        if os.path.isdir(f'{GPS}/matches_gcp'):
            if os.path.isdir(f'{GPS}/matches'): # backup
                os.rename(f'{GPS}/matches', f'{GPS}/matches=')
            os.rename(f'{GPS}/matches_gcp', f'{GPS}/matches')

        if not os.path.isdir(f'{GPS}/matches'):
            merge_dir(RTK+'/exif', GPS+'/exif')
            merge_dir(RTK+'/features', GPS+'/features')
            #merge_dir(RTK+'/reports/features', GPS+'/reports/features')
            #merge_json(RTK, GPS, 'reports/features.json')
            merge_json(RTK, GPS, 'camera_models.json')
            SfM_match(GPS, RTK, mix) # match (GPS,RTK)
        gcp = SfM_gcp_gz(GPS, RTK, thd=0.5, dst=0)

        os.rename(f'{GPS}/matches', f'{GPS}/matches_gcp')
        if os.path.isdir(f'{GPS}/matches='): # recover
            os.rename(f'{GPS}/matches=', f'{GPS}/matches')
    assert os.path.isfile(out)
    gcp = gcp if 'gcp' in vars() else dedup_gcp(GPS)
    with open(out,'w') as f: f.write('\n'.join(gcp))
    INFO(f'Created {len(gcp)-1} GCPs: {out}\n')


SfM_CFG = dict(use_exif_size=True, use_altitude_tag=True, feature_type='SIFT',
    sift_peak_threshold=1/15, feature_min_frames=20000, feature_process_size=0.5,
    matcher_type='FLANN', flann_algorithm='KDTREE', triangulation_type='ROBUST',
    matching_gps_neighbors=32, matching_gps_distance=150, matching_graph_rounds=50,
    align_orientation_prior='vertical', bundle_use_gcp=False, bundle_use_gps=True,
    align_method='auto', retriangulation_ratio=2, processes=max(1,os.cpu_count()//6),
    bundle_outlier_filtering_type='AUTO', optimize_camera_parameters=True) #odm'''
SYS = f'{os.cpu_count()} *'+os.popen('lscpu|grep name|cut -f2 -d:').readline()[9:-1]
##########################################################################################
def ODM_img_lla2(GPS, RTK, dst, mesh=0):
    from odm_filter import filter_reconstruct; INFO(SYS)

    gps, rtk = os.path.basename(GPS), os.path.basename(RTK)
    odm_cfg = {'end-with':'opensfm', 'feature-quality':'high'}
    sub = f'{dst}/odm-RTK-{rtk}'; merge_dir(RTK, sub+'/images')
    merge_dir(RTK, sub+'/opensfm/images'); RTK = sub
    ODM_cmd(RTK, odm_cfg); filter_reconstruct(RTK)

    GCP = f'{dst}/odm-GCP-{rtk}-{gps}'; merge_dir(GPS, GCP+'/images')
    sub = f'{dst}/sfm-GCP-{rtk}-{gps}'; merge_dir(GPS, sub+'/images')
    GPS = sub; SfM_config(GPS, SfM_CFG); SfM_create_gcp(GPS, RTK+'/opensfm')

    odm_cfg['gcp'] = f'/data/{os.path.basename(GPS)}/gcp_list.txt'
    ODM_cmd(GCP, odm_cfg); export_geocoords_img(GCP+'/opensfm')
    INFO('\nALL DONE!')


# Ref: https://github.com/OpenDroneMap/ODM/blob/master/opendm/osfm.py
# Ref: https://github.com/OpenDroneMap/ODM/blob/master/opendm/config.py
def ODM_cmd(src, cfg, sfm=SfM_CFG, gpu=''):
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
    args = ' '.join([f'--{k}={v}' for k,v in cfg.items()])
    root = os.path.dirname(os.path.abspath(src))
    cmd = f'docker run -ti --rm -v={root}:/data '
    if gpu!='all' and type(gpu)!=int: cmd += 'opendronemap/odm'
    else: cmd += f'--gpus={gpu} opendronemap/odm:gpu' # try GPU
    cmd += f' --project-path=/data {os.path.basename(src)} {args}'
    INFO('\n'+cmd); os.system(cmd)


from datetime import datetime as TT
APD = lambda v,t,c='': v[:-1]+f'\t-> {c}{t.total_seconds()}s\n'
IT = lambda v: TT.strptime(v[:23], '%Y-%m-%d %H:%M:%S,%f') # INFO_time
AT = lambda v: TT.strptime(v[-30:-5], '%a %b %d %H:%M:%S %Y') # asctime
##########################################################################################
def parse_log(src):
    with open(src, encoding='utf-8') as f:
        x = f.readlines(); idx = []
    for i,v in enumerate(x):
        if 'opensfm/bin/opensfm' in v.lower() or \
            'ODM app finished' in v: idx += [i]
    idx.append(len(x)-1); print(f'[{src}]\n'+x[0])
    key = ('match_features', 'Matched', 'Created')
    for i in range(len(idx)-1):
        a, b = idx[i], idx[i+1]
        if 'detect_features' in x[a]:
            s = [v.split() for v in x[a:b] if ': Found' in v]
            s = np.float64([(v[4],v[7][:-1]) for v in s]).sum(axis=0)
            print(a, x[a], 'Feat = %d\tT = %.3fs'%(*s,)); s = []
            for v in x[a:b]: s += [v for k in key if k in v]
            if s: s[2] = APD(s[2], IT(s[2])-IT(s[1])); print(*s)
        elif 'match_features' in x[a]:
            print(a, x[a], *[v for v in x[a:b] if 'Matched' in v])
        elif 'create_tracks' in x[a]:
            v = [v for v in x[a:b] if 'Good' in v][0]
            print(a, *x[a:a+2], APD(v, IT(v)-IT(x[a+1])))
        elif 'reconstruct ' in x[a]:
            v = [v for v in x[a:b] if 'Reconstruction' in v][0]
            print(a, *x[a:a+2], APD(v, IT(v)-IT(x[a+1])))
        elif 'export_geocoords --reconstruction' in x[a]:
            s = [s for s in x[a:b] if 'Undistorting' in s]
            print(a+4, APD(s[0], IT(s[-1])-IT(s[1])))
        elif 'app finished' in x[a] and 'Reconstruction' in v:
            print(a, APD(x[a], AT(x[a])-IT(v), '[undistort]: '))
            
        elif 'extract_metadata' in x[a]:
            x[a-2] = APD(x[a-2], IT(x[a])-IT(x[a-3])); print(a, *x[a-3:a+1])
        elif 'export_geocoords --image-positions' in x[a]:
            t = (IT(x[a])-IT(x[0])).total_seconds()/60
            print(a, x[a][:-1]+f'\t-> Total = {t*60}s = %.3f min\n'%t)
        elif 'export_openmvs' in x[a]:
            v = [v for v in x[a:b] if 'CPU:' in v][0]; print(a, x[a][:-1])
        elif 'app finished' in x[a] and 'CPU:' in v: # use previous v
            s = x[a].split(' '); s[-3] = v[:8]; s = ' '.join(s)
            print(a, APD(x[a], AT(x[a])-AT(s), '[mvs_texturing]: '))
    print(len(x), APD(x[-1], IT(x[-1])-IT(x[0])))


##########################################################################################
def parse_args():
    parser = argparse.ArgumentParser() # RTK-ODM-GPS
    parser.add_argument('--log', default=None, type=str, help='log file name')
    parser.add_argument('--dst', default=None, type=str, help='results folder')
    parser.add_argument('--gps', default=None, type=str, help='GPS images folder')
    parser.add_argument('--rtk', default=None, type=str, help='RTK images folder')
    parser.add_argument('--min', default=4000, type=int, help='min feature number')
    #parser.add_argument('--mesh', default=0, type=int, help='odm_texturing_25d')
    parser.add_argument('--type', default='sift', type=str, help='feature type')
    parser.add_argument('--quality', default=0.5, type=float, help='feature quality')
    parser.add_argument('--cpu', default=os.cpu_count()//6, type=int, help='concurrency')
    args = parser.parse_args(); # print(args)
    if args.log:
        os.makedirs(args.dst, exist_ok=True)
        args.log = os.path.join(args.dst, args.log)
        sys.stderr = sys.stdout = open(args.log, 'a+')
    SfM_CFG['processes'] = min(os.cpu_count(), max(1, args.cpu))
    SfM_CFG['feature_process_size'] = args.quality
    SfM_CFG['feature_type'] = args.type.upper()
    SfM_CFG['feature_min_frames'] = args.min; return args


##########################################################################################
if __name__ == '__main__':
    # python3 ODM_SfM.py >> log.txt 2>&1'''
    x = parse_args()
    ODM_img_lla2(x.gps, x.rtk, x.dst)
    #parse_log(x.log)

