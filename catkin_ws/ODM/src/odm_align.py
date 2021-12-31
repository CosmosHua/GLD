#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, json, shutil


RSZ = lambda x,r: cv2.resize(x, None, fx=r, fy=r)
##########################################################################################
def cv_sift_feat(src, sh=0):
    for i in os.listdir(src):
        im = cv2.imread(f'{src}/{i}')
        sift = cv2.SIFT_create() # cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(im, None)
        out = cv2.drawKeypoints(im, keypoints, None)

        cv2.imwrite(i+'_sift.jpg', out)
        cv2.imshow('cv_feat', RSZ(out, 0.4))
        if cv2.waitKey(sh)==27: break
    cv2.destroyAllWindows()


def cv_sift_match(src, sh=0):
    for (i,j) in combinations(os.listdir(src),2):
        im1, im2 = cv2.imread(src+'/'+i), cv2.imread(src+'/'+j)
        sift = cv2.SIFT_create() # cv2.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(im1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(im2, None)

        '''bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        out = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches[:100], im2, flags=2)
        cv2.imwrite('-'.join([i,j])+'_bf.jpg', out) #'''

        FLANN_INDEX_KDTREE = 1; search_params = dict(checks=50) # OR:{}
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        matches = sorted(matches, key=lambda x: x[0].distance/x[1].distance)
        out = cv2.drawMatchesKnn(im1, keypoints1, im2, keypoints2, matches[:100], im2, flags=0)
        cv2.imwrite('-'.join([i,j])+'_flann.jpg', out) #'''

        cv2.imshow('cv_match', RSZ(out, 0.2))
        if cv2.waitKey(sh)==27: break
    cv2.destroyAllWindows()


##########################################################################################
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/reconstruction.py
# Ref: https://github.com/mapillary/OpenSfM/blob/main/opensfm/actions/reconstruct.py
# reconstruct.run_dataset(), reconstruction.incremental_reconstruction():
# compute_image_pairs(), bootstrap_reconstruction(), grow_reconstruction()
def SfM_reconstruct(src): # incremental_reconstruction
    from opensfm.dataset import DataSet
    from opensfm.reconstruction import (tracking, compute_image_pairs,
        bootstrap_reconstruction, grow_reconstruction)

    data = DataSet(src); result = []
    gcp = data.load_ground_control_points()
    tracks_manager = data.load_tracks_manager()
    imgs = tracks_manager.get_shot_ids()
    if not data.reference_lla_exists():
        data.invent_reference_lla(imgs)

    camera_priors = data.load_camera_models()
    common_tracks = tracking.all_common_tracks(tracks_manager)
    pairs = compute_image_pairs(common_tracks, camera_priors, data)
    imgs = set(imgs); report = {'candidate_image_pairs': len(pairs)}
    for im1, im2 in pairs:
        if im1 in imgs and im2 in imgs:
            report[im1+' & '+im2] = log = {}
            v, p1, p2 = common_tracks[im1, im2]
            rec, log['bootstrap'] = bootstrap_reconstruction(
                data, tracks_manager, camera_priors, im1, im2, p1, p2)
            if rec:
                imgs.remove(im1); imgs.remove(im2)
                rec, log['grow'] = grow_reconstruction(
                    data, tracks_manager, rec, imgs, camera_priors, gcp)
                result.append(rec)
    result = sorted(result, key=lambda x: -len(x.shots))
    data.save_reconstruction(result)
    report['not_reconstructed_images'] = list(imgs)
    with open(f'{src}/reports/reconstruction.json','w') as f:
        json.dump(report, f, indent=4)


##########################################################################################
def cal_lla_axis(lat, lon, alt=0): # calculate origin & axis
    # WGS84='EPSG:4326', BeiJing(BJ54_CM_6_20)='EPSG:21460'
    from pyproj import Transformer
    ECEF = '+proj=geocent +ellps=WGS84 +datum=WGS84'
    T = Transformer.from_crs('EPSG:4326', ECEF, always_xy=True)

    lla = np.array([lon,lat,alt]); dt = np.diag([1E-4]*3)
    o = np.array(T.transform(*lla, radians=False))

    x = np.array(T.transform(*(lla+dt[0]), radians=False))
    x -= np.array(T.transform(*(lla-dt[0]), radians=False))
    x[2] = 0; x /= np.linalg.norm(x)

    y = np.array(T.transform(*(lla+dt[1]), radians=False))
    y -= np.array(T.transform(*(lla-dt[1]), radians=False))
    z = np.cross(x,y); z /= np.linalg.norm(z)
    y = np.cross(z,x); return np.array([o,x,y,z]) # (4,3)


########################################################
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
    s = cal_lla_axis(*s.values()) # lat,lon,alt
    with open(new_lla) as f: t = json.load(f)
    t = cal_lla_axis(*t.values()) # lat,lon,alt

    obj = '/odm_textured_model_geo.obj'
    with open(src+obj, encoding='utf-8') as f:
        lines = f.readlines()
    for i,v in enumerate(lines):
        v = v.split()
        if len(v) and v[0]=='v':
            v = np.float64(v[1:4]).dot(s[1:4])
            v = t[1:4].dot(v+s[0]-t[0]) # x,y,z
            lines[i] = 'v %9.6f %9.6f %9.6f\n'%(*v,)

    if os.path.isdir(dst): dst += obj
    elif dst.endswith('.obj'): dst = src+obj
    elif dst: dst = src+obj[:-4]+f'_{dst}'+obj[-4:]
    with open(dst, 'w', encoding='utf-8') as f:
        f.writelines(lines); print('Align:', dst)


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
    if len(plane): # split obj
        from odm_split import odm_split
        odm_split(src, plane)


##########################################################################################
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('data_part/out-sift'); #odm_align('odm-RTK-20211108')
    #odm_align('odm-GPS-20211109', 'odm-RTK-20211108')
    #odm_align('odm-GCP-20211108-20211109', 'odm-RTK-20211108')

