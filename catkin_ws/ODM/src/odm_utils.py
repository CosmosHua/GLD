#!/usr/bin/python3
# coding: utf-8

import os, sys, cv2, json
import numpy as np, shutil


RSZ = lambda x,r: cv2.resize(x, None, fx=r, fy=r)
SfM_DIR = os.popen('locate bin/opensfm').readline().strip()
SfM_DIR = SfM_DIR[:SfM_DIR.find('/bin')]; sys.path.append(SfM_DIR)
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
    tracks = data.load_tracks_manager()
    imgs = tracks.get_shot_ids()
    if not data.reference_lla_exists():
        data.invent_reference_lla(imgs)

    camera_priors = data.load_camera_models()
    common_tracks = tracking.all_common_tracks(tracks)
    pairs = compute_image_pairs(common_tracks, camera_priors, data)
    imgs = set(imgs); report = {'candidate_image_pairs': len(pairs)}
    for im1, im2 in pairs:
        if im1 in imgs and im2 in imgs:
            report[im1+' & '+im2] = log = {}
            v, p1, p2 = common_tracks[im1, im2]
            rec, log['bootstrap'] = bootstrap_reconstruction(
                data, tracks, camera_priors, im1, im2, p1, p2)
            if rec:
                imgs.remove(im1); imgs.remove(im2)
                rec, log['grow'] = grow_reconstruction(
                    data, tracks, rec, imgs, camera_priors, gcp)
                result.append(rec)
    result = sorted(result, key=lambda x: -len(x.shots))
    data.save_reconstruction(result)
    report['not_reconstructed_images'] = list(imgs)
    with open(f'{src}/reports/reconstruction.json','w') as f:
        json.dump(report, f, indent=4)


from matplotlib import pyplot as plt
from matplotlib import use; use('TkAgg')
IMG = ('jpg','png','tif','bmp','gif','jpeg')
##########################################################################################
def read_lla(src): # (lon,lat)
    src = os.path.abspath(src); lla = {}
    js = src if os.path.isfile(src) else \
        src+'/'+os.path.basename(src)+'.json'
    if os.path.isfile(js):
        with open(js) as f: lla = json.load(f)
        print(src, len(lla)); return lla

    from opensfm.exif import get_xmp
    for root,sub,files in os.walk(src):
        for im in files:
            im = os.path.join(root, im)
            if im.split('.')[-1].lower() not in IMG: continue
            with open(im,'rb') as f: xmp = get_xmp(f)[0]
            for k in xmp: # for various keys
                if '@drone-dji:GpsLon' in k: lon = float(xmp[k])
                if '@drone-dji:GpsLat' in k: lat = float(xmp[k])
            lla[os.path.relpath(im,src)] = [lon,lat]
    with open(js,'w') as f: json.dump(lla, f, indent=4)
    print(src, len(lla)); return lla


def sub_lla(lla, rt, pt='ct'):
    fig, ax = plt.subplots(); res = {}
    for k,v in lla.items(): res.update(v) # merge
    rt = np.sqrt(rt*len(lla)/len(res) if rt>1 else rt)

    color = ['r','g','b','k','y','c','m','pink']
    color = np.random.choice(color, len(lla), replace=0)
    p = [np.vstack([*v.values()])*1E5 for k,v in lla.items()]
    x = np.vstack(p); o = x.min(axis=0); x -= o # origin
    for v,c in zip(p,color): plt.scatter(*(v-o).T, c=c)
    plt.xlabel('longitude'); plt.ylabel('latitude')

    mi, mx = x.min(axis=0), x.max(axis=0)
    BD = lambda v: [*(v+(mi-v)*rt), *(v+(mx-v)*rt)]
    if pt=='tr': a,b,c,d = BD(mx) # top-right
    elif pt=='bl': a,b,c,d = BD(mi) # bottom-left
    elif pt=='ct': a,b,c,d = BD(x.mean(axis=0)) # center
    elif pt=='tl': a,b,c,d = BD([mi[0],mx[1]]) # top-left
    elif pt=='br': a,b,c,d = BD([mx[0],mi[1]]) # bottom-right
    elif type(pt)!=str and len(pt)>1: # any (lon.lat)->clip
        a,b,c,d = BD(np.clip(pt[:2], mi, mx)) # local,meter
        #a,b,c,d = BD(np.clip(np.array(pt[:2])*1E5-o, mi, mx))
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((a,b), c-a, d-b, fill=0))
    ax.add_patch(Rectangle(mi-3, *(mx-mi+6), alpha=0.05))
    return {k:[s for s,(i,j) in zip(v,q-o) if a<=i<=c and
        b<=j<=d] for (k,v),q in zip(lla.items(),p)}


def sub_data(src, rt=100, pt='ct', dst=''):
    src = [src] if type(src)==str else src
    src = [os.path.abspath(i) for i in src]
    lla = {i:read_lla(i) for i in src}
    sub = sub_lla(lla, rt, pt)
    if dst:
        for k,v in sub.items():
            d = os.path.join(dst, os.path.basename(k))
            os.makedirs(d, exist_ok=True)
            for i in v:
                j = os.path.join(d, os.path.basename(i))
                shutil.copyfile(os.path.join(k,i), j)
    for k,v in sub.items(): print(k,len(v),'\n',v)
    plt.show(); return sub


# https://github.com/isl-org/Open3D/blob/master/examples/python/pipelines/icp_registration.py
##########################################################################################
def icp(src, dst, init_pose=None, N=30, thd=0.001):
    '''The Iterative Closest Point method
    Input:
        src: Nx3 numpy array of source 3D points
        dst: Nx3 numpy array of target 3D points
        init_pose: 4x4 homogeneous transformation
        N: exit algorithm after N
        thd: convergence criteria
    Output:
        TF: final homogeneous transformation
        dist: Euclidean distances (errors) of the nearest neighbor'''
    # make points homogeneous, copy them so as to maintain the originals
    A = np.ones((src.shape[0],4)); A[:,:3] = src
    B = np.ones((dst.shape[0],4)); B[:,:3] = dst; err = 0
    # apply the initial pose estimation
    if init_pose is not None: A = A.dot(init_pose.T)
    for i in range(N):
        # find the nearest neighbours between the current A and B points
        dist, idx1, idx2 = nearest_neighbor(A[:,:3], B[:,:3], 2)
        # compute the transformation between the current A and nearest B points
        TF = svd_fit(A[idx1,:3], B[idx2,:3]); A = A.dot(TF.T)
        pre = err; err = np.mean(dist)
        if abs(pre-err) < thd: break
    # calculcate final tranformation
    TF = svd_fit(src, A[:,:3]); return TF, dist


def nearest_neighbor(A, B, thd):
    from scipy.spatial import KDTree
    tree = KDTree(B, leafsize=32)
    idx1 = []; idx2 = []; dist = []
    for i in range(len(A)):
        dis, idx = tree.query(A[i], k=1, p=2)
        if dis < thd:
            idx1.append(i)
            idx2.append(idx)
            dist.append(dis)
    return dist, idx1, idx2


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
def draw_registration(A, B, TF):
    import open3d as o3d, copy
    B = copy.deepcopy(B)
    A = copy.deepcopy(A); A.transform(TF)
    A.paint_uniform_color([1, 0.706, 0])
    B.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw([A, B])


########################################################
def svd_gcp(A, B=0, sz=1):
    import open3d as o3d # A=[path,ndarray]
    if type(A)==np.ndarray and type(B)!=type(A):
        return svd_fit(A[:,:3], A[:,3:])
    elif type(A)==type(B): return svd_fit(A,B)
    A = o3d.io.read_point_cloud(A+'/gps.ply')
    B = o3d.io.read_point_cloud(A+'/rtk.ply')
    #A,B = A.voxel_down_sample(sz), B.voxel_down_sample(sz)
    x = np.asarray(A.points); y = np.asarray(B.points)
    TF = svd_fit(x, y); draw_registration(A, B, TF)


##########################################################################################
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # os.chdir('../../data_part'); src = ['20211108','20211109']
    # sub_data(src, rt=1/6, pt=[111,-22], dst='')

    # os.chdir('/media/hua/20643C51643C2BC2/GLD_Data')
    # dst = os.path.expanduser('~/Git_GLD/ODM/data_part')
    # src = ['20211108_南京新生圩_精灵4RTK_正射_60h_旁70-航75_晴']
    # src += ['20211109_南京新生圩_御2GPS_正射_120h_旁65-航75_晴']
    # sub_data(src, rt=100, pt=[400,0], dst=dst)

