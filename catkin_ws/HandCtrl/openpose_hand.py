#!/usr/bin/python3
# coding: utf-8

'''
sudo apt-get install libopencv-dev # tcl-dev tk-dev python3-tk
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
cd openpose; git submodule update --init --recursive --remote
sudo bash scripts/ubuntu/install_deps.sh # for caffe
mkdir build; cd build; cmake -DBUILD_PYTHON=ON -DUSE_CUDNN=OFF ..
make -j`nproc`; sudo make install # install pyopenpose
'''
import os, sys, cv2, argparse
try: # for sudo make install
    sys.path.append('/usr/local/python')
    import openpose.pyopenpose as pyop
except:
    dir = os.popen('locate */pyopenpose.*.so')
    sys.path.append(os.path.dirname(dir.readline()))
    import pyopenpose as pyop
from time import time


RSZ = lambda im,s=1: cv2.resize(im, None, fx=s, fy=s)
##########################################################################################
def hand_pose(model_dir, left=0):
    params = {'model_folder': model_dir}
    params['net_resolution'] = '240x192'
    #params['disable_blending'] = True
    #params['number_people_max'] = 1
    params['hand'] = True; #params['face'] = True
    params['body'] = 0; params['hand_detector'] = 2
    hands_boxes = [ # left & right hands: only 1-person
        [pyop.Rectangle(0,0,0,0), pyop.Rectangle(0,0,640,640)] ]

    #opWrapper = pyop.WrapperPython(pyop.ThreadManagerMode.Synchronous)
    #opWrapper.configure(params); opWrapper.execute() # webcam

    '''opWrapper = pyop.WrapperPython(pyop.ThreadManagerMode.AsynchronousOut)
    opWrapper.configure(params); opWrapper.start() # webcam
    while cv2.waitKey(5)!=27: # faster
        t = time(); x = pyop.VectorDatum()
        if not opWrapper.waitAndPop(x): continue
        x = x[0]; im = x.cvOutputData; fps = 'FPS=%.1f'%(1/(time()-t))
        im = cv2.putText(im, fps, (5,20), 4, 0.7, (0,255,255), 1, 16)
        cv2.imshow('OpenPose', im) # hand_standalone unsupported
    cv2.destroyAllWindows()#'''

    opWrapper = pyop.WrapperPython()
    opWrapper.configure(params); opWrapper.start()
    x = pyop.Datum(); cap = cv2.VideoCapture(-1)
    hand_boxes = hand_box(cap, left)
    while cv2.waitKey(5)!=27: # slower
        t = time(); _, im = cap.read(); x.cvInputData = im
        if params['hand_detector']==2: x.handRectangles = hand_boxes
        opWrapper.emplaceAndPop(pyop.VectorDatum([x]))
        im = x.cvOutputData; fps = 'FPS=%.1f'%(1/(time()-t))
        im = cv2.putText(im, fps, (5,20), 4, 0.7, (0,255,255), 1, 16)
        cv2.imshow('OpenPose', RSZ(im,1.5)) # cv2.LINE_AA=16
    cv2.destroyAllWindows(); cap.release()#'''

    #print('Body keypoints:', x.poseKeypoints.shape) # (N,25,3)
    #print('Face keypoints:', x.faceKeypoints.shape) # (N,70,3)
    #print('Left hand keypoints:', x.handKeypoints[0].shape) # (N,21,3)
    #print('Right hand keypoints:', x.handKeypoints[1].shape) # (N,21,3)


def hand_box(sz, left=0):
    non = pyop.Rectangle(0, 0, 0, 0)
    if type(sz)==cv2.VideoCapture: # wh
        sz = max(sz.get(3), sz.get(4))
    box = pyop.Rectangle(0, 0, sz, sz)
    return [[box,non] if left else [non,box]]


##########################################################################################
def pick_tp(tp):
    while True:
        TP = dict(rospy.get_published_topics())
        for i in tp: # pick topic orderly
            if i in TP: return i


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
        return RSZ(im[:,m:w-m], 1/s)
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


##########################################################################################
if __name__ == '__main__':
    os.chdir((os.path.dirname(os.path.abspath(__file__))))
    hand_pose('./openpose/models')
