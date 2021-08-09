#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, sys, cv2, json
import infer4robot as Seg
import geometry as Geo
from objects import *

import rospy, roslib
from doormsgs.msg import doorinfo # custom
from actionlib_msgs.msg import GoalStatusArray
from cv_bridge import CvBridge, CvBridgeError
# cv_bridge does not support CompressedImage in python
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, LaserScan
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer


RSZ = Seg.RES # cv2.resize, for short
fun = lambda x: np.cos(x*np.pi)/2+0.5 # for weight
INT = lambda x,f=1: tuple(int(i*f) for i in x) # round
##########################################################################################
class ros_seg:
    def __init__(self, data_dir, bim=None, wh=1.0):
        '''Initialize ros publisher, ros subscriber'''
        rospy.init_node('ros_seg', anonymous=True); self.wh = wh
        pred, meta = Seg.Setup4Infer(data_dir, 0.6); self.sc = 0
        self.Infer = lambda x,sh: Seg.Infer2Img(x, pred, meta, wh, sh)
        self.obj = objects(bim); self.log = log_init(self.obj)
        
        pos_robot = TPC(['/base_link_pose','/amcl_pose'])
        cam_info = TPC(['/rgb/camera_info','/zed_node/left/camera_info'])
        img_depth = TPC(['/depth_to_rgb/image_raw','/zed_node/depth/depth_registered'])
        img_rgb = TPC(['/rgb/image_raw/compressed','/zed_node/left/image_rect_color/compressed'])
        self.d = 3.5 if 'zed' in img_depth else 9 # max_depth: zed=3.5, kinect=9

        out_seg = '/output/image_seg/compressed'; out_door = '/output/door_pose'
        self.out_door = rospy.Publisher(out_door, doorinfo, queue_size=10)
        self.out_seg = rospy.Publisher(out_seg, CompressedImage, queue_size=10)
        #rospy.Subscriber(img_depth, Image, self.callback_depth, queue_size=1) # show depth
        #rospy.Subscriber(img_rgb, CompressedImage, self.callback_rgb, queue_size=1) # show rgb
        #'''
        cam_info = Subscriber(cam_info, CameraInfo)
        img_depth = Subscriber(img_depth, Image, queue_size=1)
        img_rgb = Subscriber(img_rgb, CompressedImage, queue_size=1)
        pos_robot = Subscriber(pos_robot, PoseWithCovarianceStamped)
        #capt = Subscriber('/move_base/status', GoalStatusArray)
        
        #sync = TimeSynchronizer([img_rgb, img_depth, pos_robot], 10)
        sync = ApproximateTimeSynchronizer([img_rgb, img_depth, pos_robot,
                cam_info], queue_size=10, slop=1, allow_headerless=True)
        sync.registerCallback(self.callback_multi) #callback_test #'''


    ########################################################
    def callback_depth(self, depth):
        dp = RSZ(parse_depth(depth,'U'), self.wh)
        cv2.imshow('depth', dp); cv2.waitKey(2)


    def callback_rgb(self, rgb):
        seg, out = self.seg_rgb(rgb); self.pub_seg(seg)
        cv2.imshow('seg', seg); cv2.waitKey(2)


    # cv2.imshow in Synchronizer leads to crash
    def callback_test(self, rgb, depth, cpos, cam):
        print('rgb:',rgb.header.stamp.to_sec())
        print('pos:',cpos.header.stamp.to_sec())
        print('dep:',depth.header.stamp.to_sec())
        t0 = rospy.Time.now()
        dp = parse_depth(depth,'U')
        rgb = np.frombuffer(rgb.data, 'uint8')
        rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
        print((rospy.Time.now()-t0).to_sec())
        #cv2.imshow('dp',dp); cv2.imshow('rgb',rgb)
        cv2.waitKey(5); print('end\n')


    def callback_multi(self, rgb, depth, cpos, cam):
        print('\n%s:'%cpos.header.stamp.to_sec())
        cp = cpos.pose.pose; p,q = cp.position, cp.orientation
        cp = 'Robot: (x=%.2f, y=%.2f, qz=%.2f, qw=%.2f)'%(p.x,p.y,q.z,q.w)
        seg, out, rgb = self.seg_rgb(rgb, cp, mod=2); Geo.CK = cam.K

        depth = parse_depth(depth); self.scale(depth.shape[0])
        self.update_pos(out, cpos, depth, seg); self.pub_seg(seg)
        '''
        dp = RSZ(depth/depth.max()*255, self.wh).astype('uint8')
        dp = cv2.cvtColor(dp, cv2.COLOR_GRAY2BGR)
        dp = cv2.addWeighted(dp,1.0, rgb,0.2, 0)
        if dp.shape!=seg.shape: cv2.imshow('depth', dp)
        else: seg = np.concatenate([seg,dp], axis=0)'''
        #cv2.imshow('seg',seg); cv2.waitKey(5) # ->crash


    ########################################################
    def scale(self, h): # Ref: Seg.NewSZ
        if type(h)==np.ndarray: h = h.shape[0]
        if type(self.wh)==float: self.sc = self.wh
        elif type(self.wh)==int: self.sc = self.wh/h
        elif type(self.wh) in (tuple,list): self.sc = self.wh[1]/h


    def seg_rgb(self, rgb, info=None, mod=1):
        # CompressedImage->CV2: frombuffer/fromstring
        rgb = np.frombuffer(rgb.data, 'uint8')
        rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
        if mod<1: return RSZ(rgb, self.wh)
        seg = self.Infer(rgb, info) # sys.version
        return seg if mod==1 else (*seg, RSZ(rgb, self.wh))


    def pub_seg(self, seg):
        msg = CompressedImage(format='jpeg')
        msg.data = np.asarray(cv2.imencode('.jpg',seg)[1]).tobytes()
        msg.header.stamp = rospy.Time.now(); self.out_seg.publish(msg)


    def pub_door(self, pos, cpos):
        if type(pos)==str:
            id = pos; true = self.obj.find(id)
            pos = dict(pos=true['position'],drt=true['direction'])
        else: id, pos = pos; true = self.obj.find(id)

        msg = doorinfo(header=cpos.header)
        msg.camera_pose = cpos.pose.pose; msg.header.frame_id = id

        p = msg.door_pose.position; q = msg.door_pose.orientation
        p.x, p.y, p.z = pos['pos']; q.x, q.y, q.z = pos['drt']

        if type(true)!=dict: msg.door_pose_true = msg.door_pose
        else: obj2pos(msg.door_pose_true, true)
        self.out_door.publish(msg)


    ########################################################
    def new_pos(self, out, TR, depth, d, mod=1):
        cid = np.asarray(out.pred_classes,int)
        uv = get_UV(out, depth, d)[0] #(K=1,[2,N])
        ct = np.asarray(out.pred_boxes.get_centers(),int)
        #pt = Geo.Camera_2Dto3D(ct[:,0], ct[:,1], depth)
        #pt = Geo.Camera2World(TR, pt) # pt[:,i]

        if type(uv)!=np.ndarray: return # uv(2,N)
        ptc = Geo.Camera_2Dto3D(*uv, depth) #(3,N)
        ptw = Geo.Camera2World(TR, ptc) # world(3,N)
        drt = Geo.PCA(ptw)[1][-1] # norm-direction
        #if abs(drt[2])>0.2: return # invalid drt

        #pos = Geo.Camera2World(TR, ptc.mean(axis=1))
        #pos = Geo.Camera2World(TR, ptc).mean(axis=1)
        pos = ptw.mean(axis=1); dp = depth[uv[1],uv[0]]
        pos = Geo.uvRay2Plane(ct[0], [*drt,*pos], TR)
        pos = [cid[0],pos,drt]; #print('world:', pos)

        if mod==1: # update: append new objects
            q = (dp.mean()/d)**0.5 * fun(drt[2])**4
            idx = self.obj.update([*pos,ptw], q=q)
            if idx!=None: return pos, self.obj[idx], uv
        elif mod==2: # no update: just find prestored
            idx = self.obj.update2([*pos,ptw])
            if idx!=None: return pos, self.obj[idx], uv
        else: return pos, dp # no update/find


    # update pos/obj by uv+world coordinate
    def update_pos(self, out, cpos, depth, im, thd=0.3):
        box = out.pred_boxes.tensor.numpy() #(K,4)
        box = np.delete(box, (1,3), axis=1) #(K,2)
        TR = get_TR(cpos); LR,ID = get_LR(self.obj, TR)
        ct = np.asarray(out.pred_boxes.get_centers(),int)
        for i,bx in enumerate(box):
            iop = IoP(bx,LR); ids = self.obj['id']
            idx = iop.argsort()[-3:][::-1]
            #print('iop:', {ids[ID[i]]:iop[i] for i in idx})
            if len(iop)>0 and max(iop)>thd: # find by uv
                idx = idx[0]; ob = self.obj[ID[idx]]
                v = [ct[i][1]+v for v in range(-33,34)]
                u = [[u]*len(v) for u in LR[:,idx]]
                uv = np.vstack([np.hstack(u),v+v])
                pos = self.new_pos(out[i], TR, depth, self.d, mod=0)
                pos, dp = pos if pos!=None else (pos,pos)
            else: # update/find by world coordinate
                pos = self.new_pos(out[i], TR, depth, self.d, mod=2)
                #pos = self.new_pos(out[i], TR, depth, self.d, mod=1)
                if pos==None: continue
                pos, ob, uv = pos; dp = depth[uv[1],uv[0]]
            pos = log_append(self.log, ob, pos, dp, iop)
            self.pub_door(pos, cpos); print(pos)

            if type(im) in (np.ndarray, cv2.UMat):
                draw_pos(im, uv, pos, self.sc, ct[i])
                save_img(im, ob['id'], out[i], depth, self.d)


    # update pos/obj just by world coordinate
    def update_pos_w(self, out, cpos, depth, im):
        TR = get_TR(cpos)
        for i in range(len(out)):
            pos = self.new_pos(out[i], TR, depth, self.d, mod=1)
            if pos==None: continue
            pos, ob, uv = pos; dp = depth[uv[1],uv[0]]
            pos = log_append(self.log, ob, pos, dp)
            self.pub_door(pos, cpos); print(pos)
            if type(im) in (np.ndarray, cv2.UMat):
                draw_pos(im, uv, pos, self.sc) #,ct[i]
                save_img(im, ob['id'], uv, depth, self.d)


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


def get_TR(cpos):
    ps = cpos.pose.pose.position
    ps = np.asarray([ps.x, ps.y, ps.z])
    qa = cpos.pose.pose.orientation
    qa = np.asarray([qa.w, qa.x, qa.y, qa.z])
    return Geo.Quart2TR(qa, ps)


# u: horizontal-right, v: vertical-down
def get_UV(out, depth, d=4, N=5, k=2):
    bbox = out.pred_boxes; UV = [] # len(out)
    mask = out.pred_masks.byte() # bool->uint8
    for b,m in zip(bbox, mask): # find contours
        u1,v1,u2,v2 = b; v = (v2+1-v1)/4
        u1,u2,v1,v2 = INT([max(0,u1-k),u2+k,v1+v,v2-v])
        m = m[v1:v2+1,u1:u2+1]; w = depth.shape[1]

        # Find contour: horizontal scan
        v = np.asarray(range(m.shape[0]))+v1
        left = np.vstack([m.numpy().argmax(axis=1)+u1,v])
        right = np.vstack([m.argmax(dim=1).numpy()+u1,v])
        '''# expand left & right contours outward
        left = m.numpy().argmax(axis=1)+u1
        right = m.argmax(dim=1).numpy()+u1
        left = [left-i for i in range(k) if (left>=i).all()]
        right = [right+i for i in range(k) if (right+i<w).all()]
        left = np.hstack([np.vstack([u,v]) for u in left]) #(2,N)
        right = np.hstack([np.vstack([u,v]) for u in right]) #(2,N)'''

        u,v = uv = np.hstack([left,right[:,::-1]]) #(2,N)
        id = np.where((m[v-v1,u-u1]>0) & (depth[v,u]>0) & (depth[v,u]<d))[0]
        left = len(np.where(id<left.shape[1])[0]); right = len(id)-left
        UV.append(uv[:,id] if min(left,right)>N else None) #(2,N)

        # Find contour: cv2.findContours
        '''m = m.numpy().astype('uint8'); dt = np.asarray([[u1],[v1]]) #(2,1)
        cot, hi = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        u,v = cot = np.concatenate(cot,axis=0).reshape(-1,2).T # (N,2)->(2,N)
        id = np.where(m[v,u]>0)[0]; u,v = cot = cot[:,id] + dt
        id = np.where((depth[v,u]>0) & (depth[v,u]<d))[0]
        UV.append(cot[:,id] if len(id)>N else None) #(2,N)'''
    return UV # (len(out),[2,N])


def get_LR(obj, TR): # u:(left,right)
    w = np.cross(obj['direction'],[0,0,1])
    w = w*obj['width'].reshape(-1,1)/2 #(N,3)
    p = obj['position']; N = len(p) #(N,3)
    # pt(3,2N)->uv(2,2n)->u(2,n)->sort(left<right)
    LR = Geo.World2Camera(TR, np.vstack([p-w,p+w]).T)
    ID = np.where(np.bitwise_and(LR[2,:N]>0,LR[2,N:]>0))[0]
    LR = LR[:,np.hstack([ID,ID+N])] # forword z>0, N->n
    LR = Geo.Camera_3Dto2D(LR)[0].reshape(2,-1) # uv->u
    return np.sort(LR,axis=0), ID # u(2,n):left+right


def draw_pos(im, uv, pos, sc=1, ct=[]):
    ct = ct if len(ct)>1 else uv.mean(axis=1)
    ct, uv = INT(ct,sc), (uv*sc).T.astype(int)
    if type(pos)!=tuple: pos = pos, {'pos':(0,0,0)}
    info = '%s:(%.2f,%.2f,%.2f)' % (pos[0],*pos[1]['pos'])
    color = INT(np.random.randint(0,256,3))
    cv2.circle(im, ct, 1, color, 2) # uv(2,N)->(N,2)
    cv2.putText(im, info, ct, 4, 0.55, color) # info
    for i in uv: cv2.circle(im, tuple(i), 1, color, 2)
    #cv2.polylines(im, [uv], False, color, 2)
    #cv2.drawContours(im, [uv], -1, color, 2)


# Intersection over Present
def IoP(lr, LR): # lr(2,), LR(2,N)
    assert lr.shape[0]==LR.shape[0]==2
    if len(lr.shape)>1: lr, LR = LR, lr
    inter_lf = np.where(LR[0]>lr[0], LR[0], lr[0])
    inter_rt = np.where(LR[1]<lr[1], LR[1], lr[1])
    return (inter_rt-inter_lf)/(lr[1]-lr[0]) #(N,)


# Maybe: im.shape!=depth.shape
def save_img(im, id, uv, depth, d=4):
    if type(uv)==Seg.Instances: # out[i]
        uv = get_UV(uv, depth, d)[0]
    if type(uv)!=np.ndarray: return
    # compute norm vector in camera space
    h,w = depth.shape[:2]; u,v = uv #(2,N)
    u = u.clip(0,w-1); v = v.clip(0,h-1)
    pc = Geo.Camera_2Dto3D(u, v, depth)
    dz = abs(Geo.PCA(pc)[1][-1][2]) # norm_z
    du = 1-abs(2*u.mean()/w-1); global dir,idz
    # norm_z(in camera)>thd; u/v near center
    if dz>0.9 and du>2/4:
        if id not in idz: idz[id] = []
        dz *= du*(max(u)-min(u)); p = idz[id]; p += [dz]
        #if max(p)>dz or len(p)<2: return
        
        dst = '%s/%s.jpg'%(dir,id) #'_%.2f'%dz
        h,w = im.shape[:2]; wh = int(w/4),int(h/6)
        sc = 2*(h/720); info = 'Saved: %s.jpg'%id
        if max(p)<=dz:
            cv2.imwrite(dst,im); print('=>',dst)
            cv2.putText(im, dst, wh, 4, sc, (0,255,255), 2)
        elif os.path.isfile(dst):
            cv2.putText(im, info, wh, 4, sc, (0,255,255), 2)
        return dst


from shutil import rmtree
##########################################################################################
if __name__ == '__main__':
    dir = 'Capture'; idz = {}
    if not os.path.isdir(dir): os.mkdir(dir)
    else: rmtree(dir); os.mkdir(dir) # clear dir
    root = os.path.expanduser('~/GLD_Git/Data_Door/coco_door3')
    seg = ros_seg(root, 'BIM5.json', 1.0)

    try: rospy.spin()
    finally: log_output(seg.log)

