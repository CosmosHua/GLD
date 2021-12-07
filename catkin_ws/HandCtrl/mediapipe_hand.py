#!/usr/bin/python3
# coding: utf-8

import os, cv2
import numpy as np
import mediapipe as mp


##########################################################################################
def mp_face_det(cap):
    det = mp.solutions.face_detection; draw = mp.solutions.drawing_utils
    box = draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
    with det.FaceDetection(min_detection_confidence=0.5) as pose:
        while cv2.waitKey(5)!=27:
            _, im = cap.read(); res = mp_infer(pose,im)
            if res.detections:
                for i in res.detections:
                    draw.draw_detection(im, i, bbox_drawing_spec=box)
            cv2.imshow('MediaPipe', im)
        cv2.destroyAllWindows()


########################################################
def mp_face_mesh(cap):
    det = mp.solutions.face_mesh; draw = mp.solutions.drawing_utils
    face = draw.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1)
    with det.FaceMesh(static_image_mode=False, max_num_faces=5,
        min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cv2.waitKey(5)!=27:
            _, im = cap.read(); res = mp_infer(pose,im)
            if res.multi_face_landmarks:
                for i in res.multi_face_landmarks:
                    draw.draw_landmarks(im, i, det.FACE_CONNECTIONS,
                        landmark_drawing_spec=face, connection_drawing_spec=face)
            cv2.imshow('MediaPipe', im)
        cv2.destroyAllWindows()


########################################################
def mp_hand_pose(cap):
    det = mp.solutions.hands; draw = mp.solutions.drawing_utils
    hand = draw.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=2)
    with det.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cv2.waitKey(5)!=27:
            _, im = cap.read(); res = mp_infer(pose,im)
            if res.multi_hand_landmarks:
                for i in res.multi_hand_landmarks:
                    draw.draw_landmarks(im, i, det.HAND_CONNECTIONS,
                        landmark_drawing_spec=hand, connection_drawing_spec=hand)
            cv2.imshow('MediaPipe', im)
        cv2.destroyAllWindows()


########################################################
def mp_body_pose(cap):
    det = mp.solutions.pose; draw = mp.solutions.drawing_utils
    body = draw.DrawingSpec(color=(255,)*3, thickness=1, circle_radius=2)
    with det.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cv2.waitKey(5)!=27:
            _, im = cap.read(); res = mp_infer(pose,im)
            draw.draw_landmarks(im, res.pose_landmarks, det.POSE_CONNECTIONS,
                landmark_drawing_spec=body, connection_drawing_spec=body)
            cv2.imshow('MediaPipe', im)
        cv2.destroyAllWindows()


########################################################
def mp_holistic(cap):
    det = mp.solutions.holistic; draw = mp.solutions.drawing_utils
    body = draw.DrawingSpec(color=(255,)*3, thickness=1, circle_radius=2)
    face = draw.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1)
    with det.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cv2.waitKey(5)!=27:
            _, im = cap.read(); res = mp_infer(pose,im)
            draw.draw_landmarks(im, res.face_landmarks, det.FACE_CONNECTIONS,
                landmark_drawing_spec=face, connection_drawing_spec=face)
            draw.draw_landmarks(im, res.pose_landmarks, det.POSE_CONNECTIONS,
                landmark_drawing_spec=body, connection_drawing_spec=body)
            cv2.imshow('MediaPipe', im)
        cv2.destroyAllWindows()


########################################################
def mp_objectron(cap, obj='Shoe'):
    det = mp.solutions.objectron; draw = mp.solutions.drawing_utils
    box = draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
    with det.Objectron(static_image_mode=False, model_name=obj,
        min_detection_confidence=0.5, min_tracking_confidence=0.99) as pose:
        while cv2.waitKey(5)!=27:
            _, im = cap.read(); res = mp_infer(pose,im)
            if res.detected_objects:
                for i in res.detected_objects:
                    draw.draw_landmarks(im, i.landmarks_2d, det.BOX_CONNECTIONS,
                        landmark_drawing_spec=box, connection_drawing_spec=box)
                    draw.draw_axis(im, i.rotation, i.translation)
            cv2.imshow('MediaPipe', im)
        cv2.destroyAllWindows()


########################################################
def mp_infer(mpp, im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Mark image writeable=False: improve performance
    im.flags.writeable = 0; return mpp.process(im)


CLS = [str(i) for i in range(10)]+['o','t','l','b']
##########################################################################################
def hand_data(cap, dst='Hand'):
    PS = {k:[] for k in CLS}; k = K = ''
    wh = np.array([cap.get(3),cap.get(4)])
    det = mp.solutions.hands; draw = mp.solutions.drawing_utils
    hand = draw.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=3)
    with det.Hands(max_num_hands=1, min_detection_confidence=0.75) as pose:
        while k!=27: # dir(i.landmark)
            _, im = cap.read(); k = cv2.waitKey(5)
            im = cv2.flip(im,1); res = mp_infer(pose,im)
            if res.multi_hand_landmarks: # find hand
                i = res.multi_hand_landmarks[0] # normalize
                p = [(m.x,m.y) for m in i.landmark]*wh/wh[1]
                if k>0 and chr(k) in CLS: K = chr(k) # trigger
                if K in CLS: PS[K].append(p-p[9]) # localize/centralize
                draw.draw_landmarks(im, i, det.HAND_CONNECTIONS, hand, hand)
            if k==32: K = '' # pause
            cv2.putText(im, K, (5,20), 4, 0.7, (255,)*3)
            cv2.imshow(dst, im)
        cv2.destroyAllWindows(); save_data(PS, dst)


def save_data(ps, dst='Hand'):
    if os.path.isfile(dst+'.npz'):
        P = np.load(dst+'.npz') # load
        for k in set(ps)|set(P): # merge
            if k not in P or len(P[k])<1: continue
            elif k in set(P)-set(ps): # k in P and k not in ps
                d = input(f'[omit=Enter, add=Blank, map={set(ps)}] for {set(k)}: ')
                if d in ps: ps[d] = np.concatenate([ps[d],P[k]]) if len(ps[d])>0 else P[k]
                elif d==' ': ps[k] = P[k]
            else: ps[k] = np.concatenate([ps[k],P[k]]) if len(ps[k])>0 else P[k]
    np.savez(dst, **ps); print({k:len(ps[k]) for k in ps}) # save


import torch
from torch.utils.data import Dataset, DataLoader
##########################################################################################
class HandDataset(Dataset):
    def __init__(self, src='Hand', s=1E-5):
        P = np.load(src+'.npz'); self.data = []
        for i,(k,v) in enumerate(P.items()):
            self.data += [(i,p.astype('float32')) for p in v]
        # x_transform: flip + scale, keypoint jitter with magnitude s/2
        #self.x_transform = lambda x: torch.ravel(torch.randn(2)*(x+(torch.rand(x.shape)-0.5)*s))
        self.x_transform = lambda x: torch.ravel(torch.randn(2)*(1+(torch.rand(x.shape)-0.5)*s)*x)
        self.y_transform = None # lambda x: torch.tensor(x)
        self.cls = [k for k in P]; print(self.cls)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        y, x = self.data[idx]
        if self.x_transform: x = self.x_transform(x)
        if self.y_transform: y = self.y_transform(y)
        return x, y # y=label, x=key_points (u,v)


import torch.nn as nn
import torch.nn.functional as F
##########################################################################################
class hswish(nn.Module):
    def forward(self, x): return x*F.relu6(x+3, inplace=True)/6


class Net(nn.Module):
    def __init__(self, hid=[128,128], cls=CLS):
        super(Net, self).__init__()
        block = []; hid = [21*2, *hid]
        for i in range(len(hid)-1):
            block.append(nn.Linear(hid[i], hid[i+1]))
            block.append(nn.BatchNorm1d(hid[i+1]))
            block.append(hswish()) # nn.ReLU6(inplace=True)
        classifier = [nn.Dropout(0.5), nn.Linear(hid[-1], len(CLS))]
        self.net = nn.Sequential(*block, *classifier)
        init_params(self.modules())

    def forward(self, x): return self.net(x)
        #with autocast(): return self.net(x)


from torch.nn import init
########################################################
def init_params(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None: init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.001)
            if m.bias is not None: init.constant_(m.bias, 0)
        elif type(m) in (nn.BatchNorm2d, nn.BatchNorm1d):
            init.constant_(m.weight, 1); init.constant_(m.bias, 0)


##########################################################################################
def load_save(net, don, dst='last.pt'):
    if type(don)==str and os.path.isfile(dst):
        x = torch.load(dst, map_location=don) # load
        don = x.pop('names') if 'names' in x else None
        net.load_state_dict(x); return don
    elif type(don) in (list,tuple): # save
        x = net.state_dict(); x['names'] = don
        torch.save(x, dst) # append names
    elif type(don)==dict: # utils: load->save
        x = torch.load(dst) if os.path.isfile(dst) else net.state_dict()
        x.update(don); torch.save(x, dst) # append dict


from torch import optim
from torch.cuda.amp import autocast, GradScaler
########################################################
def hand_train(src, epoch=80000, eps=1E-8):
    print(f'Hand: {torch.cuda.get_device_name()}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net().to(device); net.train(); load_save(net,device,'last.pt')
    data = HandDataset(src); names = data.cls; gs = GradScaler(); m = 1E3
    data = DataLoader(data, batch_size=1024, shuffle=True, num_workers=8)
    Loss = nn.CrossEntropyLoss()

    opt = optim.SGD(net.parameters(), lr=1E-3, momentum=0.9,
        weight_decay=1E-4, nesterov=True)
    #opt = optim.AdamW(net.parameters(), weight_decay=1E-3)
    PlateauLR = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
        factor=0.2, patience=50, cooldown=50, min_lr=1E-8, verbose=1)
    #CyclicLR = optim.lr_scheduler.CyclicLR(opt, base_lr=1E-6, max_lr=0.1)
    CycleLR1 = optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01,
        epochs=epoch, steps_per_epoch=len(data))
    try:
        for i in range(1, epoch+1):
            avg = 0 # avg_loss
            for j,(x,y) in enumerate(data):
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True )# zero gradient
                with autocast(): L = Loss(net(x),y) # forward
                gs.scale(L).backward(); gs.step(opt); gs.update()
                #L = Loss(net(x),y); L.backward(); opt.step()
                CycleLR1.step(); #CyclicLR.step()
                avg += L.item()/len(y)
            avg /= len(data); PlateauLR.step(metrics=avg)
            print('epoch=%3d, loss=%.5e'%(i+1, avg))
            if avg<=m: m = avg; load_save(net, names, 'best.pt')
            if avg<eps: break
    finally: load_save(net, names, 'last.pt')


########################################################
@torch.no_grad()
def hand_infer(cap):
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net().to(device); net.eval() # load weights
    names = load_save(net, device, 'last.pt')

    wh = np.array([cap.get(3),cap.get(4)]); k = K = ''
    det = mp.solutions.hands; draw = mp.solutions.drawing_utils
    hand = draw.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=3)
    with det.Hands(max_num_hands=1, min_detection_confidence=0.75) as pose:
        while k!=27: # dir(i.landmark)
            _, im = cap.read(); k = cv2.waitKey(5)
            im = cv2.flip(im,1); res = mp_infer(pose,im)
            if res.multi_hand_landmarks: # find hand
                i = res.multi_hand_landmarks[0]
                p = [(m.x,m.y) for m in i.landmark]*wh
                x = ((p-p[9])/wh[1]).astype('float32')
                x = torch.tensor(x).ravel()[None].to(device)
                K = names[net(x)[0].argmax().item()]
                draw.draw_landmarks(im, i, det.HAND_CONNECTIONS, hand, hand)
            else: K = ''
            cv2.putText(im, K, (5,20), 4, 0.7, (255,)*3)
            cv2.imshow('Hand', im)
        cv2.destroyAllWindows()


from numpy.linalg import norm
from itertools import combinations as cmb
INT = lambda x,s=1: tuple(int(i*s) for i in x)
MOD = {'m':'Move', 'p':'Pose', 'c':'Chasis', 'g':'Grip'}
##########################################################################################
# TODO: check status switch or not for once action
def flush(cap, n=8): # flush cam
    for i in range(n): cap.grab()


def get_xyz(im, p, p0, r0, K, S=4, d=4, u=200):
    pt = p[9]; rd = norm(pt-p[13]); r0 += d
    dx = np.log(1 if r0-d<rd<r0 else rd/r0)
    yz = pt-(p0 if norm(pt-p0)>r0 else pt)
    xyz = np.array([u*dx, *(-yz)])*S/20
    if K=='p': xyz /= 8 # Arm_Base frame

    p0 = INT(p0); color_st = (255,)*3
    pos = 'xyz=(%+3.2f %+3.2f %+3.2f)' % (*xyz,)
    cv2.putText(im, pos, (150,20), 4, 0.7, color_st)
    cv2.circle(im, p0, int(r0), (166,)*3, -1, cv2.LINE_AA)
    cv2.circle(im, p0, int(r0-d), (88,)*3, -1, cv2.LINE_AA)
    cv2.circle(im, p0, int(rd), color_st, 1, cv2.LINE_AA)
    cv2.arrowedLine(im, p0, INT(pt), color_st, 1, cv2.LINE_AA)
    return Vector3(*xyz) # radius=x, pt=(y,z)


@torch.no_grad()
def hand_control(cap):
    if os.path.isfile('last.pt'):
        torch.backends.cudnn.benchmark = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = Net().to(device); net.eval(); load_save(net,device,'last.pt')
        i2mod = {1:'s', 2:'f', 3:'c', 4:'p', 5:'m', 9:'g', 10:'h'}

    wh = np.array([cap.get(3),cap.get(4)]); G,S,K,k = 1,4,'',0
    det = mp.solutions.hands; draw = mp.solutions.drawing_utils
    hand = draw.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=3)
    with det.Hands(max_num_hands=1, min_detection_confidence=0.75) as pose:
        while k!=27: # dir(i.landmark)
            _, im = cap.read(); k = cv2.waitKey(5)
            if 48<k<58: S = k-48 # speed=1-9
            elif k==32: GO(Twist()) # Blank=pause
            elif k==13: HM(Int8()); rospy.sleep(2) # Enter

            im = cv2.flip(im,1); res = mp_infer(pose,im)
            if res.multi_hand_landmarks: # find hand
                i = res.multi_hand_landmarks[0]
                p = [(m.x,m.y) for m in i.landmark]*wh

                K = chr(k) if k>0 else K # key->mode
                if k>0 and chr(k) in 'mpc': # trigger
                    p0, r0 = p[9], norm(p[9]-p[13])
                elif 'net' in dir(): # hand->mode
                    x = ((p-p[9])/wh[1]).astype('float32')
                    x = torch.tensor(x).ravel()[None].to(device)
                    x = i2mod.get(net(x)[0].argmax().item(),'')
                    if x in 'mpc' and x!=K: # trigger
                        p0, r0 = p[9], norm(p[9]-p[13])
                    K = x # update mode. # distance -> grip
                elif sum(norm(d[1]-d[0]) for d in cmb(p[6:9],2))/3<norm(p[6]-p[5])/4: K = 'g'
                if K in 'mpc': xyz = get_xyz(im, p, p0, r0, K, S)
                draw.draw_landmarks(im, i, det.HAND_CONNECTIONS, hand, hand)
            else: K = '' # reset mode
            info = f'%s L={S}'%MOD.get(K,'OFF')
            cv2.putText(im, info, (5,20), 4, 0.7, (255,)*3)
            cv2.imshow('Hand', im)

            if K=='h': HM(Int8()); flush(cap) # standby
            elif K=='m': GO(Twist(linear=xyz)) # arm_move
            elif K=='p': GO(Twist(angular=xyz)) # arm_pose
            elif K=='g': G = 1-G; GP(Int8(G)); flush(cap) # grip
            elif K=='f' and S<9: S += 1; flush(cap) # fast
            elif K=='s' and S>1: S -= 1; flush(cap) # slow
            elif K=='c': pass # chasis
        cv2.destroyAllWindows(); HM(Int8())


import rospy
from std_msgs.msg import Int8
from geometry_msgs.msg import Twist, Vector3
##########################################################################################
if __name__ == '__main__':
    cap = cv2.VideoCapture(-1)
    os.chdir(os.path.dirname(__file__))
    #mp_face_det(cap); #mp_face_mesh(cap)
    #mp_hand_pose(cap); #mp_body_pose(cap)
    #mp_holistic(cap); #mp_objectron(cap)

    #hand_data(cap, dst='Hand')
    ##hand_train(src='Hand')
    #hand_infer(cap)

    rospy.init_node('hand_ctrl', anonymous=True)
    HM = rospy.Publisher('/xarm/go_ready', Int8, queue_size=1).publish
    MD = rospy.Publisher('/xarm/mode_switch', Int8, queue_size=1).publish
    GP = rospy.Publisher('/xarm/gripper_ctl', Int8, queue_size=1).publish
    GO = rospy.Publisher('/xarm/velocity_go', Twist, queue_size=1).publish
    hand_control(cap)#'''

