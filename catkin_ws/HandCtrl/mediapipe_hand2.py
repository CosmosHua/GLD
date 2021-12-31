#!/usr/bin/python3
# coding: utf-8

import os, cv2
import numpy as np
import mediapipe as mp


##########################################################################################
def mp_infer(mpp, im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im.flags.writeable = 0; return mpp.process(im)


import torch
import torch.nn as nn
import torch.nn.functional as F
CLS = [str(i) for i in range(10)]+['o','t','l','b']
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
        #init_params(self.modules())

    def forward(self, x): return self.net(x)
        #with autocast(): return self.net(x)


##########################################################################################
def load_save(net, don, dst='last.pt'):
    if type(don)==str and os.path.isfile(dst):
        x = torch.load(dst, map_location=don) # load
        don = x.pop('names') if 'names' in x else None
        net.load_state_dict(x); return don
    elif type(don) in (list, tuple): # save
        x = net.state_dict(); x['names'] = don
        torch.save(x, dst) # append names
    elif type(don)==dict: # utils: load->save
        x = torch.load(dst) if os.path.isfile(dst) else net.state_dict()
        x.update(don); torch.save(x, dst) # append dict


##########################################################################################
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
                    K = x # update mode
                elif sum(norm(d[1]-d[0]) for d in cmb(p[6:9],2))\
                    /3<norm(p[6]-p[5])/4: K = 'g' # dist->grip'''
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
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cap = cv2.VideoCapture(-1); hand_infer(cap)
    '''rospy.init_node('hand_ctrl', anonymous=True)
    HM = rospy.Publisher('/xarm/go_ready', Int8, queue_size=1).publish
    MD = rospy.Publisher('/xarm/mode_switch', Int8, queue_size=1).publish
    GP = rospy.Publisher('/xarm/gripper_ctl', Int8, queue_size=1).publish
    GO = rospy.Publisher('/xarm/velocity_go', Twist, queue_size=1).publish
    hand_control(cap)#'''

