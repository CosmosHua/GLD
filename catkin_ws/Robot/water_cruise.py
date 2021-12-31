#!/usr/bin/python3
# coding: utf-8

import matplotlib, gi
matplotlib.use('Agg') #'TkAgg'
#gi.require_version('Gtk','2.0')

import time as tm
import numpy as np
import json, rospy
import os, cv2, sys
sys.path.append('yolov5')

from water import *
from baidu_speech import *
from cam_rs import rs_stop_devices
from cam_rs import rs_init, rs_rgbd
from det_face import det_face, MTCNN
from det_face import load_net, load_base
from det_win import det_win, mark_win
from yolov5.infer import yolov5
from threading import Thread, Event
from geometry_msgs.msg import Point
from std_msgs.msg import String


########################################################
NAME = {'DiaoZhiZhong':'刁总', 'LiuQian':'刘谦总', 'WangAiHua':'爱华总',
        'YuanZhengGang':'袁总', 'HePing':'何平总', 'YunLangSheng':'云总',
        'ZhiFei':'只飞总', 'WangShaoShan':'少山总', 'LiShuJian':'树剑总',
        'LiBin':'李宾总', 'BaoSong':'鲍松', 'JiangHui':'姜老师', 'KangXingHao':'兴豪',
        'ChenYanTong':'彦彤', 'MiaoYan':'缪琰', 'SongYongBo':'永博', 'Joshua':'付华'}
SAY = { 1: '检测到有人在使用%s', 0: '可能无人使用%s', -1: '尚未检查%s',
        'start': '开始巡检第%s层的会议室', 'goto': '即将前往%s会议室',
        'finish': '第%s层的会议室巡检结束', 'home': '是否回到充电桩',
        'block': '哎呀，被挡住啦，请让一下', 'abort': '是否中止巡检',
        'arrive': '小小爱已顺利到达%s', 'closed': '小小爱无法进门',
        'greet': '嗨，%s，%s好，祝您工作愉快'}


MXS = {'max_speed_linear':1.2, 'max_speed_angular':3.0, 'max_speed_ratio':1.0}
TMH = lambda x: '上午' if x<11.5 else '中午' if x<13 else '下午' if x<18 else '晚上'
##########################################################################################
def init_greet(cf=5):
    for i in {TMH(h) for h in range(24)}:
        for n in NAME.values(): speak_(SAY['greet']%(n,i))
        tm.sleep(3) # wait for speak_()
    for i in get_marker(TCP).keys():
        if not (i.isdigit() and i.startswith(str(cf))): continue
        for k in ['goto','arrive',1,0,-1]: speak_(SAY[k]%' '.join(list(i)))
        tm.sleep(3) # wait for speak_()


def say_hello(face, th=0.7, s=0.1): # greet
    name = [NAME[k] for k,v in face if k in NAME and v<th]
    if len(name)<1: return # slow down->salute

    mxs = tcp_cmd(TCP,'get_params')['results']
    for k,v in MXS.items(): # reduce max_speed
        if k in mxs: v = MXS[k] = mxs[k]
        tcp_cmd(TCP,'set_params?%s=%.3f'%(k,v*s))

    t = tm.localtime(); t = t.tm_hour+t.tm_min/60
    greet = SAY['greet'] % ('，'.join(name), TMH(t))
    speak(greet); print('\tSalute:', greet)

    for k,v in MXS.items(): # restore max_speed
        tcp_cmd(TCP,'set_params?%s=%.3f'%(k,v))


MT, SW = [0], Event() # Thread trigger for block/wait
RSZ = lambda im,s=1: cv2.resize(im, None, fx=s, fy=s)
DIST = {480: [-0.367385, 0.098687, 0.003397, 0.003075, 0.0], # (640,480)
        600: [-0.369208, 0.105639, -0.001231, 0.001386, 0.0]} # (800,600)
CMTX = {480: [363.69313, 0.0, 311.56425, 0.0, 484.758071, 236.692749, 0.0, 0.0, 1.0],
        600: [445.813097, 0.0, 393.517346, 0.0, 595.943691, 309.187313, 0.0, 0.0, 1.0]}
##########################################################################################
def init_distort(ht): # for cv2.undistort()
    global CMTX, DIST # cameraMatrix, distCoeffs
    ht = int(ht.get(4) if type(ht)==cv2.VideoCapture else ht)
    if type(CMTX)==dict and type(DIST)==dict:
        assert ht in CMTX and ht in DIST, f'{ht} not in CMTX/DIST'
        CMTX, DIST = np.array(CMTX[ht]).reshape(3,3), np.array(DIST[ht])


def init_cam(wh=(640,480), cap=-1):
    cam = rs_init()[1] # dict: RealSense
    if not cam: # UBS_Cam -> VideoCapture
        cam = cv2.VideoCapture(cap)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, wh[0]) # 3
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, wh[1]) # 4
        #cam.set(cv2.CAP_PROP_FPS, fps); fps = cam.get(5) # 5
        init_distort(cam); cam.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 38
        if MT: Thread(target=Get, args=(cam,MT), daemon=True).start()
        print('camera_res:', cam.get(3), cam.get(4))
    return cam # dir()/vars()/locals()


def Get(cap, MT=1): # Thread callback
    while cap.isOpened(): # VideoCapture
        if type(MT)==list and MT:
            SW.clear() # block/wait
            _, MT[0] = cap.read()
            if _: SW.set() # no block
        else: cap.grab() # clear buffer


def get_rgb(cam, dev=0):
    if type(cam)==cv2.VideoCapture:
        if not MT: _, im = cam.read() # no Thread
        elif type(MT)!=list: _, im = cam.retrieve()
        elif hasattr(MT[0],'copy'): im = MT[0].copy()
        else: SW.wait(); im = MT[0].copy()
        im = cv2.undistort(im, CMTX, DIST)
    elif type(cam)==dict: # RSCam: 0=body, -1=face
        im = rs_rgbd(list(cam.values())[dev])['Color']
    if 'im' in dir(): return RSZ(im, 720/im.shape[0])


Date = lambda: tm.strftime('%Y-%m-%d-%H:%M:%S', tm.localtime())
##########################################################################################
def det_body(im, dst=None): # save
    im, _, dt, res = YOLO.infer1(im, True, True)
    print('\t%s: %.2fms'%(dst, dt), res) # infer1
    if type(dst)!=str: return im, res, dt # res=dict

    tc = (0,255,255) if len(res)>0 else (255,)*3
    ret = LB[len(res)>0]; msg = dst[:-4]+': '+ret
    cv2.rectangle(im, (0,0), (len(msg)*12,24), (88,)*3, -1)
    cv2.putText(im, msg, (2,18), 4, 0.6, tc) # ret=str
    cv2.imwrite(DST+'/'+dst, im); return im, ret, res, dt


def det_show(im, js={}, mk=''):
    if js and mk: # det_body
        dst = f'{Date()} {mk}.jpg'
        im, js[dst] = det_body(im, dst)[:2]
    elif mk==-1: # det_face->recognize
        im, res = det_face(im, mtcnn, FACE, BASE)[:2]
        if type(res)==list: say_hello(res)
    cv2.imshow('det',im); cv2.moveWindow('det',0,0)
    return cv2.waitKey(5) # default=-1


##########################################################################################
def Abort(k):
    if k==27: # emergency->abort
        tcp_cmd(TCP, 'estop?flag=true')
        cv2.destroyAllWindows(); speak_(SAY['abort'])
        k = 27 if 'y' in input('Abort?[y/n]: ') else 1
        if k==27: tcp_cmd(TCP, 'move/cancel')
        tcp_cmd(TCP, 'estop?flag=false')
    return k # k, 27=abort, 1=continue


def Status(all=False):
    st = tcp_filter(TCP, False, bt=0.2, n=1)
    st, nt = st['callback'], st['notification']
    nt = nt if nt else {'code':'', 'description':''}
    st = st['results'] if st else {'move_status':''}
    #if 'move_retry_times' in st and st['move_retry_times']:
    #    print(f"move_retry_times: {st['move_retry_times']}")
    #if st['move_status']: print('\tstatus:', st['move_status'])
    if nt['code']: print('\t%s:'%nt['code'], nt['description'])
    if not all: return nt['code'], st['move_status'] # specific
    else: return nt, st # notification & robot_status['results']


AZ = 35; dT = AZ/10 # Arm_to_Win: 10Hz
##########################################################################################
def Arm_Standby(N=5, dt=2):
    for i in range(N): Home.publish(String())
    print('=> Arm_Backward'); tm.sleep(dt)


def Arm_to_Win(dt=2, z=AZ): # fps=10Hz
    if det_win(get_rgb(Cam,0))=='no': return 0
    print('=> Arm_Forward to window')
    for i in range(AZ): # det_win->adjust
        im = get_rgb(Cam, dev=0) # 0=body
        x,y = det_win(im); mark_win(im,x,y,i)
        if str in [type(x),type(y)]: continue
        #cv2.imshow('det',im); cv2.waitKey(5)
        if det_show(im)==27: return 27
        ct = np.asarray(im.shape[1::-1])//2
        ct = np.asarray([sum(x)//2,y])-ct
        Pub.publish(Point(*ct,i<z)) # forward
    tm.sleep(dt) # wait Arm not vibrant


AF, DF = 0, 0.22 # Camera Angle/Distance offset: Dofbot_Arm
# Foldable=(0, 0.38), Magnetic=(0, 0.25), Pillar=(-np.pi/2, 0.2)
SF = '&distance_tolerance=0.05&theta_tolerance=0.01&max_continuous_retries=5'
MV = lambda m,a: f'move?marker={m}&angle_offset={a}'+SF
# joy_control: max_angular=1.0rad/s, max_linear=0.5m/s, duration=0.5s
JC = lambda a,d: 'joy_control?angular_velocity=%.4f&linear_velocity=%.4f'%(a,d)
##########################################################################################
# return: 27=abort, 1=continue, -1=default; 0=fail
def inspect_(cmd, js={}, mk=''): # (0,27,1,-1)
    tcp_cmd(TCP,'estop?flag=false'); Arm_Standby()
    tcp_cmd(TCP,cmd); par = cmd.split('?') # len>1
    par = dict(i.split('=') for i in par[1].split('&'))

    if not mk and 'marker' in par: mk = par['marker']
    ofs = [par[i] for i in par if 'angle_offset' in i]
    if ofs: ofs = abs(float(ofs[0])-AF)<0.05 # 1,0,[]

    first = mk.isdigit() and ofs # go->arrive
    dev = -int(first or ofs==[]) # 0=body, -1=face
    #if first: speak(SAY['goto']%' '.join(list(mk)))
    while True: # det_face or not -> det_body
        k = det_show(get_rgb(Cam,dev), mk=dev)
        if Abort(k)==27: return 27 # det_face/show
        code, st = Status() # get code & move_status
        if code=='01200': speak(SAY['block'])
        if code=='01203' and 'I' in mk: speak_(SAY['closed'])
        if st=='failed' or code in ('01003','01203'): #'01005'
            tcp_cmd(TCP,'move/cancel'); return 0 # 0=fail
        if st=='succeeded' or code=='01002': # detect body
            if first: speak_(SAY['arrive']%' '.join(list(mk)))
            if 'W' in mk: st = Arm_to_Win() # Arm_Forward
            k = det_show(get_rgb(Cam,0), js, mk) # 0=body
            if 'W' in mk: Arm_Standby() # Arm_Backward
            return Abort(k) if st!=0 else 0 # k=(-1,27,1)


########################################################
def inspect_w(js, mk, a=0, N=10):
    d = DF*(1-np.cos(a)) # water advance
    N = int(max(abs(a/0.5), abs(d/0.25), N))
    # max of each JC: angle=0.5rad, dis=0.25m
    Thread(target=Arm_to_Win, daemon=True).start()
    cmd = JC(2*a/N, 2*d/N) # dT: Arm forward duration
    for i in range(N): tcp_cmd(TCP,cmd); tm.sleep(dT/N)
    tm.sleep(1); k = det_show(get_rgb(Cam,0), js, mk)
    cmd = JC(-2*a/N, -2*d/N) # det_body->backword
    Thread(target=Arm_Standby, daemon=True).start()
    for i in range(N): tcp_cmd(TCP,cmd); tm.sleep(dT/N)
    return Abort(k) # k=(-1,27,1)


##########################################################################################
def CamRot(mk, ao=0): # require AF & DF
    p = format_pos(MARK[mk]['pose']); t = p['theta']
    cx, cy = p['x']+np.cos(t)*DF, p['y']+np.sin(t)*DF
    rt, ao = (t+ao)+np.pi, (t+ao)+AF # reverse/offset
    x, y = cx+DF*np.cos(rt), cy+DF*np.sin(rt)
    return 'move?location=%.3f,%.3f,%.7f'%(x,y,ao)+SF


def Inspect(mk, js, AG=(-30,30)): # AG: degree
    AG = np.asarray(sorted(AG))*np.pi/180
    # only wood_door has indoor/window views
    if mk+'W' in MARK or 'G' in mk: # look once
        return inspect_(MV(mk,AF), js, mk)
    elif 'W' in mk: # window of wood_door
        k = inspect_(MV(mk,AF), js, mk)
        if k in (0,27) or LB[1] in js.values(): return k
        for a in AG*2: # rotate around camera
            #k = inspect_(CamRot(mk,a), js, mk)
            k = inspect_w(js, mk, a) # joy_control
            if k in (0,27) or LB[1] in js.values(): return k
    else: # indoor/glass door
        for a in np.append([0],AG): # rotate robot
            k = inspect_(MV(mk,a+AF), js, mk)
            if k in (0,27) or LB[1] in js.values(): return k
    return k # 27=abort, 1=continue, -1=default, 0=fail


LB = {1:'in use', 0:'vacant maybe', -1:'-'}; DST = '.'
LK = lambda x: 1 if LB[1] in x else 0 if LB[0] in x else -1
LK = lambda x: max([k for k,v in LB.items() if v in x], default=-1)
##########################################################################################
def water_cruise(cf=5, mk=''):
    im = det_body(get_rgb(Cam,0))[0]
    cv2.imshow('det',im); cv2.waitKey(5)

    global DST; DST = Date()[:10]+f'_F{cf}'
    os.makedirs(DST, exist_ok=True); rst = f'{DST}/{DST}.json'
    dT = Date()[:-3]; JS = {dT:{}} # init->reverse order
    if os.path.isfile(rst): # load history->reserve
        with open(rst,'r+') as f: JS.update(json.load(f))

    tcp_cmd(TCP,'request_data?topic=robot_status') # 2Hz
    target = target_layout(TCP,cf); speak(SAY['start']%cf)
    if mk in target and mk.startswith(str(cf)):
        target = {k:v for k,v in target.items() if k==mk}
    for out, giw in target.items(): # giw=dict(G,I,W)
        speak(SAY['goto']%' '.join(list(out)))
        js = JS[dT][out] = {'result': LB[-1]} # init
        k = Inspect(out, js) # outdoor
        for x in giw['G']: # loop: glass_walls
            if LB[1] in js.values() or k==27: break
            k = Inspect(x, js) # update js/JS
        for x in giw['I']: # loop: indoors, fail->skip
            if LB[1] in js.values() or k in (0,27): break
            k = Inspect(x, js) # update js/JS
        for x in giw['W']: # loop: windows
            if LB[1] in js.values() or k==27: break
            k = Inspect(x, js) # update js/JS
        js['result'] = LB[LK(js.values())] # update JS
        speak(SAY[LK(js.values())]%' '.join(list(out)))
        if k==27: break # emergency->abort->estop
    with open(rst,'w+') as f: json.dump(JS,f,indent=4)
    tcp_cmd(TCP,'estop?flag=true'); cv2.destroyAllWindows()

    #Speak_([SAY['finish']%cf, SAY['home']])
    back = input('Task done! Back home?[y/n]: ')
    back = back if back in MARK else 'home'*(back=='y')
    if back: inspect_('move?marker=%s'%back) # charging
    tcp_cmd(TCP,'estop?flag=true'); cv2.destroyAllWindows()


##########################################################################################
if __name__ == '__main__':
    rospy.init_node('arm_adjust', anonymous=True)
    Home = rospy.Publisher('/arm/go/home', String, queue_size=5)
    Pub = rospy.Publisher('/arm/track/point', Point, queue_size=1)

    TCP = tcp_init(); Cam = init_cam((800,600)) # use MT
    YOLO = yolov5('yolov5/yolov5l.pt', cls=[0]) # person
    FACE, mtcnn = load_net('yolov5/vggface2.pt'), MTCNN()
    BASE = load_base('GLDFace', FACE, mtcnn); k = 0
    MARK = tcp_cmd(TCP,'markers/query_list')['results']
    while k!='n': # yolo: 0=person, 63=laptop
        water_cruise(); k = input('\nAgain?[y/n]: ')
    if type(Cam)==cv2.VideoCapture: Cam.release()
    elif type(Cam)==dict: rs_stop_devices(Cam)
    TCP.close() # not necessary

