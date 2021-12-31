#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, cv2, rospy
from time import time, sleep
from std_msgs.msg import String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
from threading import Thread, Event
from serial import Serial


MT, SW = [0], Event() # Thread trigger for block/wait
DIST = {480: [-0.367385, 0.098687, 0.003397, 0.003075, 0.0], # (640,480)
        600: [-0.369208, 0.105639, -0.001231, 0.001386, 0.0]} # (800,600)
CMTX = {480: [363.69313, 0.0, 311.56425, 0.0, 484.758071, 236.692749, 0.0, 0.0, 1.0],
        600: [445.813097, 0.0, 393.517346, 0.0, 595.943691, 309.187313, 0.0, 0.0, 1.0]}
##########################################################################################
def cam_test(mt=[0], fps=30, dt=1, cap=-1):
    if type(cap) in (int,str): cap = cv2.VideoCapture(cap)
    elif type(cap)!=cv2.VideoCapture: cap = cv2.VideoCapture(-1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800) # 3
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600) # 4
    cap.set(cv2.CAP_PROP_FPS, fps); fps = cap.get(5) # 5
    init_distort(cap); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 38
    if mt: Thread(target=Get, args=(cap,mt), daemon=True).start()
    print('camera_res:', cap.get(3), cap.get(4))

    while cv2.waitKey(5)!=27:
        if not mt: _, im = cap.read()
        elif type(mt)!=list: _, im = cap.retrieve()
        elif hasattr(mt[0],'copy'): im = mt[0].copy()
        else: SW.wait(); im = mt[0].copy()
        im = cv2.undistort(im, CMTX, DIST)
        sleep(dt) # simulate, must before imshow
        cv2.imshow(f'fps={fps}, mt={bool(mt)}', im)
    cv2.destroyAllWindows(); cap.release()


##########################################################################################
def init_distort(ht): # for cv2.undistort()
    global CMTX, DIST # cameraMatrix, distCoeffs
    ht = int(ht.get(4) if type(ht)==cv2.VideoCapture else ht)
    if type(CMTX)==dict and type(DIST)==dict:
        assert ht in CMTX and ht in DIST, f'{ht} not in CMTX/DIST'
        CMTX, DIST = np.array(CMTX[ht]).reshape(3,3), np.array(DIST[ht])


def Get(cap, mt=1): # Thread: RGB
    while cap.isOpened(): # VideoCapture
        if type(mt)==list and mt:
            SW.clear() # block/wait
            _, mt[0] = cap.read()
            if _: SW.set() # no block
        else: cap.grab() # clear buffer


def get_rgb(rgb, mt):
    if type(rgb)==cv2.VideoCapture:
        if not mt: _, im = rgb.read()
        elif type(mt)!=list: _, im = rgb.retrieve()
        elif hasattr(mt[0],'copy'): im = mt[0].copy()
        else: SW.wait(); im = mt[0].copy()
        im = cv2.undistort(im, CMTX, DIST)
    elif type(rgb)==CompressedImage:
        im = CVB.compressed_imgmsg_to_cv2(rgb)
    elif type(rgb)==np.ndarray: im = rgb.copy()
    elif type(rgb)==Image: im = CVB.imgmsg_to_cv2(rgb)
    return im if 'im' in dir() else None


RSZ = lambda im,s=1: cv2.resize(im, None, fx=s, fy=s)
##########################################################################################
def track_face(cap, face=None, base=None, mt=[0]):
    from det_face import det_face, MTCNN, box_face
    from det_face import load_net, load_base

    mtcnn = MTCNN(); ID = ''
    if None not in (face, base):
        face = load_net(face)
        base = load_base(base, face, mtcnn)
    det = os.popen('locate haarcascade_frontalface').read()
    if not det: det = os.popen('find GLDFace/*.xml').read()
    if det: det = cv2.CascadeClassifier(det.strip().split()[-1])
    if mt: Thread(target=Get, args=(cap,mt), daemon=True).start()

    while cv2.waitKey(5)!=27:
        im = get_rgb(cap, mt); t = time()
        if type(im)!=np.ndarray: continue

        ct = np.asarray(im.shape[1::-1])//2
        if None in (face, base): # detect only
            # detect only: OpenCV (x,y,w,h)
            '''gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            bx = det.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for i in bx: i[2:] += i[:2] # (x,y,w,h)->(x,y,x2,y2)
            info = range(len(bx)); bx = None if len(bx)<1 else bx #'''
            # detect only: MTCNN (x,y,x2,y2)
            bx, info = mtcnn.detect(im, landmarks=False)
            if bx is not None:
                im = box_face(im, bx, info, (time()-t)*1000)
                bx = max(bx, key=lambda x: np.prod(x[2:]-x[:2]))
                bx = np.round((bx[:2]+bx[2:])/2).astype(int)
                cv2.circle(im, (bx[0],bx[1]), 1, (255,0,0), 2)
                pub_pt.publish(Point(*(bx-ct),0)) # publish Point
        else: # recognize: MTCNN + FaceNet
            res = det_face(im, mtcnn, face, base)
            if len(res)>2:
                im, res, bx = res[:3]; res = list(zip(*res))[0]
                ID = ID if ID in res else res[0]; bx = bx[res.index(ID)]
                bx = np.round((bx[:2]+bx[2:])/2).astype(int)
                cv2.circle(im, (bx[0],bx[1]), 1, (255,0,0), 2)
                pub_pt.publish(Point(*(bx-ct),0)) # publish Point
        pub_im.publish(CVB.cv2_to_compressed_imgmsg(im))
        cv2.imshow('det', RSZ(im, 720/im.shape[0]))
    cv2.destroyAllWindows()


##########################################################################################
def GetZ(son, z): # Thread: Sonic(cm)
    while True: z[0] = float(son.readline()[:-4])


def track_win(cap, mt=[0], son=None):
    from det_win import det_win, mark_win; z = [3]
    if mt: Thread(target=Get, args=(cap,mt), daemon=True).start()
    if son: Thread(target=GetZ, args=(son,z), daemon=True).start()

    while cv2.waitKey(5)!=27:
        im = get_rgb(cap, mt)
        if type(im)!=np.ndarray: continue
        x,y = det_win(im,1); mark_win(im,x,y,z[0])
        cv2.imshow('det', RSZ(im, 720/im.shape[0]))
        if str in [type(x),type(y)]: continue
        ct = np.asarray(im.shape[1::-1])//2
        ct = np.asarray([sum(x)//2,y])-ct
        pub_pt.publish(Point(*ct, 1 if 3<z[0]<50 else 0))
        pub_im.publish(CVB.cv2_to_compressed_imgmsg(im))
    cv2.destroyAllWindows()


##########################################################################################
if __name__ == '__main__':
    Cap = 'http://192.168.10.126:8080/stream?topic=/arm/image'
    Cap = cv2.VideoCapture(-1); CVB = CvBridge() # '/dev/video0'

    rospy.init_node('arm_track', anonymous=True)
    home = rospy.Publisher('/arm/go/home', String, queue_size=1)
    pub_pt = rospy.Publisher('/arm/track/point', Point, queue_size=1)
    pub_im = rospy.Publisher('/arm/track/compressed', CompressedImage, queue_size=1)
    #def Cache(rgb): global Cap; Cap = rgb # callback: alter Cap->cache CompressedImage
    #sub = rospy.Subscriber('/arm/image/compressed', CompressedImage, Cache, queue_size=1)
    #Son = Serial('/dev/ttyUSB0'); os.system('sudo chmod 777 %s'%Son.port)
    init_distort(Cap); print('camera_res:', Cap.get(3), Cap.get(4))

    home.publish(String()); rospy.sleep(2); #track_face(Cap)
    #track_face(Cap, 'yolov5/vggface2.pt','GLDFace')
    home.publish(String()); rospy.sleep(2); track_win(Cap)
    home.publish(String()); rospy.sleep(2)

