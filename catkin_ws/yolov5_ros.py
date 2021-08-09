#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, cv2, sys, json
sys.path.append('yolov5')
from yolov5.infer import yolov5
import time as tm

import rospy, roslib
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
# cv_bridge does not support CompressedImage in python
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, LaserScan
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer


tc = {'in use':(0,255,0), 'vacant maybe':(0,0,255), '':(0,255,255)}
cam_depth = {'RealSense':'/camera/aligned_depth_to_color/image_raw', 
    'Kinect':'/depth_to_rgb/image_raw', 'ZED':'/zed_node/depth/depth_registered'}
cam_color = {'RealSense':'/camera/color/image_raw/compressed', 'Kinect':'/rgb/image_raw/compressed', 
    'ZED':'/zed_node/left/image_rect_color/compressed', 'USB':'/usb_cam/image_raw/compressed'}
##########################################################################################
class ros_det:
    def __init__(self, model, dst, cls=None):
        '''Initialize ros publisher, ros subscriber'''
        self.YOLO = yolov5(model, cls=cls); self.dst = dst
        self.infer1 = lambda x: self.YOLO.infer1(x, True, True)
        self.results = {'520w':'', '523':'', '501':'', '502':'', '522':''}
        rospy.init_node('ros_det', anonymous=True); self.show = []

        img_rgb = TPC(cam_color.values()); #img_depth = TPC(cam_depth.values())
        self.out_det = rospy.Publisher('/output/det/compressed', CompressedImage, queue_size=10)

        rospy.Subscriber('/take_photo', String, self.callback_capt, queue_size=1)
        rospy.Subscriber(img_rgb, CompressedImage, self.callback_rgb, queue_size=1)

        '''img_depth = Subscriber(img_depth, Image, queue_size=1)
        img_rgb = Subscriber(img_rgb, CompressedImage, queue_size=1)
        sync = TimeSynchronizer([img_rgb, stat], queue_size=10)
        #sync = ApproximateTimeSynchronizer([img_rgb, img_depth, stat],
        #                queue_size=10, slop=1, allow_headerless=True)
        sync.registerCallback(self.callback_multi) #callback_test'''


    ########################################################
    def callback_rgb(self, rgb): self.rgb = rgb # store


    def callback_capt(self, stat):
        rgb = np.frombuffer(self.rgb.data, 'uint8')
        rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)

        ff = '%s/%s.jpg' % (self.dst, stat.data)
        im, _, dt, res = self.infer1(rgb)
        print('%s: %.2fms'%(ff,dt), res)
        
        key = str(stat.data).split()[0]
        if len(res)>0: self.results[key] = 'in use'
        if key not in self.results or self.results[key]!='in use':
            self.results[key] = 'vacant maybe'
        '''res = 'in use' if len(res)>0 else 'vacant maybe'
        if key not in self.results or res=='in use':
            self.results[key] = res'''
        
        res = self.results[key] # for short
        cv2.rectangle(im, (0,2), (330,35), (66,)*3, -1)
        cv2.putText(im, f'{key}: {res}', (2,26), 4, 0.8, tc[res])
        self.show.append(im); cv2.imwrite(ff, im)


    '''# cv2.imshow in Synchronizer leads to crash
    def callback_test(self, rgb, depth, stat):
        print('rgb:',rgb.header.stamp.to_sec())
        print('dep:',depth.header.stamp.to_sec())
        t0 = rospy.Time.now()
        dp = parse_depth(depth, 'U')
        rgb = np.frombuffer(rgb.data, 'uint8')
        rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
        print((rospy.Time.now()-t0).to_sec())
        #cv2.imshow('dp',dp); cv2.imshow('rgb',rgb)
        cv2.waitKey(5); print('end\n')


    def pub_det(self, det):
        msg = CompressedImage(format='jpeg')
        msg.data = np.asarray(cv2.imencode('.jpg',det)[1]).tobytes()
        msg.header.stamp = rospy.Time.now(); self.out_det.publish(msg)'''


##########################################################################################
def TPC(tp): # pick existed topic
    while True:
        tps = rospy.get_published_topics()
        tps = {t[0]:t[1] for t in tps}
        x = [i for i in tp if i in tps]
        if len(x)>0: return x[0]


'''def parse_depth(depth, mod='F'):
    # Filter nan/inf: (-inf, inf, nan)->0
    #np.seterr(divide='ignore', invalid='ignore')
    # right = CvBridge().imgmsg_to_cv2(rgb, 'bgr8')
    depth = CvBridge().imgmsg_to_cv2(depth, '32FC1')
    depth = np.where(abs(depth)<np.inf, depth, 0)
    if 'F' in mod: return depth # float, meters
    # np.astype('uint8') can also filter nan/inf
    return (depth/depth.max()*255).astype('uint8')'''


def show_standby(im, dic, h0=37):
    for i,(k,v) in enumerate(dic.items()):
        cv2.rectangle(im, (0,h0+i*30), (240,h0+i*30+28), (99,)*3, -1)
        cv2.putText(im, f'{k}: {v}', (2,h0+i*30+20), 4, 0.6, tc[v])


##########################################################################################
if __name__ == '__main__':
    root = 'yolov5/'; dst = 'test'
    os.makedirs(dst, exist_ok=True)
    det = ros_det(root+'yolov5l.pt', dst, [0,63])
    #try: rospy.spin() # 0=person, 63=laptop
    while not rospy.core.is_shutdown():
        im, dt = 0, 1
        if len(det.show)>0:
            im = det.show[0].copy()
            dt = 200; del det.show[0]
        elif hasattr(det, 'rgb'):
            im = np.frombuffer(det.rgb.data, 'uint8')
            im = cv2.imdecode(im, cv2.IMREAD_COLOR)
            k = [k for k,v in det.results.items() if v=='']
            info = f'Going to: {k[0]}' if len(k)>0 else 'Finish'
            cv2.rectangle(im, (0,2), (330,35), (66,)*3, -1)
            cv2.putText(im, info, (2,26), 4, 0.8, tc[''])#'''
        if type(im)==np.ndarray:
            show_standby(im, det.results)
            cv2.imshow('det', im); cv2.waitKey(dt)
    rec = tm.strftime('%Y-%m-%d %H:%M:%S', tm.localtime())
    with open(dst+f'/{rec}.json','w+') as js:
        json.dump(det.results, js, indent=4)

