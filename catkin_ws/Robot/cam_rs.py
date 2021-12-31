#!/usr/bin/python3
# coding: utf-8

import cv2, sys
import numpy as np
import pyrealsense2 as rs


# Multi-Camera: wrappers/python/examples/box_dimensioner_multicam/realsense_device_manager.py
##########################################################################################
# Get SN of all connected Intel RealSense Devices
def rs_get_devices(show=False):
    devices = [] # available_devices
    for d in rs.context().devices:
        if d.get_info(rs.camera_info.name).lower()!='platform camera':
            devices.append(d.get_info(rs.camera_info.serial_number))
            if show: print(d) # device: type & S/N
    return devices # list: serial number


# Enable all connected Intel RealSense Devices
def rs_start_devices(config, IR=False, show=True):
    devices = {} # enabled_devices
    for serial in rs_get_devices(show):
        # Enable the device
        pipe = rs.pipeline() # create pipeline
        config.enable_device(serial) # enable device
        profile = pipe.start(config) # start streaming
        # Enable the IR_Emitter of device
        sensor = profile.get_device().first_depth_sensor()
        sensor.set_option(rs.option.emitter_enabled, 1 if IR else 0)
        if IR: sensor.set_option(rs.option.laser_power, 330)
        devices[serial] = (pipe, profile) # add enabled_devices
    return devices # dict: {serial:(pipe,profile)}


def rs_stop_devices(devices):
    for (pipe, profile) in devices.values(): pipe.stop()


# Load from json of settings readable by the RealSense
def rs_load_json(devices, file):
    with open(file, 'r') as f: ff = f.read().strip()
    for (pipe, profile) in devices.values():
        advanced_mode = rs.rs400_advanced_mode(profile.get_device())
        advanced_mode.load_json(ff)


rs_stop_streams = lambda cfg: cfg.disable_all_streams()
##########################################################################################
# Poll frames from all connected Intel RealSense Devices
# This will return at least one frame from each device.
def rs_poll_frames(devices, align=True):
    result = {} # align to color_stream
    if align: align = rs.align(rs.stream.color)
    while len(result)<len(devices):
        for (serial, (pipe, profile)) in devices.items():
            #frames = pipe.poll_for_frames() # without block
            #if len(frames)!=len(profile.get_streams()): continue
            frames = pipe.wait_for_frames() # block-function
            if type(align)==rs.align: frames = align.process(frames)
            result[serial] = rs_parse_frames(frames, profile) # dict
    return result # {'Color','Depth','Infrared 1','Infrared 2'}


# Get frames from single connected Intel RealSense Device
def rs_get_frames(pipe, profile, align=True):
    assert type(pipe)==rs.pipeline
    assert type(profile)==rs.pipeline_profile
    if align and type(align)!=rs.align:
        align = rs.align(rs.stream.color)
    frames = pipe.wait_for_frames() # align to color_stream
    if type(align)==rs.align: frames = align.process(frames)
    return rs_parse_frames(frames, profile)


def rs_parse_frames(frames, profile):
    result = {} # extract as numpy
    assert type(frames)==rs.composite_frame
    assert type(profile)==rs.pipeline_profile
    for sm in profile.get_streams():
        if (sm.stream_type()==rs.stream.infrared):
            fm = frames.get_infrared_frame(sm.stream_index())
        else: fm = frames.first_or_default(sm.stream_type())
        # frames.get_color_frame(); frames.get_depth_frame()
        result[sm.stream_name()] = np.asanyarray(fm.get_data())
    return result # {'Color','Depth','Infrared 1','Infrared 2'}


rs_rgbd = lambda device: rs_get_frames(*device, True)
##########################################################################################
def rs_init(res=(1280,720), fps=15):
    config = rs.config(); print('RealSense:',*res) # create config
    config.enable_stream(rs.stream.depth, *res, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, *res, rs.format.bgr8, fps)
    #config.enable_stream(rs.stream.color, *res, rs.format.rgb8, fps)
    #config.enable_stream(rs.stream.infrared, 1, *res, rs.format.y8, fps)
    #config.enable_stream(rs.stream.infrared, 2, *res, rs.format.y8, fps)
    # pipe = rs.pipeline(); profile = pipe.start(config) # 1st device
    devices = rs_start_devices(config) # {serial: (pipe,profile)}
    return config, devices # pipe.stop()


########################################################
def rs_test(fps=15, dt=1):
    from time import sleep
    config, devices = rs_init(fps=fps)
    while not sleep(dt): # before imshow
        for sn, dev in devices.items():
            im = rs_rgbd(dev)['Color']
            cv2.imshow(f'sn={sn}', im)
        if cv2.waitKey(5)==27: break
    rs_stop_devices(devices); cv2.destroyAllWindows()


##########################################################################################
def rs_yolov5(yolo, cls=None):
    sys.path.append('yolov5')
    from yolov5.infer import yolov5

    config, devices = rs_init()
    det = yolov5(yolo, cls=cls) # init
    while cv2.waitKey(5)!=27:
        for sn, dev in devices.items():
            im = rs_rgbd(dev)['Color']
            im, _, dt, res = det.infer1(im)
            if res: print('%.2fms:'%dt, res)
            cv2.imshow(sn, im); #cv2.imwrite(dst,im)
    rs_stop_devices(devices); cv2.destroyAllWindows()


########################################################
def rs_face(face=None, base=None):
    from det_face import det_face, MTCNN
    from det_face import load_net, load_base

    mtcnn = MTCNN()
    config, devices = rs_init()
    if face!=None and base!=None:
        face = load_net(face)
        base = load_base(base, face, mtcnn)
    while cv2.waitKey(5)!=27:
        for sn, dev in devices.items():
            im = rs_rgbd(dev)['Color']
            im, res = det_face(im, mtcnn, face, base, sn)[:2]
            if type(res)==list: print(res)
    rs_stop_devices(devices); cv2.destroyAllWindows()


########################################################
def rs_window():
    from det_win import det_win, mark_win

    config, devices = rs_init()
    while cv2.waitKey(5)!=27:
        for sn, dev in devices.items():
            im = rs_rgbd(dev)['Color']
            mark_win(im, *det_win(im))
            cv2.imshow(sn, im)
    rs_stop_devices(devices); cv2.destroyAllWindows()


##########################################################################################
if __name__ == '__main__':
    #rs_yolov5('yolov5/yolov5x.pt', cls=[0,63])
    #rs_face('yolov5/vggface2.pt', 'GLDFace')
    rs_window()

