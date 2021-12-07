'''
Images must be in ./Kitti/testing/image_2/ and camera matricies in ./Kitti/testing/calib/
Uses YOLO to obtain 2D box, PyTorch to get 3D box, plots both
'''


from library.Math import *
from library.Plotting import *
from torch_lib.Dataset import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo

import argparse
import numpy as np
import os, cv2, time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--src', default='eval/image_2/',
                    help='Relative path to the directory containing images to detect.')
# TODO: support multiple cal matrix input types
parser.add_argument('--cal-dir', default='camera_cal/',
                    help='Relative path to the directory containing camera calibration form KITTI.')
parser.add_argument('--video', action='store_true',
                    help='Weather or not to advance frame-by-frame as fast as possible. \
                    By default, this will pull images from ./eval/video')
parser.add_argument('--show-yolo', action='store_true',
                    help='Show the 2D BoundingBox detecions on a separate image')
parser.add_argument('--hide-debug', action='store_true',
                    help='Supress the printing of each 3d location')
parser.add_argument('--save', default='eval/out/',
                    help='Relative path to save result images.')


def main():
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    assert len(model_lst)>0, 'No previous model found, please train first!'

    print('Using previous model %s'%model_lst[-1])
    my_vgg = vgg.vgg19_bn(pretrained=False)
    # TODO: load bins from file or something
    model = Model.Model(features=my_vgg.features, bins=2).cuda()
    checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # load yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(bins=2)

    FLAGS = parser.parse_args()
    image_dir = FLAGS.src; cal_dir = FLAGS.cal_dir
    os.makedirs(FLAGS.save, exist_ok=True)
    if FLAGS.video and FLAGS.image_dir == 'eval/image_2/' and FLAGS.cal_dir == 'camera_cal/':
        image_dir = 'eval/video/image_2/'
        cal_dir = 'eval/video/'

    image_dir = os.path.abspath(os.path.dirname(__file__)) + '/' + image_dir
    # using P_rect from global calibration file
    calib_path = os.path.abspath(os.path.dirname(__file__)) + '/' + cal_dir
    calib_file = calib_path + 'calib_cam_to_cam.txt'

    # using P from each frame
    # calib_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/testing/calib/'

    averages = ClassAverages.ClassAverages()
    imgs = [i for i in os.listdir(image_dir) if i.split('.')[-1] in ('png','jpg','jpeg')]
    for im in sorted(imgs):
        # P for each frame
        # calib_file = calib_path + id + '.txt'

        start_time = time.time()
        truth_img = cv2.imread(os.path.join(image_dir, im))
        rt = 480/truth_img.shape[0]
        truth_img = cv2.resize(truth_img, None, fx=rt, fy=rt)
        img = truth_img.copy()
        yolo_img = truth_img.copy()
        detections = yolo.detect(yolo_img)

        for detection in detections:
            if not averages.recognized_class(detection.detected_class): continue

            # This is throwing when the 2d bbox is invalid
            # TODO: better check
            try:
                detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
            except: continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = detection.box_2d
            detected_class = detection.detected_class

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            cos, sin = orient[argmax, :2]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax] - np.pi

            if FLAGS.show_yolo:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
            else:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)
            if not FLAGS.hide_debug: print('Estimated location: %s'%location) # x,y,z

        if not FLAGS.hide_debug:
            print('Got %s poses in %.3f seconds\n'%(len(detections), time.time()-start_time))

        if FLAGS.show_yolo: img = np.concatenate((truth_img, img), axis=0)
        cv2.imshow('3D-DeepBox', img); k = cv2.waitKey(int(FLAGS.video)*5)

        if k==ord('s'): cv2.imwrite(os.path.join(FLAGS.save, im), img)
        elif k==27: return


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):
    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)
    if img_2d is not None: plot_2d_box(img_2d, box_2d)
    yaw = alpha + theta_ray
    plot_3d_box(img, cam_to_img, yaw, dimensions, location) # 3d boxes
    return location


if __name__ == '__main__':
    main()

