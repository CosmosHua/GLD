'''
This script will use the 2D box from the label rather than from YOLO,
but will still use the neural nets to get the 3D position and plot onto the
image. Press space for next image and escape to quit
'''


from library.Math import *
from library.Plotting import *
from torch_lib.Dataset import *
from torch_lib import Model, ClassAverages

import os, cv2, time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
import numpy as np

# to run car by car
single_car = False


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    weights_path = root + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    assert len(model_lst)>0, 'No previous model found, please train first!'

    print ('Using previous model %s'%model_lst[-1])
    my_vgg = vgg.vgg19_bn(pretrained=False)
    model = Model.Model(features=my_vgg.features, bins=2).cuda()
    checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # defaults to /eval
    dataset = Dataset(root + '/eval')
    averages = ClassAverages.ClassAverages()

    all_images = dataset.all_objects()
    for key in sorted(all_images.keys()):
        start_time = time.time()
        data = all_images[key]

        truth_img = data['Image']
        img = np.copy(truth_img)
        objects = data['Objects']
        cam_to_img = data['Calib']

        for detectedObject in objects:
            label = detectedObject.label
            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img
            input_tensor.cuda()

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(label['Class'])

            argmax = np.argmax(conf)
            cos, sin = orient[argmax, :2]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax] - np.pi

            location = plot_regressed_3d_bbox(img, truth_img, cam_to_img, label['Box_2D'], dim, alpha, theta_ray)
            print('Truth pose: %s\nEstimated location: %s'%(label['Location'], location)) # x,y,z

            # plot car by car
            if single_car:
                numpy_vertical = np.concatenate((truth_img, img), axis=0)
                cv2.imshow('3D-DeepBox', numpy_vertical); cv2.waitKey(0)

        print('Got %s poses in %.3f seconds\n'%(len(objects), time.time()-start_time))

        # plot image by image
        if not single_car:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imshow('3D-DeepBox', numpy_vertical)
            if cv2.waitKey(0) == 27: return


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):
    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)
    if img_2d is not None: plot_2d_box(img_2d, box_2d)
    yaw = alpha + theta_ray
    plot_3d_box(img, cam_to_img, yaw, dimensions, location) # 3d boxes
    return location


if __name__ == '__main__':
    main()

