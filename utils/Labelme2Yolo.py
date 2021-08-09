#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, sys, yaml
import os.path as osp
import argparse, uuid
from PIL import Image
from glob import glob
from shutil import copy, copyfile
from tqdm import tqdm

import labelme, imgviz
try: import pycocotools.mask
except ImportError:
    print('Please: pip install pycocotools\n'); sys.exit(1)


##########################################################################################
def Labelme2Yolo(ym='det'):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    parser.add_argument('--viz', help='visualize', action='store_true')
    args = parser.parse_args(); output_dir = args.output_dir

    assert not osp.exists(output_dir); os.makedirs(output_dir)
    if args.viz: os.makedirs(osp.join(output_dir, 'Viz'))
    os.makedirs(osp.join(output_dir, 'images'))
    os.makedirs(osp.join(output_dir, 'labels'))
    print('Creating dataset:', output_dir)

    cls_to_id = {}; ofs = 0
    for i, line in enumerate(open(args.labels).readlines()):
        cls_name = line.strip()
        if cls_name.startswith('_'): ofs = 1; continue
        class_id = i - ofs # start with -1 or 0
        cls_to_id[cls_name] = class_id

    label_files = glob(osp.join(args.input_dir, '*.json'))
    for image_id, jsonfile in enumerate(tqdm(label_files)):
        label_file = labelme.LabelFile(filename=jsonfile)
        img = labelme.utils.img_data_to_arr(label_file.imageData)

        base = osp.splitext(osp.basename(jsonfile))[0]+'.jpg'
        dst_img = osp.join(output_dir, 'images', base).replace('\\','/')
        if label_file.imagePath.endswith('.jpg'):
            #copy(osp.join(args.input_dir, label_file.imagePath), dst_img)
            copyfile(osp.join(args.input_dir, label_file.imagePath), dst_img)
        else: imgviz.io.imsave(dst_img, img) #Image.fromarray(img).save(dst_img)

        masks = {} # for area
        for shape in label_file.shapes:
            points = shape['points']
            label = shape['label']
            group_id = shape.get('group_id')
            shape_type = shape.get('shape_type', 'polygon')
            mask = labelme.utils.shape_to_mask(img.shape[:2], points, shape_type)

            if group_id is None: group_id = uuid.uuid1()
            instance = (label, group_id)
            if instance in masks:
                masks[instance] = masks[instance] | mask
            else: masks[instance] = mask

        H,W = img.shape[:2]; box = []; res = ''
        for (cls_name, group_id), mask in masks.items():
            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            bbox = pycocotools.mask.toBbox(mask).ravel()
            bbox[0] += bbox[2]/2; bbox[1] += bbox[3]/2
            bbox[::2] /= W; bbox[1::2] /= H; box += [bbox]

            if cls_name not in cls_to_id: continue
            cls_id = cls_to_id[cls_name] # top_left->center
            res += '%d %.6f %.6f %.6f %.6f\n'%(cls_id, *bbox)

        dst_txt = osp.join(output_dir, 'labels', base[:-4])
        with open(dst_txt+'.txt', 'w') as f: f.write(res)

        if args.viz and box: # center->top_left
            x,y,w,h = np.array(box).transpose(); x-=w/2; y-=h/2
            box = np.array([y*H,x*W,(y+h)*H,(x+w)*W]).transpose()
            c2i = lambda x: cls_to_id[x] if x in cls_to_id else len(cls_to_id)
            lab, cap, mk = zip(*[(c2i(c),c,mk) for (c,g),mk in masks.items()])
            viz = imgviz.instances2rgb(image=img, labels=lab,
                bboxes=list(box), #masks=mk,
                captions=cap, font_size=12, line_width=2)
            imgviz.io.imsave(osp.join(output_dir, 'Viz', base), viz)

    res = dict(train=f'../{output_dir}/images/', val=f'../{output_dir}/images/',
        nc=len(cls_to_id), names=[i for i in cls_to_id])
    with open(osp.join(output_dir, ym+'.yaml'), 'w') as f:
        yaml.dump(res, f, sort_keys=False)


def ShowYolo(src):
    import cv2
    for i in glob(src+'/labels/*.txt'):
        im = osp.basename(i)[:-4]+'.jpg'
        im = cv2.imread(src+'/images/'+im)
        H,W = im.shape[:2]; box = []
        for ln in open(i,'r').readlines():
            ln = ln.strip().split()
            box.append([float(j) for j in ln])
        if len(box)<1: continue # center->top_left
        c,x,y,w,h = np.array(box).transpose(); x-=w/2; y-=h/2
        box = np.array([y*H,x*W,(y+h)*H,(x+w)*W]).transpose()
        c = np.int32(c).tolist(); cap = [str(j) for j in c]

        viz = imgviz.instances2rgb(image=im, labels=c,
            bboxes=list(box), #masks=mk,
            captions=cap, font_size=12, line_width=2)
        cv2.imshow('det', viz)
        if cv2.waitKey()==27: break


##########################################################################################
if __name__ == '__main__':
    Labelme2Yolo()
    #ShowYolo('Handle_Door')

