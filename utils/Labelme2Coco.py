#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, sys, json
import os.path as osp
import argparse, uuid
from PIL import Image
from glob import glob
from datetime import datetime
from shutil import copy, copyfile
from collections import defaultdict
from tqdm import tqdm

import labelme, imgviz
try: import pycocotools.mask
except ImportError:
    print('Please: pip install pycocotools\n'); sys.exit(1)


##########################################################################################
def Labelme2Coco(rd=False):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    parser.add_argument('--viz', help='visualize', action='store_true')
    args = parser.parse_args(); output_dir = args.output_dir

    assert not osp.exists(output_dir); os.makedirs(output_dir)
    if args.viz: os.makedirs(osp.join(output_dir, 'Viz'))
    os.makedirs(osp.join(output_dir, 'Images'))
    print('Creating dataset:', output_dir)

    now = datetime.now()
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f') ),
        licenses=[dict(url=None, id=0, name=None)],
        images=[], # license, url, file_name, height, width, date_captured, id
        type='instances',
        annotations=[], # segmentation, area, iscrowd, image_id, bbox, category_id, id
        categories=[], # supercategory, id, name
    )

    cls_to_id = {}; ofs = 0
    for i, line in enumerate(open(args.labels).readlines()):
        cls_name = line.strip()
        if cls_name.startswith('_'): ofs = 1; continue
        class_id = i - ofs # start with -1 or 0
        cls_to_id[cls_name] = class_id
        d = cls_name.find('_') # rfind('_')
        supcat = cls_name[:d] if d>0 else cls_name
        data['categories'].append(
            dict(supercategory=supcat, id=class_id, name=cls_name) )

    label_files = glob(osp.join(args.input_dir, '*.json'))
    for image_id, jsonfile in enumerate(tqdm(label_files)):
        label_file = labelme.LabelFile(filename=jsonfile)
        img = labelme.utils.img_data_to_arr(label_file.imageData)

        base = osp.splitext(osp.basename(jsonfile))[0]+'.jpg'
        dst_img = osp.join(output_dir, 'Images', base).replace('\\','/')
        if label_file.imagePath.endswith('.jpg'):
            #copy(osp.join(args.input_dir, label_file.imagePath), dst_img)
            copyfile(osp.join(args.input_dir, label_file.imagePath), dst_img)
        else: imgviz.io.imsave(dst_img, img) #Image.fromarray(img).save(dst_img)
        dst_img = osp.relpath(dst_img, output_dir).replace('\\','/') if rd else base

        data['images'].append(
            dict(license=0,
                url=None,
                file_name= dst_img,
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id) )

        masks = {} # for area
        segmentations = defaultdict(list) # for segmentation
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

            if shape_type == 'rectangle':
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            if shape_type == 'circle':
                (x1, y1), (x2, y2) = points
                r = np.linalg.norm([x2-x1, y2-y1])
                # r*(a-2*sin(a/2))<x, a=2*pi/N => N>pi*(r/3x)**(1/3)
                #N = max(int(np.pi*(r/3)**(1/3)), 12)
                # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
                N = max(int(np.pi/np.arccos(1-1/r)), 12)
                i = np.arange(N)
                x = x1 + r * np.sin(2 * np.pi / N * i)
                y = y1 + r * np.cos(2 * np.pi / N * i)
                points = np.stack((x, y), axis=1).ravel().tolist()
            else: points = np.asarray(points).ravel().tolist()
            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in cls_to_id: continue
            cls_id = cls_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask)) # (x,y,w,h)
            bbox = pycocotools.mask.toBbox(mask).ravel().tolist()

            data['annotations'].append(
                dict(id=len(data['annotations']),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0) )

        if args.viz and masks:
            c2i = lambda x: cls_to_id[x] if x in cls_to_id else len(cls_to_id)
            lab, cap, mk = zip(*[(c2i(c),c,mk) for (c,g),mk in masks.items()])
            viz = imgviz.instances2rgb(image=img, labels=lab,
                masks=mk, captions=cap, font_size=12, line_width=1)
            imgviz.io.imsave(osp.join(output_dir, 'Viz', base), viz)

    ann_file = osp.join(output_dir, 'annotations.json')
    with open(ann_file,'w+') as ff: json.dump(data, ff, indent=4)


##########################################################################################
if __name__ == '__main__':
    Labelme2Coco()

