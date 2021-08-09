#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, sys, cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import matplotlib; matplotlib.use('qt5agg')


##########################################################################################
def ShowCoco(root:str, cls:str, N=10):
    # COCOAPI init annotations
    coco=COCO(root+'/annotations.json')
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms = [i['name'] for i in cats] # sorted
    print('\nCOCO categories: \n%s\n'%', '.join(nms))
    nms = set([i['supercategory'] for i in cats])
    if None in nms: nms.discard(None); nms.add('NULL')
    print('COCO supercategories: %s\n'%', '.join(nms))

    # get all images containing given categories, select one at random
    imgDir = [i for i in os.listdir(root) if os.path.isdir(root+'/'+i)][0]
    catIds = coco.getCatIds(catNms=[cls])
    imgIds = coco.getImgIds(catIds=catIds)
    print('\nImages IDs: ', imgIds)
    for i in np.random.randint(0,len(imgIds),10):
        img = coco.loadImgs(imgIds[i])[0]
        print('Selected Image INFO: ', img)

        # load and display image
        dd = root+('' if imgDir in img['file_name'] else '/'+imgDir)
        im = cv2.imread('%s/%s'%(dd, img['file_name']))[...,::-1]
        #plt.imshow(im); plt.axis('off'); plt.show()

        # load and display instance annotations
        plt.imshow(im); plt.axis('off')
        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns); plt.show()


##########################################################################################
if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    ShowCoco('coco_door', 'door')
