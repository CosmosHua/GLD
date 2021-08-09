#!/usr/bin/python3
# coding: utf-8

import os, sys, json
from collections import defaultdict

usage_text = """
This script creates a COCO-style annotation file by fusing one or more annotation files.

Usage: python FuseSet.py output_name [set1 range1 [set2 range2 [...]]]

To use, specify the output annotation name and any number of (set, range) pairs, where the sets
are in the form <set_name>/annotations.json and ranges are python-evalable ranges. The resulting
json will be spit out as <output_name>.json in the current folder.

For instance,
    python FuseSet.py trainval35k train2014 : val2014 :-5000

This will create an trainval35k.json file including annotations of
all images from train2014 and the first 35000 images from val2014.

You can also specify only one set:
    python FuseSet.py minival5k val2014 -5000:

This will put annotations of the last 5k images from val2014 into minival5k.json."""


anns_path = '%s/annotations.json'
fields_once = ('info', 'licenses', 'type')
fields_to_fuse = ('images', 'annotations') #'categories'
ReCat = {'tv': 'monitor'} # rename categories


##########################################################################################
if __name__ == '__main__':
    if len(sys.argv) < 4 or len(sys.argv) % 2 != 0:
        print(usage_text); exit()
    
    out_JS = sys.argv[1]; ST = sys.argv[2:]
    ST = [(ST[i],ST[i+1]) for i in range(0,len(ST),2)]

    out = {x:[] for x in fields_once + fields_to_fuse}

    CATs = {} # store all {name:cat}
    cat_sid, img_sid, ann_sid = 0, 0, 0
    for idx, (name, pick) in enumerate(ST):
        print('Loading set: %s' % name)
        with open(anns_path % name, 'r') as ff:
            JS = json.load(ff)
        
        if idx==0: # only keep the first set's
            for x in fields_once:
                if x in JS: out[x] = JS[x]

        # re-index images->{id:image}
        print('Rebuilding image index...')
        for x in JS['images']: x['id'] += img_sid
        imgs = {x['id']:x for x in JS['images']}

        # Fuse categories: To avoid index conflicts,
        # re-index all new cats, then fuse pre-existed.
        FMap = {} # {new_id:pre-exist_id}
        print('Rebuilding category index...')
        for x in JS['categories']:
            # needless: blank->underline, rename cat
            x['name'] = x['name'].replace(' ','_')
            if x['name'] in ReCat: x['name']=ReCat[x['name']]

            x['id'] += cat_sid; cls = x['name']
            if cls not in CATs: CATs[cls] = x
            else: FMap[x['id']] = CATs[cls]['id']

        # re-index anns->{img_id:anns}
        ann_mid = ann_sid # max ann_id
        anns = defaultdict(lambda:[])
        print('Rebuilding annotations...')
        for x in JS['annotations']:
            x['id'] += ann_sid # re-index new anns
            x['image_id'] += img_sid # re-index new imgs
            cid = x['category_id'] + cat_sid # re-index new cats
            x['category_id'] = cid if cid not in FMap else FMap[cid]
            anns[x['image_id']].append(x) # store {image_id:anns}
            if ann_mid<x['id']: ann_mid = x['id'] # find max
        
        # update start_id; +1 for 0-index
        ann_sid = ann_mid+1 # update sid
        img_sid = max(imgs)+1 # =imgs.keys()
        for x in CATs.values(): # find max
            if cat_sid<x['id']: cat_sid = x['id']
        cat_sid += 1 # for next 0-index set

        pick_ids = sorted(list(imgs.keys()))
        pick_ids = eval('pick_ids[%s]' % pick)
        print('Fusing %d images...\n' % len(pick_ids))
        for x in pick_ids:
            out['images'].append(imgs[x])
            out['annotations'] += anns[x]
    out['categories'] = list(CATs.values())

    print('Saving result...Done!')
    with open(out_JS+'.json','w+') as ff:
        json.dump(out, ff, indent=4)
    
########################################################
    IDC = {CATs[x]['id']:x for x in CATs}
    # YOLACT LABEL_MAP: =>continuous 1-index
    LMAP = {x:i+1 for i,x in enumerate(IDC)}

    print("\nFused Categories:\n", IDC)
    print("\nYOLACT CLASSES:\n", list(CATs))
    print("\nYOLACT LABEL_MAP:\n", LMAP)

    with open(out_JS+'.txt','w') as ff:
        ff.write(str(list(CATs))+"\n")
        ff.write(str(LMAP)+"\n")

