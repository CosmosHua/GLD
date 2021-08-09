#!/usr/bin/python3
# coding: utf-8

import os, json
from glob import glob


##########################################################################################
def AddItem(src, KV): # KV: [(key,val)]/{key:val}
    for ff in glob(src+'/*.json'):
        with open(ff,'rb+') as f: data = json.load(f)
        if type(KV) in (list,tuple):
            for key,val in KV: data[key] = val
        elif type(KV)==dict: data.update(KV)
        with open(ff,'w+') as f: json.dump(data, f, indent=4)


def ReItem(src, KON):
    for ff in glob(src+'/*.json'):
        with open(ff,'rb+') as f: data = json.load(f)
        idx = [data['imagePath'].rfind(i) for i in ('/','\\')]
        data['imagePath'] = data['imagePath'][max(idx)+1:]
        data['imageData'] = None # --nodata: remove base64
        if type(KON) in (list,tuple): # [(k,*v) for k,v in KON.items()]
            for key, old, new in KON: # only for str
                if key in data: data[key] = data[key].replace(old,new); continue
                for s in data['shapes']: s[key] = s[key].replace(old,new)
        elif type(KON)==dict: # {k:(o,n) for k,o,n in KON}
            for key, (old,new) in KON.items(): # only for str
                if key in data: data[key] = data[key].replace(old,new); continue
                for s in data['shapes']: s[key] = s[key].replace(old,new)#'''
        with open(ff,'w+') as f: json.dump(data, f, indent=4)


##########################################################################################
def Fit2Labelme(src, fmt, single=True, inplace=True):
    shp = dict(label='', points=[], group_id=None, shape_type='', flags={})
    out = dict(version='4.4.0', flags={}, shapes=[shp], imagePath='',
        imageData=None, imageHeight=0, imageWidth=0) # labelme --nodata
    src = glob(src+'/*.json') if os.path.isdir(src) else [src]
    for ff in src:
        #with open(ff,'r', encoding='UTF-8') as f:
        with open(ff,'rb') as f: data = json.load(f)
        if single: # for separate jsons
            if 'Colabeler' in fmt:
                if not data['labeled']: os.remove(ff); continue
                Colabeler2Labelme(out, data)
            elif 'Testin' in fmt: Testin2Labelme(out, data)
            elif 'Stardust' in fmt: Stardust2Labelme(out, data)
            fo = ff if inplace else ff[:-5]+'_.json'
            with open(fo,'w+') as f: json.dump(out, f, indent=4)
        else: # for merged json
            for dc in data: # list of dicts
                if 'Colabeler' in fmt:
                    if not dc['labeled']: continue
                    Colabeler2Labelme(out, dc)
                elif 'Testin' in fmt: Testin2Labelme(out, dc)
                elif 'Stardust' in fmt: Stardust2Labelme(out, dc)
                fo = os.path.splitext(out['imagePath'])[0]+'.json'
                with open(fo,'w+') as f: json.dump(out, f, indent=4)


def Colabeler2Labelme(out, data):
    def obj2shp(x): # x: {'name',('bndbox'/'polygon'/'cubic_bezier')}
        shp = dict(label=x['name'], points=[], group_id=None, shape_type='', flags={})
        shp['shape_type'] = 'rectangle' if 'bndbox' in x else 'polygon'
        p = list(x.keys())[1]; p = [float(v) for k,v in x[p].items() if 'c' not in k]
        shp['points'] = [[p[i],p[i+1]] for i in range(0,len(p),2)]; return shp
    out['shapes'] = [obj2shp(x) for x in data['outputs']['object']]
    out['imagePath'] = os.path.basename(data['path'])
    out['imageHeight'] = data['size']['height']
    out['imageWidth'] = data['size']['width']


def Stardust2Labelme(out, data):
    # annotations: for multi-type, such as polygon,rectangle,...
    # |-slotsChildren: for multi-instance, such as cup1,cup2,...
    # |--children: for multi-label, such as material,colour,...
    # |---value: for compatible-label, such as blue and white,...
    rec = data['task']['task_params']['record']
    out['imagePath'] = os.path.basename(rec['attachment'])
    out['imageHeight'] = rec['metadata']['size']['height']
    out['imageWidth'] = rec['metadata']['size']['width']
    
    shp = out['shapes'][0]; out['shapes'] = []
    for shape in data['task_result']['annotations']:
        for instance in shape['slotsChildren']:
            slot = instance['slot']; shp['shape_type'] = slot['type']
            shp['points'] = [list(i.values()) for i in slot['vertices']]
            for label in instance['children']:
                shp['label'] = ','.join(label['input']['value'])
                out['shapes'].append(shp.copy())


def Testin2Labelme(out, data):
    out['imagePath'] = os.path.basename(data['imageName'])
    out['imageHeight'] = data['imageHeight']
    out['imageWidth'] = data['imageWidth']
    shp = out['shapes'][0]; out['shapes'] = []
    for instance in data['Data']['svgArr']:
        shp['label'] = instance['name']
        shp['shape_type'] = instance['tool']
        shp['points'] = [list(i.values()) for i in instance['data']]
        out['shapes'].append(shp.copy())


##########################################################################################
if __name__ == '__main__':
    #Fit2Labelme('.', 'Testin', False)
    KV = {'version':'4.2.10', 'flags':{}}; AddItem('.', KV)
    #KON = {'label':('door_other','door_unknown')}; ReItem('.', KON)
