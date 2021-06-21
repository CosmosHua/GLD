#!/usr/bin/python3
# coding: utf-8

import time as tm
import numpy as np
import os, cv2, json
from glob import glob
from shutil import copyfile
from tqdm import trange, tqdm


INT = lambda x,f=1: tuple(int(i*f) for i in x)
REN = lambda x: ' '.join(x[:-4].split()[::-1])+x[-4:]
##########################################################################################
def test_cam(id=0):
    cap = cv2.VideoCapture(id)
    while cap.isOpened():
        ret, img = cap.read(); assert ret
        cv2.imshow('cap', img)
        if cv2.waitKey(5)==27: break
    cap.release(); cv2.destroyAllWindows()


##########################################################################################
def format_json(src:str, dst=None, dt=4):
    if type(dst)!=str: dst = src[:-5]+'_.json'
    assert src.endswith('.json') and os.path.isfile(src)
    with open(src,'r') as ff: js = json.load(ff)
    with open(dst,'w') as ff: json.dump(js,ff,indent=dt)


##########################################################################################
def binarize(src, thd=100, ext='.png'):
    src = src+'/*'+ext if os.path.isdir(src) else src
    for i in glob(src):
        im = cv2.imread(i,0); cv2.imshow('im', im)
        cv2.createTrackbar('x', 'im', thd, 255, lambda x:0)
        while True:
            thd = cv2.getTrackbarPos('x', 'im')
            img = ((im>thd)*255).astype('uint8')
            cv2.imshow('im', img); k = cv2.waitKey(5)
            if k in (ord('s'), 27):
                cv2.imwrite(i[:-4]+'_.png', img); break
        if k==27: break
    cv2.destroyAllWindows()


def brighten(src, s=1.2, b=0, dst=None):
    im = cv2.imread(src) if type(src)==str else src
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hsv[...,2] = (hsv[...,2]*s+b).clip(0,255).astype('uint8')
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if type(src)==str and type(dst)!=str:
        dst = src[:-4]+f'_{s}'+src[-4:]
    if type(dst)==str: cv2.imwrite(dst, im)
    return im


# cv2.ROTATE_90_COUNTERCLOCKWISE=2=rot-1
# cv2.ROTATE_90_CLOCKWISE=0, cv2.ROTATE_180=1
##########################################################################################
def vid_process(vid, dst, gap=1, rot=0): # rot=(0,1,2,3)
    '''while True:
        rt, im = vid.read(); assert rt
        i = vid.get(cv2.CAP_PROP_POS_FRAMES)
        if i%gap != 0: continue
        if rot>0: im = cv2.rotate(im, rot-1)
        if type(dst)==cv2.VideoWriter: dst.write(im)
        elif type(dst)==str:
            cv2.imwrite(dst+'_%6d.jpg'%i, im)'''
    N = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in trange(N, ascii=True):
        rt, im = vid.read(); #assert rt
        if i%gap!=0 or not rt: continue
        if rot>0: im = cv2.rotate(im, rot-1)
        if type(dst)==cv2.VideoWriter: dst.write(im)
        elif type(dst)==str:
            cv2.imwrite(dst+'_%6d.jpg'%i, im)


def vid2img(src='.', gap=1, rot=0): # rot=(0,1,2,3)
    for v in glob('%s/*.mp4'%src):
        dst = v[:-4]; os.makedirs(dst, exist_ok=True)
        vid = cv2.VideoCapture(v); print('Process: %s'%v)
        vid_process(vid, f'{dst}/{os.path.basename(dst)}', gap, rot)
        vid.release()


def vid_rot(src='.', gap=1, rot=0): # rot=(0,1,2,3)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for v in glob('%s/*.mp4'%src):
        vid = cv2.VideoCapture(v)
        fps = vid.get(cv2.CAP_PROP_FPS)
        N = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if rot in (1,3): w,h = h,w # rotate 90

        dst = v[:-4]+'_.mp4'; print('Process: %s'%v)
        dst = cv2.VideoWriter(dst, fourcc, fps, (w,h))
        vid_process(vid, dst, gap, rot)
        vid.release(); dst.release()


##########################################################################################
def joint_vid(src, dst, drt='horizon'):
    vid, F, N, H, W, R = [], [], [], [], [], []
    if os.path.isfile(dst+'.mp4'): os.remove(dst+'.mp4')
    drt = drt.startswith('h'); src = sorted(glob(f'{src}/*.mp4'))

    print('Please input rotations. Clockwise: 0=0, 1=90, 2=180, 3=270.')
    rot = input(f'{src}\n(default=0, space delimited): ').split()
    rot = [int(i) for i in rot if i.isdigit()]; rot += [0]*(len(src)-len(rot))

    for v,ro in zip(src,rot): # ro=(0,1,2,3)
        v = cv2.VideoCapture(v); vid.append(v)
        F.append(v.get(cv2.CAP_PROP_FPS))
        N.append(int(v.get(cv2.CAP_PROP_FRAME_COUNT)))
        W.append(int(v.get(cv2.CAP_PROP_FRAME_WIDTH)))
        H.append(int(v.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if ro%2!=0: W[-1],H[-1] = H[-1],W[-1]

    print('\nPlease input ranges by seconds (space delimited).')
    for v,n,f in zip(src,N,F):
        rg = input(f'{v} (fps=%.1f, %.2fs): '%(f,n/f)).split()
        rg = [0]*(len(rg)<2) + [float(i)*f for i in rg] + [n]*(len(rg)<1)
        rg = max(0,min(rg)), min(n,max(rg)); R.append(rg)
    N = [int(j-i) for i,j in R]; p = N.index(min(N)) # set anchor
    if drt: W,H = zip(*[(H[p]*w//h,H[p]) for w,h in zip(W,H)])
    else:   W,H = zip(*[(W[p],W[p]*h//w) for w,h in zip(W,H)])

    fps = vid[p].get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    wh = (sum(W),H[p]) if drt else (W[p],sum(H))
    vid_writer = cv2.VideoWriter(dst+'.mp4', fourcc, fps, wh)

    for i in trange(N[p], ascii=True):
        ret, img = [], []
        #ret, img = zip(*[v.read() for v in vid])
        for v,n,(a,b),ro,w,h in zip(vid,N,R,rot,W,H):
            v.set(cv2.CAP_PROP_POS_FRAMES, int(a+i*n/N[p]))
            rt, im = v.read(); #assert rt
            if ro>0: im = cv2.rotate(im, ro-1)
            im = cv2.resize(im, (w,h))
            img.append(im); ret.append(rt)
        if False in ret: del img[-1]; continue
        im = np.hstack(img) if drt else np.vstack(img)
        cv2.imshow(dst, im); cv2.waitKey(1)
        vid_writer.write(im)
    for v in vid: v.release()


##########################################################################################
def blur_score(im, ksz=60): # estimate motion blur
    if type(im)==str and os.path.isfile(im):
        im = cv2.imread(im, 0) # gray
    elif im.ndim>2: # for BGR
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if type(ksz)==int and ksz>0: # using FFT
        cy, cx = [i//2 for i in im.shape[:2]]
        fft = np.fft.fftshift(np.fft.fft2(im))
        fft[cy-ksz:cy+ksz, cx-ksz:cx+ksz] = 0
        fft = np.fft.ifft2(np.fft.ifftshift(fft))
        return np.mean(np.log(np.abs(fft)))*200
    else: # using Laplacian(ksize=1)
        return cv2.Laplacian(im, cv2.CV_64F).var()


def blur_filter(src, dst=None, dev='D435I'):
    print(f'processing: {src}')
    if type(dst)!=str: dst = []
    else: os.makedirs(dst, exist_ok=True)
    for i in tqdm(glob(f'{src}/*.jpg')):
        im = cv2.imread(i); var = blur_score(im)
        if type(dst)==list: # date = tm.ctime(t)[4:]
            i = os.path.basename(i)[:-4]; t = float(i)
            depth = f'{src}/{i}.png'; blur = float('%.3f'%var)
            date = tm.strftime('%Y-%m-%d %H:%M:%S', tm.localtime(t))
            info = dict(color=i+'.jpg', blur=blur, date=date)
            if os.path.isfile(depth): info['depth'] = i+'.png'
            dst.append(info)
        elif type(dev)!=str and var>dev: # dev=100
            copyfile(i, dst+'/'+os.path.basename(i))
        '''im = cv2.putText(im, '%.2f'%var, (2,30), 4, 1, (222,)*3)
        cv2.imshow('im', im); cv2.waitKey(1)'''
    device = dev if type(dev)==str else None
    dst = dict(device=device, images=dst)
    with open(f'{src}.json','w+') as ff:
        json.dump(dst, ff, indent=4); return dst


from threading import Thread
from multiprocessing import Process, Pool, cpu_count
##########################################################################################
def multi_process(func, src='.'):
    src = [i for i in os.listdir(src) if os.path.isdir(i)]
    for i in src:
        # Method 1: using multi_thread
        tp = Thread(target=func, args=(i,)).start()
        # Method 2: using multi_processing
        #tp = Process(target=func, args=(i,)).start()#'''
    # Method 3: using multi_processing_pool
    #pool = Pool().map(func, src)
    # Method 4: using multi_processing_pool
    '''pool = Pool(len(src)) # cpu_count()
    for i in src: pool.apply_async(func, args=(i,))
    pool.close(); pool.join()#'''


##########################################################################################
if __name__ == '__main__':
    #test_cam()
    #binarize('hua.png')
    #vid2img(gap=10)
    vid_rot(rot=0)
    # for 6: 0 1; 71; 1.4 432.5
    #joint_vid('.', 'fuse')
    #multi_process(blur_filter)

