#!/usr/bin/python3
# coding: utf-8

import os, cv2
import numpy as np


##########################################################################################
def packet_det(sp):
    a,b = [],[]; w = len(sp)
    for i in range(w-1): # endpoint>0
        if i==0 and sp[i]: a.append(i)
        if not sp[i] and sp[i+1]: a.append(i+1)
        if sp[i] and not sp[i+1]: b.append(i)
    if sp[w-1]: b.append(w-1) # endpoint>0

    assert len(a)==len(b), f'\na={a}\nb={b}'
    return np.asarray([a,b]).transpose() # px
    # px = list(zip(a,b)); a,b = list(zip(*px))
    # wid = [b[i]-a[i]+1 for i in range(len(a))]
    # gap = [a[i]-b[i-1]-1 for i in range(1,len(a))]


def packet_reduce(sp, px, dw=20):
    ix, dw = list(px.flatten()), max(20,dw)
    for i in range(1,len(px)):
        a,b = px[i,0],px[i-1,1]; gap = a-b-1
        if dw>gap: # merge small gaps
            ix.remove(a); ix.remove(b); sp[b+1:a]=1
    px = np.asarray(ix).reshape(-1,2)
    for i in range(len(px)):
        a,b = px[i,:2]; wid = b-a+1
        if dw>wid: # erase small packets
            ix.remove(a); ix.remove(b); sp[a:b+1]=0
    px = np.asarray(ix).reshape(-1,2); return px


########################################################
def mark_win(im, x, y, z=0): # alter im
    if str in [type(x),type(y)]: return 'no'
    info = 'x=%d, y=%d, z=%.2fcm' % (sum(x)/2,y,z)
    cv2.putText(im, info, (2,18), 4, 0.8, (0,255,0))
    cv2.rectangle(im, (x[0],0), (x[1],y*2), (0,255,255), 2)
    cv2.rectangle(im, (x[0],0), (x[1],y*2), (255,0,0), 1)
    cv2.circle(im, (sum(x)//2,y), 5, (0,255,255), 2)
    cv2.circle(im, (sum(x)//2,y), 5, (255,0,0), 1)


def det_win(im, show=False):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    v = lab[...,0]; n,r = np.histogram(v,10) # dark
    v, n, no = v.mean(), n.argmax(), '=> Dark Window'
    if v<r[2] and n<2: print(f'{no}:',v,n); return 'no'
    if show: print(f'avg={v}\tmode=({r[n]},{r[n+1]})')

    # H=[0,180), S=[0,255), V=[0,255)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    his = 2*hsv.mean(axis=0) # (hsv>128).sum(axis=0)
    vs = (his[...,2]-his[...,1]).clip(0) # V-S
    av = vs.mean() # vs.sum()/np.count_nonzero(vs)
    sp = vs>av # vs.clip(av); (vs>av)*av+av

    px = packet_det(sp); h,w = im.shape[:2]
    px = packet_reduce(sp, px, w/50)
    if len(px)<1: print(no); return 'no'

    # compare jitter: np.linalg.norm(x,ord=1)/len(x)
    jit = [abs(vs[a+1:b+1]-vs[a:b]).mean() for a,b in px]
    x = px[np.argmax(jit)]; return (x,h//2) # np.append


########################################################
def win_det(src, out=None):
    from glob import glob
    import matplotlib.pyplot as plt

    res = {} # [(left,right),center_y]
    if out: os.makedirs(out, exist_ok=True)
    for f in sorted(glob(src+'/*.jpg')):
        im = cv2.imread(f); f = os.path.basename(f)

        # H=[0,180), S=[0,255), V=[0,255)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        his = 2*hsv.mean(axis=0) # (hsv>128).sum(axis=0)
        vs = (his[...,2]-his[...,1]).clip(0) # V-S
        av = vs.mean() # vs.sum()/np.count_nonzero(vs)
        sp = vs>av # vs.clip(av); (vs>av)*av+av

        plt.imshow(im[::-1, :, ::-1])
        plt.gca().invert_yaxis(); plt.title(f)
        #plt.plot(his[...,0], color='red') # H
        #plt.plot(his[...,1], color='cyan') # S
        #plt.plot(his[...,2], color='yellow') # V
        plt.plot(vs.clip(av), color='b')
        plt.plot(sp*av+av, color='y'); #plt.show()

        px = packet_det(sp); h,w = im.shape[:2]
        px = packet_reduce(sp, px, w/50)
        plt.plot(sp*av+av, color='w'); plt.show()
        if len(px)<1: continue # no packet

        # compare jitter: np.linalg.norm(x,ord=1)/len(x)
        jit = [abs(vs[a+1:b+1]-vs[a:b]).mean() for a,b in px]
        x = px[np.argmax(jit)]; print(jit,'->',x)

        # show results
        mark_win(im, x, h//2); res[f] = [tuple(x), h//2]
        cv2.imshow('det', im); k = cv2.waitKey(bool(out))
        if out: cv2.imwrite(out+'/'+f[:-4]+'_hua.jpg', im)
        if k==27: break
    cv2.destroyAllWindows(); return res


##########################################################################################
if __name__ == '__main__':
    src = '2021-02-04_F5'
    win_det(src, src+'/win_opencv')

