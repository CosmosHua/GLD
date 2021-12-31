#!/usr/bin/python3
# coding: utf-8

import numpy as np
import os, cv2, torch
from time import time
from PIL import Image
# pip3 install facenet-pytorch
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# TODO: add Face_Alignment to MTCNN using landmarks
T2N = lambda x: x.permute(0,2,3,1).numpy().astype('uint8')[...,::-1]
##########################################################################################
def load_img(im, rz=1): # im=BGR, img=RGB
    if type(im)==str and os.path.isfile(im): im = cv2.imread(im)
    #if type(im)==str and os.path.isfile(im): im = Image.open(im)
    if type(im)==np.ndarray: img = Image.fromarray(im[...,::-1])
    if type(im)==Image.Image: img,im = im, np.array(im)[...,::-1]
    assert type(im)==np.ndarray and type(img)==Image.Image
    return im, img.resize((np.array(img.size)*rz).astype(int))
    #return im, img.resize([int(i*rz) for i in img.size])


def cat_face(faces, msg=None, sh=''):
    if faces.dim()<4: faces = faces[None,]
    faces = T2N(faces); N,H,W,C = faces.shape
    dh = int(N**0.5); dw = int(np.ceil(N/dh))
    if dh>dw: dh, dw = dw, dh # require: dw>=dh
    im = np.zeros((dh*(H+1)+1, dw*(W+1)+1, C), 'uint8')
    msg = msg if hasattr(msg,'__getitem__') else ['']*N
    for k,fc in enumerate(faces):
        i,j = 1+(k//dw)*(H+1), 1+(k%dw)*(W+1)
        im[i:i+H, j:j+W] = fc # i=height, j=width
        cv2.putText(im, str(msg[k]), (j,i+12), 3, 0.5, (255,)*3)
    if sh and type(sh)==str: cv2.imshow(sh, im)
    return im


def box_face(im, boxes, msg, dt, bc=(0,255,255)):
    cv2.rectangle(im, (0,0), (im.shape[1],20), (88,)*3, -1)
    cv2.putText(im, '%.2fms: %s'%(dt,msg), (2,15), 3, 0.5, bc)
    if boxes is None or len(msg)!=len(boxes): return im
    for b, m in zip(boxes, msg):
        b = b.astype(int); r = min(1.0,(b[2]-b[0])/55)**0.6
        cv2.rectangle(im, (b[0],b[1]), (b[2],b[3]), bc)
        cv2.putText(im, str(m), (b[0],b[1]-2), 3, 0.5*r, bc)
    return im


##########################################################################################
def set_mtcnn(mtcnn, all, largest, post=False, factor=0.6, min=20):
    if type(mtcnn)!=MTCNN: mtcnn = MTCNN()
    mtcnn.keep_all = all; mtcnn.post_process = post; mtcnn.factor = factor
    mtcnn.select_largest = largest; mtcnn.min_face_size = min; return mtcnn


def load_net(net):
    if type(net)==str and os.path.isfile(net):
        net = InceptionResnetV1(pretrained=net).eval()
    assert type(net)==InceptionResnetV1; return net


def load_base(base, net=None, mtcnn=None):
    if type(base)==str: # base: [feat(n,512), names]
        if os.path.isfile(base): base = torch.load(base)
        if os.path.isdir(base):
            if os.path.isfile(f'{base}/base.pt'):
                base = torch.load(f'{base}/base.pt')
            else: base = init_base(base, net, mtcnn)
    assert type(base)==list; return base # list


Norm = lambda x: (x-127.5)/128 # post_process
########################################################
def init_base(root, net, mtcnn=None, K=100):
    # obtain single face in sample => keep_all=False
    mtcnn = set_mtcnn(mtcnn, all=False, largest=True)
    data = ImageFolder(root); workers = 0 if os.name=='nt' else 4
    data.idx_to_class = {v:k for k,v in data.class_to_idx.items()}

    collate_fn = lambda x: x[0] # for batch=1: just 1st in list
    # collate_fn() used to handle list of (img,label) sample:
    # loader = DataLoader(data, batch_size=2, collate_fn=collate_fn)
    # for i in loader: print(i) # Try: lambda x: list(zip(*x)) OR x
    loader = DataLoader(data, collate_fn=collate_fn, num_workers=workers)

    faces, names, feat = [], [], []
    for img, idx in loader: # keep_all=False: (C,H,W)
        face, box, prob = mtcnn(img, landmarks=False)
        if face is None or box is None: continue
        faces.append(face); names.append(data.idx_to_class[idx])
    faces = torch.stack(faces) # stack: add dim->(N,C,H,W)
    img = cat_face(faces, names, sh='face'); cv2.waitKey()
    cv2.imwrite(f'{root}/{root}.png', img) # save cat_face

    net = load_net(net); faces = Norm(faces)
    for i in range(0, len(faces), K): # feat: (n,512)
        feat.append(net(faces[i:i+K]).detach().cpu())
    base = [torch.cat(feat), names] # cat: merge dim
    torch.save(base, f'{root}/base.pt') # save base
    cv2.destroyAllWindows(); return base


def cmp_feat(faces, net, base, mtcnn=None):
    if type(faces)!=torch.Tensor: # not norm
        faces = torch.Tensor(faces) #(N,C,H,W)
    if faces.dim()<4: faces = faces[None,]

    net = load_net(net); base = load_base(base,net,mtcnn)
    feat = net(Norm(faces)).detach().cpu() #(N,512)->(N,n)
    dis = np.stack([(base[0]-e).norm(dim=1) for e in feat])
    idx = dis.argmin(axis=1) # =torch.stack().numpy()
    return [(base[1][i],d[i]) for i,d in zip(idx,dis)]


RCG = lambda x,t=0.8: [(k if v<t else 'unknown')+': %.3f'%v for k,v in x]
##########################################################################################
def det_face(im, mtcnn=None, net=None, base=None, sh=None):
    mtcnn = set_mtcnn(mtcnn, all=True, largest=False)
    rz = 0.5; im, img = load_img(im,rz); t0 = time();
    # MTCNN() & InceptionResnetV1() both are modified
    boxes, probs = mtcnn.detect(img, landmarks=False)
    faces = mtcnn.extract(img, boxes, save_path=None)
    #faces, boxes, probs = mtcnn(img, landmarks=False)
    dt = (time()-t0)*1E3; res = msg = probs
    if boxes is None:
        im = box_face(im, boxes, msg, dt)
        if type(sh)==str: cv2.imshow(sh, im)
        return im, dt # no faces
    
    if type(base) in (str, list) and \
        type(net) in (str, InceptionResnetV1):
        res = cmp_feat(faces, net, base, mtcnn)
        msg = RCG(res) # face recognition
    boxes /= rz; dt = (time()-t0)*1E3
    im = box_face(im, boxes, msg, dt)

    if type(sh)==str: cv2.imshow(sh, im)
    #if type(sh)==int: print('%.2fms:'%dt, msg)
    if type(sh)==int: cat_face(faces, msg, sh='face')
    return im, res, boxes, faces, dt #(N,C,H,W)


##########################################################################################
if __name__ == '__main__':
    mtcnn = MTCNN(); src = './'
    net = load_net('yolov5/vggface2.pt')
    base = load_base('GLDFace', net, mtcnn)
    #det_face(im, mtcnn, net, base, 'det'); cv2.waitKey()
    for i in os.listdir(src):
        if i[-4:] not in ('.jpg','.png','jpeg'): continue
        im = det_face(src+i, mtcnn, net, base, 'det')[0]
        cv2.waitKey(); cv2.destroyAllWindows()
        if input('Save result?[y/n]: ')=='y':
            cv2.imwrite(src+i[:-4]+'_'+i[-4:], im)
    cv2.destroyAllWindows()

