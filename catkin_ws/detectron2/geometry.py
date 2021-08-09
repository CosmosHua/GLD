#!/usr/bin/python3
# coding: utf-8

import cv2
import numpy as np


CK = [263.6988, 0, 344.7538, 0, 263.6988, 188.0399, 0, 0, 1] # cam.K
##########################################################################################
# 2D: origin=top-left, u=rightward, v=downward
# 3D: origin=center, x=rightward, y=downward, z=forward
def Camera_2Dto3D(u, v, dp): # (3,N), units: mm
    dp = np.asarray(dp); dim = len(dp.shape)
    # filter nan/inf: (-inf, inf, nan)->0
    dp = np.where(abs(dp)<np.inf, dp, 0)
    u,v = [np.int32(u).ravel() for i in (u,v)]
    z = dp[v,u] if dim>1 else dp # z=depth
    fx,_,cx,_,fy,cy = CK[:6] # cam intrinsics
    pt = np.array([z*(u-cx)/fx, z*(v-cy)/fy, z])
    pt = pt[:, np.where(z!=0)[0]] # filter z=0
    return pt*(1 if z.dtype==np.uint16 else 1E3)


def Camera_3Dto2D(pt):
    x,y,z = np.asarray(pt)[:3] # (3,N)
    fx,_,cx,_,fy,cy = CK[:6] # cam intrinsics
    zz = np.where(z!=0, z, np.finfo(float).eps)
    u = (x*fx + z*cx)/zz; v = (y*fy + z*cy)/zz
    return np.asarray([u,v]) # (2,N)


##########################################################################################
# Camera extrinsics/extrinsic parameters
# quat = np.asarray([w,x,y,z]), t = [x,y,z]
def Quart2TR(quat, t=0): # Robot2World_TRMatrix
    q = np.asarray(quat); n = np.dot(q,q) # Quaternion
    t, dim = ([0]*3,3) if type(t) in (int,float) else (t,4)
    if n<np.finfo(q.dtype).eps: return np.identity(dim)
    q *= np.sqrt(2.0/n); q = np.outer(q,q)
    TR = np.asarray( # transform/rotation matrix
        [[1.0-q[2,2]-q[3,3], q[1,2]+q[3,0], q[1,3]-q[2,0], 0.0],
         [q[1,2]-q[3,0], 1.0-q[1,1]-q[3,3], q[2,3]+q[1,0], 0.0],
         [q[1,3]+q[2,0], q[2,3]-q[1,0], 1.0-q[1,1]-q[2,2], 0.0],
         [t[0], t[1], t[2], 1.0]], dtype=q.dtype).T
    return TR if dim>3 else TR[:3,:3] # extrinsics


def Camera2World(TR, pt, ofs=0.35): #(3,N)
    #x,y,z = pt[:3]; x,y,z = z,-x,ofs-y # Cam->Robot
    #pt = np.asarray([x,y,z]).reshape([3,-1]) # (3,N)
    pt = np.asarray([pt[2], -pt[0], ofs-pt[1]])
    pt = np.insert(pt[:3], 3, values=1.0, axis=0)
    return TR.dot(pt[:4])[:3] # Robot->World


def World2Camera(TR, pt, ofs=0.35): #(3,N)
    #x,y,z = pt[:3]; x,y,z = -y,ofs-z,x # Robot->Cam
    pt = np.insert(pt[:3], 3, values=1.0, axis=0)
    pt = np.linalg.pinv(TR).dot(pt)[:3] # Robot
    return np.asarray([-pt[1], ofs-pt[2], pt[0]])


##########################################################################################
def PCA(pt, rank=0):
    dim, N = pt.shape # (3,N)->normalized
    pt = pt - pt.mean(axis=1, keepdims=True)
    cov = pt.dot(pt.T) / (N-1) # covariance matrix
    val, vec = np.linalg.eig(cov) # np.linalg.svd
    rank = dim if rank<1 else rank # dim>=max(rand)
    idx = val.argsort()[::-1][:rank] # descending
    return val[idx], vec[:,idx].T # (rank,dim)


# Best-fit linear plane, for the Eq: z = a*x + b*y + c.
# Ref: https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
def FitPlane(pt): # pt=(N,3)
    pt = np.asarray(pt); x,y,z = pt.transpose()
    A = np.c_[x, y, np.ones(len(pt))] # z=A.dot(P)
    P, res, rank, s = np.linalg.lstsq(A, z)
    a,b,c = P; n = np.linalg.norm([a,b,-1])
    return np.array([a,b,-1,c])/n


# uv: (2,N); TR: camera extrinsics
# plane: ax+by+cz+d=0, n=(a,b,c), d=-n*p0
# Ref: https://www.cnblogs.com/qiu-hua/p/8001177.html
def uvRay2Plane(uv, plane, TR=0):
    # (p1,p2): arbitrary points, same uv & diff depth
    p1, p2 = [Camera_2Dto3D(*uv,i) for i in (1,50)]
    if type(TR)==np.ndarray: # camera->world (3,)
        p1, p2 = [Camera2World(TR,i) for i in (p1,p2)]
    n, d = np.asarray(plane[:3]), plane[3:]
    d = -n.dot(d[:3]) if len(d)>1 else d # if d=p0
    k = -(n.dot(p1)+d)/n.dot(p2-p1) # slope ratio
    return p1 + k*(p2-p1) # intersect point


##########################################################################################
def Quart2Euler(qt):
    if type(qt) in (list,tuple): x,y,z,w = qt
    elif type(qt)==dict:
        x,y,z,w = qt['x'],qt['y'],qt['z'],qt['w']
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(z*z+y*y))
    roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = np.arcsin(2*(w*y-x*z))
    return yaw, roll, pitch # radian: [-pi,pi]


def Euler2Quart(yaw, roll, pitch):
    euler = [yaw/2, roll/2, pitch/2]
    siny, sinr, sinp = np.sin(euler)
    cosy, cosr, cosp = np.cos(euler)
    x = sinp*siny*cosr + cosp*cosy*sinr
    y = sinp*cosy*cosr + cosp*siny*sinr
    z = cosp*siny*cosr - sinp*cosy*sinr
    w = cosp*cosy*cosr - sinp*siny*sinr
    return x, y, z, w # euler: radian


##########################################################################################
if __name__ == '__main__':
    u = range(4); v = range(1,5); d = range(-1,3)
    pc = Camera_2Dto3D(u,v,d); print('cam3d:\n', pc)
    uv = Camera_3Dto2D(pc); print('cam2d:\n', uv)
    # uv=[-0. 0. 2. 3.], [ 1. 0. 3. 4.]

    q = np.asarray([np.pi/4, 0, 0, np.pi/4]) # z-axis 90
    t = np.asarray([1,2,3]); pc = np.asarray([1,1,1])
    T = Quart2TR(q,t); print('TR:\n', T)

    pt = Camera2World(T,pc); print('world:\n', pt)
    pc = World2Camera(T,pt); print('cam3d:\n', pc)
    # pt=[2. 3. 2.85], pc=[1. 1. 1.]

    uv = Camera_3Dto2D(pc); print('cam2d:\n', uv)
    p1 = Camera2World(T,Camera_2Dto3D(0,0,0)); print(p1,pt)
    pt = uvRay2Plane(uv, T, [0,0,1,0]); print('world:\n', pt)
    # uv=[697.7875 540.5965], pt=[2. 3. 2.85]
    # p1=[1. 2. 3.85], uv_ray_pt=[4.85 5.85 0.]

    v,vc = PCA(np.random.rand(3,100)); print(v,'\n',vc)
