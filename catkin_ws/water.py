#!/usr/bin/python3
# coding: utf-8

import numpy as np
import socket, json


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


TYPE = ('response','callback','notification')
##########################################################################################
def tcp_init(addr=None): # initialize tcp
    #ip = input('IP:'); port = int(input('Port:'))
    if type(addr)!=tuple: addr = ('192.168.10.10', 31001)
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.connect(addr); return tcp # tcp.settimeout(0.2)


# bytes.decode(encoding='utf-8')->str
def tcp_get(tcp=None, show=False, bt=0.2):
    if type(tcp)!=socket.socket: tcp = tcp_init()
    t = tcp.gettimeout(); tcp.settimeout(bt)
    try: # for case of tcp.recv() time_out
        res = tcp.recv(4096) # recv() is block func
        while res[-1:]!=b'\n': res += tcp.recv(4096)
    except: tcp.settimeout(t); return [{}] # time_out
    res = b','.join(res.split(b'\n')[:-1]) #.decode()
    #res = ','.join(res.decode().split('\n')[:-1])
    res = json.loads(b'[%s]'%res) # f'[{res}]'
    if show: # print(time.ctime()) # old->new
        for i in res: print(json.dumps(i,indent=4))
    tcp.settimeout(t); return res # list of dicts


def tcp_filter(tcp=None, show=False, bt=2, n=1):
    N = 100; st = tcp_get(tcp, show, bt)[-N:]
    rst = {t:[] for t in TYPE}; n = n if n else N
    for i in st[::-1]: # latest: new->old
        if 'type' in i and len(rst[i['type']])<n:
            rst[i['type']].append(i) # classify
    rst = {k:(v if v else [{}]) for k,v in rst.items()}
    if n==1: rst = {k:v[0] for k,v in rst.items()}
    return rst # notification may lost/invalid


# str.encode(encoding='utf-8')->bytes
def tcp_cmd(tcp=None, cmd=None, show=False):
    if type(tcp)!=socket.socket: tcp = tcp_init()
    if type(cmd)!=str: cmd = input('[cmd]: ')
    print('=>', cmd.split('&')[:2]) # echo cmd
    tcp_get(tcp, False, 0.2) # flush buffer
    tcp.send(f'/api/{cmd}'.encode()) # execute
    return tcp_filter(tcp,show,3)['response']


def dc(x): a,*b = x; return {a:dc(b)} if b else a
##########################################################################################
def parse_cmd(cmd):
    cmd, *par = cmd.split('?') # ?: 1/0
    par = dict(i.split('=') for i in par[0].split('&')) \
            if par else {} # parse param pairs -> dict
    return {cmd:par} # dc(cmd.split('/')+[par])


def format_pos(ps): # convert format
    if 'theta' not in ps: # need: 'orientation'
        ps.update(ps['position']) # extract 'position'
        ps['theta'] = Quart2Euler(ps['orientation'])[0]
        ps.pop('position'); ps.pop('orientation')
    return ps # keys: ('x','y','z','theta')


def get_marker(tcp=None, key=None):
    while True: # avoid getting empty dict
        mark = tcp_cmd(tcp, 'markers/query_list')
        if 'results' in mark: mark = mark['results']; break
    return dict(filter(key, mark.items())) if callable(key) else mark
    # OR: dict(x for x in mark.items() if key(x))
    # OR: {k:v for k,v in mark.items() if key((k,v))}
    # OR: {x[0]:x[1] for x in mark.items() if key(x)}


##########################################################################################
# valid: only all passages are rectangular
def pos_cost(a, b, axis=None): # a=current, b=target
    a, b = format_pos(a), format_pos(b)
    x, y = b['x']-a['x']+1E-20, b['y']-a['y']
    t = np.arctan(y/x)-(axis if axis!=None else a['theta'])
    cos = abs(np.cos(t)); sin = abs(np.sin(t))
    dis = np.linalg.norm([x,y]); return dis*(sin+cos)


# suffix: W=Window, I=Indoor, G=Glass_wall
# eg: TG={'522':{'I':['522I'],'W':['522W']}}
def target_layout(tcp=None, cf=None): # NOT optimal
    # OR: 'map/get_current_map'->'results'->'floor'
    st = tcp_cmd(tcp,'robot_status')['results']; TG = {}
    cp = st['current_pose']; cf = cf if type(cf)==int else st['current_floor']
    mark = get_marker(tcp); axis = format_pos(mark['home']['pose'])['theta']
    door = {k:v['pose'] for k,v in mark.items() if k.isdigit() and v['floor']==cf}
    for i in range(len(door)): # door will be altered
        m = min(door, key=lambda k: pos_cost(cp, door[k], axis))
        cp = door[m]; door.pop(m); TG[m] = {} # nearest door
        TG[m]['I'] = [k for k in mark if m+'I' in k] # indoor
        TG[m]['W'] = [k for k in mark if m+'W' in k] # window
        TG[m]['G'] = [k for k in mark if m+'G' in k] # glass_wall
    print(f'F{cf}:', '->'.join(TG)); return TG


##########################################################################################
def marker_dis(tcp=None, all=False):
    st, dis = tcp_cmd(tcp,'robot_status')['results'], {}
    cp, cf = st['current_pose'], st['current_floor']
    for k,v in get_marker(tcp).items(): # cp={theta,x,y}
        if v['floor']!=cf: continue # current floor
        v = format_pos(v['pose']) # {x,y,z,theta}
        dis[k] = np.linalg.norm([cp[i]-v[i] for i in cp])
    #dis = dict(sorted(zip(dis.values(), dis.keys())))
    return dis if all else min(dis,key=lambda k:dis[k])


def marker_rename(tcp=None, ren={}):
    mark = get_marker(tcp)
    for old, new in ren.items():
        old, new = str(old), str(new)
        if old not in mark: continue
        v = mark[old]; ps = format_pos(v['pose'])
        new = f"name={new}&type={v['key']}&floor={v['floor']}"
        new += f"&x={ps['x']}&y={ps['y']}&theta={ps['theta']}"
        tcp_cmd(tcp, f'markers/insert_by_pose?{new}')
        tcp_cmd(tcp, f'markers/delete?name={old}')


##########################################################################################
if __name__ == '__main__':
    tcp = tcp_init()
    tcp_cmd(tcp, 'move?marker=home')
    while True: tcp_cmd(tcp, show=1)

