#!/usr/bin/python3
# coding: utf-8

import os, re
import requests, json


URL = 'http://localhost:8000/api'
##########################################################################################
def init(user='hua', pswd='123', url=URL):
    auth = {'username':user, 'password':pswd}
    token = requests.post(f'{url}/token-auth/', data=auth).json()['token']
    return token, {'Authorization': f'JWT {token}'}


TOKEN, AUTH = init('hua', '123', URL)
##########################################################################################
def project(opt):
    url = f'{URL}/projects'; res = {}
    for k,v in opt.items(): # opt=dict
        if k in ('create','add','new'): # create a project with v=name
            res[k] = requests.post(f'{url}/', headers=AUTH, data={'name':v}).json()
        elif k in ('patch','update'): # update a project with v=pid
            res[k] = requests.patch(f'{url}/{v}/', headers=AUTH).json()
        elif k in ('delete','remove'): # remove a project with v=pid
            res[k] = requests.delete(f'{url}/{v}/', headers=AUTH).json()
        elif k in ('get','list'): # list projects with v=filter
            v = f'{v}/' if type(v)!=str else f'?{v}'
            res[k] = requests.get(f'{url}/{v}', headers=AUTH).json()
    return res


##########################################################################################
def get_img(src, ext=['jpg','png']):
    ext = '|'.join(['.*\.'+i for i in (ext if type(ext)!=str else [ext])])
    img = [src+'/'+i for i in os.listdir(src) if re.match(ext, i.lower())]
    return [('images', open(src+'/'+i,'rb')) for i in img]


def task_info(pid, tid, itm):
    rt = requests.get(f'{URL}/projects/{pid}/tasks/', headers=AUTH).json()
    rt = [i[itm] for i in rt if i['id']==tid]; return rt[0] if rt else None


def task(pid, opt):
    url = f'{URL}/projects/{pid}/tasks'; res = {}
    for k,v in opt.items(): # opt=dict
        if k in ('create','add','new'): # create a task with v=(src,ext)
            img = get_img(*v); data = {'name':os.path.basename(v), 'options':[]}
            res[k] = requests.post(f'{url}/', headers=AUTH, files=img, data=data).json()
        elif k in ('patch','update'): # update a task with v=tid
            res[k] = requests.patch(f'{url}/{v}/', headers=AUTH).json()
        elif k in ('import','load'): # import a task with v=zip
            #zip = {'url':(v, open(v,'rb'), 'application/x-www-form-urlencoded')}
            zip = {'filename':(os.path.basename(v), open(v,'rb'), 'application/zip')}
            res[k] = requests.post(f'{url}/import', headers=AUTH, files=zip).json()
        elif k in ('get','list'): # list all tasks, v useless
            res[k] = requests.get(f'{url}/', headers=AUTH).json()
        elif k in ('asset','download'): # download assets with v=(tid,dst,asset)
            dst, asset = os.path.expanduser('~/Desktop'), 'all.zip'
            if type(v) in (list,tuple) and len(v)>2: v, dst, asset = v[:3]
            ret = requests.get(f'{url}/{v}/download/{asset}', headers=AUTH)
            res[k] = dst = dst + f'/{task_info(pid,v,"name")}-{asset}'
            with open(dst,'wb') as ff: ff.write(ret.content)
        elif k in ('cancel','stop'): # cancel a task with v=tid
            res[k] = requests.post(f'{url}/{v}/cancel/', headers=AUTH).json()
        elif k in ('delete','remove'): # remove a task with v=tid
            res[k] = requests.post(f'{url}/{v}/delete/', headers=AUTH).json()
        elif k in ('restart','start'): # restart a task with v=tid
            res[k] = requests.post(f'{url}/{v}/restart/', headers=AUTH).json()
    return res


##########################################################################################
def ps_node(opt):
    url = f'{URL}/processingnodes'; res = {}
    for k,v in opt.items(): # opt=dict
        if k in ('create','add','new'): # create a node with v=(name,port)
            data = {'hostname':v[0], 'port':v[1]}
            res[k] = requests.post(f'{url}/', headers=AUTH, data=data).json()
        elif k in ('patch','update'): # update a project with v=pid
            res[k] = requests.patch(f'{url}/{v}/', headers=AUTH).json()
        elif k in ('delete','remove'): # remove a project with v=pid
            res[k] = requests.delete(f'{url}/{v}/', headers=AUTH).json()
        elif k in ('get','list'): # list projects with v=filter
            v = f'{v}/' if type(v)!=str else f'?{v}'
            res[k] = requests.get(f'{url}/{v}', headers=AUTH).json()
    return res


##########################################################################################
if __name__ == '__main__':
    #ret = project({'get':'ordering=id'})
    ret = task(8, {'asset':'fd24d8b9-4d7c-4e94-91c3-86599c672f77'})
    print(json.dumps(ret, indent=2))

