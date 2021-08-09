#!/usr/bin/python3
# coding: utf-8

import os, sys, cv2
from aip import AipBodyAnalysis


import base64, requests
##########################################################################################
def get_token(API_Key=None, Secret_Key=None):
    if not type(API_Key)==type(Secret_Key)==str:
        API_Key = 'K6PWqtiUTKYK1fYaz13O8E3i'
        Secret_Key = 'IDBUII1j6srF1XVNDX32I2WpuwBWczzK'
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type='+\
        f'client_credentials&client_id={API_Key}&client_secret={Secret_Key}'
    res = requests.get(host)
    if res: return res.json()['access_token'] # token


def get_gesture(src):
    #token = get_token()
    header = {'content-type': 'application/x-www-form-urlencoded'}
    token = '24.832d6446af224fc5c13f59630cce8b32.2592000.1634266079.282335-18550528'
    url = f'https://aip.baidubce.com/rest/2.0/image-classify/v1/gesture?access_token={token}'
    with open(src,'rb') as ff: im = {'image': base64.b64encode(ff.read())}
    res = requests.post(url, data=im, headers=header)
    if res: print(res.json())


##########################################################################################
def init_hand(App_ID=None, API_Key=None, Secret_Key=None):
    if not type(App_ID)==type(API_Key)==type(Secret_Key)==str:
        Secret_Key = 'IDBUII1j6srF1XVNDX32I2WpuwBWczzK'
        App_ID, API_Key = '18550528', 'K6PWqtiUTKYK1fYaz13O8E3i'
    return AipBodyAnalysis(App_ID, API_Key, Secret_Key)


def gesture(src):
    hand = init_hand(); im = cv2.imread(src)
    res = hand.gesture(im); print(res)


##########################################################################################
if __name__ == '__main__':
    gesture('./yolov5/data/images/zidane.jpg')

