#!/usr/bin/python3
# coding: utf-8

import os, sys, io
from aip import AipSpeech
from subprocess import call
from pydub import AudioSegment
from pydub.playback import play
from playsound import playsound
from threading import Thread

#import pyttsx3 as tts
#tts.init().setProperty('rate',160)
#tts.speak('Baidu TTS Service Fail!')

# Ref: pydub/playback.py and os.system('ffplay --help')
#os.system('sudo apt remove python3-pyaudio') # pyaudio backend
#playsound = lambda x: play(AudioSegment.from_file(x)) # prefer ffplay backend
#playsound = lambda x: os.system(f'ffplay -nodisp -autoexit -hide_banner -v quiet "{x}"')
#playsound = lambda x: call(['ffplay', '-nodisp','-autoexit','-hide_banner','-v','quiet', x])
##########################################################################################
def init_speech(App_ID=None, API_Key=None, Secret_Key=None):
    if not type(App_ID)==type(API_Key)==type(Secret_Key)==str:
        Secret_Key = 'd39ec848d016a8474c7c25e308b310c3'
        App_ID, API_Key = '8168466', 'pmUzrWcsA3Ce7RB5rSqsvQt2'
    return AipSpeech(App_ID, API_Key, Secret_Key)


speech = init_speech(); opt = {'per':4,'vol':10}
speech.setSocketTimeoutInMillis(500) # wait server respond
speech.setConnectionTimeoutInMillis(500) # connect server
# opt=dict: spd=speed(0-9), pit=pitch(0-9), vol=volume(0-15)
# per=person(0=女声，1=2=男声，3=度逍遥，4=度丫丫, default=0)
##########################################################################################
def speak(text, speech=speech, dir='TTS', opt=opt):
    # Conditional os.mkdir() will crash Thread!
    os.makedirs(dir, exist_ok=True) # NOT Alter
    text = str(text); dst = f'{dir}/{text}.mp3'
    if os.path.isfile(dst): playsound(dst); return
    if type(speech)!=AipSpeech: speech = init_speech()
    try: res = speech.synthesis(text, 'zh', 1, options=opt)
    except: print(u'TTS Network Error: '+text); return
    if type(res)!=bytes: print(u'TTS Fail: '+text); return
    with open(dst,'wb') as fp: fp.write(res); print(text)
    #play(AudioSegment.from_file(io.BytesIO(res),format='mp3'))
    playsound(dst); #play(AudioSegment.from_file(dst))


def Speak(text, speech=speech, dir='TTS', opt=opt):
    os.makedirs(dir, exist_ok=True)
    if type(speech)!=AipSpeech: speech = init_speech()
    if type(text) in (list,tuple):
        for i in text: speak(i, speech, dir, opt)
    elif type(text)==dict:
        for i in text.items(): speak(i, speech, dir, opt)
    else: speak(text, speech, dir, opt)


lang = {'en':{'dev_pid':1737}, 'zh':{'dev_pid':1537},
        'ct':{'dev_pid':1637}, 'sc':{'dev_pid':1837}}
##########################################################################################
def ASR(voc, speech=speech, fmt='wav', opt=lang['zh']):
    if type(voc)==str and os.path.isfile(voc):
        if voc[-3:] not in ('pcm','wav','arm'):
            seg = AudioSegment.from_file(voc)
            voc = voc[:-3]+'wav' # convert
            seg.export(voc, format='wav')
        with open(voc,'rb') as f: seg = f.read()
        fmt = voc[-3:]
    else: return # TODO: deal with MIC stream
    if type(speech)!=AipSpeech: speech = init_speech()
    res = speech.asr(voc, fmt, 16000, options=opt)
    return res['result'] if 'result' in res else None


speak_ = lambda x: Thread(target=speak, args=(x,), daemon=True).start()
Speak_ = lambda x: Thread(target=Speak, args=(x,), daemon=True).start()
##########################################################################################
if __name__ == '__main__':
    speak('Hello', init_speech())

