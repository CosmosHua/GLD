#!/usr/bin/python3
# coding: utf-8

import os, pygame
import time as tm
import numpy as np
from glob import glob


xy2pt = lambda x,p: np.asarray(x)*p[1]+p[2]
xy2mk = lambda x: chr(x[0]+65) + str(x[1]+1) # 'H8'
mk2xy = lambda m: (ord(m[0])-65, int(m[1:])-1) # x,y
get_surf = lambda sz: pygame.Surface(sz, pygame.SRCALPHA)
def draw(win,sf,pt=(0,0)): win.blit(sf,pt); pygame.display.update()
##########################################################################################
def pt2xy(x, p): # p=(N,sp,mg); x.clip(0,N-1)
    x = ((np.asarray(x)-p[2])/p[1]).round().astype(int)
    if ((x>=0)&(x<p[0])).all(): return tuple(x)


def blit_txt(win, txt, pt, fs, fc, font=None, ct=(1,1)):
    if font is None or type(font)==str:
        font = pygame.font.SysFont(font, fs)
    txt = str(txt); ret = font.render(txt, True, fc)
    fs, pt, ct = np.array([font.size(txt), pt, ct])
    a, b = np.where(ct>1), np.where(ct==1)
    pt[a] = pt[a]-fs[a]; pt[b] = pt[b]-fs[b]/2
    win.blit(ret, pt); return ret


BLACK,WHITE = 1,2; ST = {0:'Long', 1:'Win', 2:'Four', 3:'Live3'}
COLOR = {BLACK: (0,0,0), WHITE: (255,255,255), 0: (125,125,125)}
##########################################################################################
class Renju(object):
    def __init__(self, N=15, sp=30):
        mg = sp + round(min(sp*0.15,7.5))
        self.param = N, sp, mg # mg: margin
        sz = sp*(N-1) + 2*mg; self.sz = (sz,sz)
        pygame.display.set_caption('Renju @ Joshua')
        self.win = pygame.display.set_mode(self.sz)
        self.init_board(); self.reset() # initialize


    def reset(self): # initialize
        N = self.param[0]; draw(self.win, self.bg)
        self.board = np.zeros((N,N), dtype=int)
        self.chess = get_surf(self.sz); self.seq = []


    ########################################################
    def init_board(self, bg='Renju.jpg'):
        if not os.path.isfile(bg):
            bg = get_surf(self.sz) # fill color
            bg.fill(color=(155,111,55,255)) # RGBA
        else: # load image: opaque->transparent
            bg = pygame.image.load(bg).convert_alpha()
            bg = pygame.transform.scale(bg, self.sz)
        self.bg = bg; #bg.set_alpha(255) # opaque

        N, sp, mg = self.param; color = (0,0,0)
        low, high = xy2pt([0, N-1], self.param)
        for i in range(N): # draw coordinates & labels
            k, vm, hm = xy2pt(i, self.param), str(i+1), chr(i+65)
            pygame.draw.line(bg, color, [low,k], [high,k], 1)
            pygame.draw.line(bg, color, [k,low], [k,high], 1)
            blit_txt(bg, vm, [low-mg*0.45,k], int(sp*0.7), color)
            blit_txt(bg, hm, [k,low-mg*0.35], int(sp*0.6), color)
        low, ct, high = xy2pt([3, N//2, N-4], self.param) # markers
        pygame.draw.circle(bg, color, [ct,ct], 4, 0)
        pygame.draw.circle(bg, color, [low,low], 4, 0)
        pygame.draw.circle(bg, color, [low,high], 4, 0)
        pygame.draw.circle(bg, color, [high,low], 4, 0)
        pygame.draw.circle(bg, color, [high,high], 4, 0)
        d = 4; wh = sp*(N-1) + 2*d + 1 # draw frame->pretty
        pygame.draw.rect(bg, color, [mg-d, mg-d, wh, wh], 3)


    ########################################################
    def in_seq(self): # whether: seq in seq2
        m1 = ''.join([m for i,m,v,s in self.seq])
        m2 = ''.join([m for i,m,v,s in self.seq2[:len(self.seq)]])
        return m1==m2 # seq=current moves: return m1 in m2


    def retract(self): # retract 1-move
        x,y = mk2xy(self.seq[-1][1]); draw(self.win, self.bg)
        self.board[y,x] = 0; self.seq.pop(); self.draw_moves(0)


    def backward(self): # backward 1-move
        if len(self.seq)<1: return
        if not ('seq2' in dir(self) and self.in_seq()):
            self.seq2 = self.seq.copy() # init/update
        self.retract() # retract 1-move


    def forward(self): # forward 1-move
        if 'seq2' in dir(self) and self.in_seq() and \
            len(self.seq2)>len(self.seq): # Ref: move()
            i,m,v,s = sq = self.seq2[len(self.seq)]
            x,y = mk2xy(m); self.board[y,x] = v
            self.seq.append(sq); self.draw_moves(-1)


    def to_first(self): # backward all
        if len(self.seq)<1: return
        if not ('seq2' in dir(self) and self.in_seq()):
            self.seq2 = self.seq.copy() # init/update
        self.reset() # retract all moves


    def to_last(self): # forward all
        if 'seq2' in dir(self) and self.in_seq():
            n = len(self.seq); self.seq += self.seq2[n:]
            self.draw_moves(n) # self.seq = self.seq2.copy()
            for i,m,v,s in self.seq[n:]: self.board[mk2xy(m)[::-1]] = v


    ########################################################
    def move(self, pos): # seq=[i,m,v,s]
        bd = self.board; xy = pt2xy(pos, self.param)
        if xy is None or bd[xy[::-1]]!=0: return
        x,y = xy; m = xy2mk(xy); seq = self.seq; i = len(seq)+1
        v = bd[y,x] = BLACK if i%2 else WHITE; seq.append([i,m,v,0])
        find_seq(find1, bd, seq); self.draw_moves(-1) # update seq


    def draw_moves(self, a=0, b=None):
        n = len(self.seq); b = b if b else n
        fs = lambda k,d=3,o=0: round((sp-k*d-o)*0.95)
        if a==0 or b<n: self.chess = get_surf(self.sz)
        sp = self.param[1]; r = sp/2-1; T = BLACK+WHITE
        for i,m,v,s in self.seq[a:b]:
            pos = xy2pt(mk2xy(m), self.param)
            pygame.draw.circle(self.chess, COLOR[v], pos, r, 0)
            pygame.draw.circle(self.chess, (125,)*3, pos, r, 1)
            blit_txt(self.chess, i, pos, fs(len(str(i))), COLOR[T-v])
        if 'pos' in dir(): self.hover(pos); #print('%3d:%4s'%(i,m))
        draw(self.win, self.chess) # also for: self.seq==[]


    ########################################################
    def hover(self, pos, alpha=111): # pre-show move
        v = WHITE if len(self.seq)%2 else BLACK
        sp = self.param[1]; r = sp/2; win = self.win
        pre = get_surf((sp,sp)); pre.set_alpha(alpha)
        pygame.draw.circle(pre, COLOR[v], (r,r), r-1, 0)
        pygame.draw.circle(pre, (125,)*3, (r,r), r-1, 1)
        win.blit(self.bg, (0,0)); win.blit(self.chess, (0,0))
        self.blit_stat(); draw(win, pre, np.array(pos)-r)


    ########################################################
    def blit_stat(self): # see: find_seq(find1)
        if len(self.seq)>4: b,w,v = self.state()
        N, sp, mg = self.param; sz = self.sz[0]
        ht = sp*0.9; stat = get_surf((sz,ht))
        if 'b' in dir() and v!=0: # whole
            pt = [(N//2)*sp+mg, ht/2]
            fc = COLOR[BLACK if v%2 else WHITE]
            txt = '%d: %s'%(abs(v), ST[1-(v<0)])
            blit_txt(stat, txt, pt, int(ht), fc)
        for i,m,v,s in self.seq[-2:]: # black/white
            if 'b' in dir(): s = b if i%2 else w
            s = ST[(s>0).argmax()] if (s>0).any() else ''
            txt, fc = '%d: %s %s'%(i,m,s), COLOR[v]
            ct = 0 if i%2 else 2; fs = int(sp*0.7)
            pt = [(0 if i%2 else N-1)*sp+mg, ht/2]
            blit_txt(stat, txt, pt, fs, fc, ct=(ct,1))
        self.win.blit(stat, (0,sz-ht+1))


    def state(self): # see: find_seq(find1)
        st = np.array([s[-1] for s in self.seq])
        b, w = st[::2], st[1::2] # st=(n,C)
        # (kp,k0) use first>0; (k1,k2,..) use last
        kp, k0 = np.argmax(b[:,:2]>0, axis=0) # for black
        b[-1,0] = 2*kp if b[kp,0]>0 else -1 # kp prior to k0
        b[-1,1] = 0 if b[kp,0]>0 else 2*k0 if b[k0,1]>0 else -1
        kp, k0 = np.argmax(w[:,:2]>0, axis=0) # for white
        w[-1,0] = 2*kp+1 if w[kp,0]>0 else -1 # kp prior to k0
        w[-1,1] = 0 if w[kp,0]>0 else 2*k0+1 if w[k0,1]>0 else -1
        b, w = b[-1], w[-1] # for whole: v=1_idx
        if b[1]==w[1]==0: v = -min(b[0],w[0])-1 # 2-long
        #elif b[1]+w[1]==max(b[1],w[1]): # 1-long, 1-win
        #    long, win = max(b[0],w[0]), max(b[1],w[1])
        elif b[1]*w[1]==0: v = -max(b[0],w[0])-1 # 1-long
        elif b[1]*w[1]<0: v = max(b[1],w[1])+1 # 1-win
        elif b[1]!=w[1]: v = min(b[1],w[1])+1 # 2-win
        else: v = 0 # b[1]==w[1]==-1: not win/long
        self.bwv = (b,w,v); return self.bwv
        # win(v>0), long(v<0), none(v=0)


    ########################################################
    def is_end(self): # find2: (BLACK,WHITE)
        b,w = find_all(find2, self.board)
        b = b[1] if hasattr(b,'len') else b
        w = w[1] if hasattr(w,'len') else w
        return BLACK if b>0 else WHITE if w>0 else 0


    def about(self):
        sz = self.sz[0]; fc = (0,0,255)
        sp = self.param[1]; fs = int(sp*0.75)
        txt = ['Version: 1.0', 'Date: 2021-08-05',
            'Author: Joshua', 'cosmoscosmos@163.com']
        w,h = wh = np.array([sz*0.55, sz*0.32]).round()
        info = get_surf((w,h)); info.fill((233,)*4)
        pygame.draw.rect(info, fc, [0,0,w,h], 1)
        pygame.draw.line(info, fc, [0,sp],[w,sp], 1)
        blit_txt(info, 'About', [w/2, sp/2+1], fs, fc)
        for i,x in enumerate(txt):
            mg = sp/2; dh = (h-sp-2*mg)/len(txt)
            blit_txt(info, x, [w/2, sp+mg+(i+0.5)*dh], fs, fc)
        draw(self.win, info, ((sz-wh)/2+1).round())


    def capture(self, dir='.'): # screenshot
        os.makedirs(dir, exist_ok=True)
        tt = tm.strftime('%Y%m%d_%H%M%S', tm.localtime())
        pygame.image.save(self.win, f'{dir}/{tt}.jpg')


    def save(self, dir='.', cap=True):
        os.makedirs(dir, exist_ok=True)
        tt = tm.strftime('%Y%m%d_%H%M%S', tm.localtime())
        if cap: pygame.image.save(self.win, f'{dir}/{tt}.jpg')
        seq = dict(zip(list('imvs'), zip(*self.seq)))
        for k in seq: seq[k] = np.asarray(seq[k])
        np.savez(f'{dir}/{tt}', bd=self.board, **seq)


    ########################################################
    def load(self, dir='.'):
        rec = sorted(glob(f'{dir}/*.npz'))
        if not rec: print('No Records!'); return

        eg = os.path.basename(rec[0])[:-4]
        mu = Menu(self.win, self.param[1]*0.7, eg)
        k = mu(rec, [self.bg,self.chess])
        if k==None: return

        print('load: [%3d] %s'%(k,rec[k]))
        with np.load(rec[k]) as bs:
            self.board = bs['bd'] # load npz
            self.seq = list(zip(*[bs[i] for i in list('imvs')]))
        self.draw_moves(0)


##########################################################################################
class Menu(object):
    def __init__(self, win, fs, eg, font=None):
        fs = self.fs = int(round(fs))
        if font is None or type(font)==str:
            font = pygame.font.SysFont(font, fs)
        sz = self.sz = np.array(win.get_size())
        self.fc, self.bg = (0,0,255), (222,)*4
        self.font = font; self.win = win

        fs = np.array(font.size(f' {eg} '))
        mg = max(4,fs[0]/50), max(4,fs[1]/5)
        dd = fs+mg; self.n = sz//dd # nw,nh
        dw, dh = self.d = (dd+sz/self.n)/2


    def get_wh(self, N):
        assert self.n.prod()>=N
        nw, nh = self.n; dw, dh = self.d
        nc = min(nw, np.ceil((N*dh/dw)**0.5))
        nr = min(nh, np.ceil(N/nc))
        return [nc,nr], self.d*[nc,nr]


    def show_menu(self, src, pre=[]):
        (nc, nr), wh = self.get_wh(len(src))
        self.surf = surf = get_surf(wh); surf.fill(self.bg)
        for i,x in enumerate(src): # column first
            x = os.path.basename(x)[:-4]
            pos = [(i//nr)+0.5, (i%nr)+0.5]*self.d
            blit_txt(surf, x, pos, self.fs, self.fc, self.font)
        for s in pre: self.win.blit(s,(0,0))
        draw(self.win, surf, (self.sz-wh)/2)


    def pos2n(self, src, pos): # pos=(x,y)
        (nc, nr), wh = self.get_wh(len(src))
        x,y = xy = (pos-(self.sz-wh)/2)//self.d
        if ((xy>=0)&(xy<[nc,nr])).all(): # column first
            if 0<=(x*nr+y)<len(src): return int(x*nr+y)


    def hover(self, src, pos, pre): # pos=(x,y)
        (nc, nr), wh = self.get_wh(len(src))
        x,y = xy = (pos-(self.sz-wh)/2)//self.d
        if ((xy>=0)&(xy<[nc,nr])).all():
            x,y = xy*self.d; surf = self.surf.copy()
            pygame.draw.rect(surf, self.fc, [x,y,*self.d], 1)
            for s in pre: self.win.blit(s,(0,0))
            draw(self.win, surf, (self.sz-wh)/2)


    def __call__(self, src, pre=[]):
        N = len(src); n = self.n.prod(dtype=int)
        src = [src[i:i+n] for i in range(0,N,n)]
        k = 0; self.show_menu(src[k])
        while True:
            for e in pygame.event.get():
                if e.type==pygame.KEYDOWN:
                    if e.key==pygame.K_ESCAPE: return
                    elif e.key==pygame.K_1: return -1
                    elif e.key==pygame.K_0: return 0
                    elif e.key==pygame.K_PAGEUP:
                        k = max(k-1, 0)
                        self.show_menu(src[k], pre); break
                    elif e.key==pygame.K_PAGEDOWN:
                        k = min(k+1, len(src)-1)
                        self.show_menu(src[k], pre); break
                elif e.type==pygame.MOUSEBUTTONDOWN:
                    p = self.pos2n(src[k], e.pos)
                    if e.button==1 and p!=None: return k*n+p
                elif e.type==pygame.MOUSEMOTION:
                    self.hover(src[k], e.pos, pre)


cvt = lambda x: str(np.asarray(x,int)).strip('[]')
find2 = lambda x,k=5: (find1(x,BLACK,k), find1(x,WHITE,k))
##########################################################################################
def find1(x, v=BLACK, K=5): # once
    x = str(np.asarray(x,int).ravel())
    kp = x.find(cvt([v]*(K+1))) # K+1
    k0 = x.find(cvt([v]*K)) # K

    s = v*np.ones((K,K),int) # K-1
    s[np.diag_indices_from(s)] = 0
    k1 = max([x.find(cvt(i)) for i in s])

    s = np.zeros((K-1,K+1),int) # Live(K-2)
    t = s[:,1:-1]; t[:] = v # to form Live(K-1)
    t[np.diag_indices_from(t)] = 0 # (K-1,K-1)
    k2 = max([x.find(cvt(i)) for i in s])
    return (kp, k0, k1, k2) # C=4: {-1,1,..}


########################################################
def find_seq(fun, bd, seq, K=5): # fun: 1-version, (4;C)=>(C)
    if len(seq)<1: return 0 # seq=[i,m,v,s], [crosswise, oblique]
    i,m,v,s = seq[-1]; x,y = mk2xy(m); bd2,N = np.fliplr(bd),len(bd)
    CO = bd[y,:], bd[:,x], bd.diagonal(x-y), np.diag(bd2,(N-1-x)-y)
    seq[-1][-1] = np.array([fun(p,v,K) for p in CO]).max(axis=0)


def find_all(fun, bd, K=5): # fun: 2-version
    bd2, N = np.fliplr(bd), len(bd); st = []
    for i in range(N): # crosswise: 2*N
        st.append(fun(bd[:,i],K)) # vertical
        st.append(fun(bd[i,:],K)) # horizontal
    for i in range(1-N,N): # oblique: 2*(2*N-1)
        st.append(fun(bd.diagonal(i),K)) # diagonal
        st.append(fun(np.diag(bd2,i),K)) # anti-diag
    return np.array(st).max(axis=0) # (6*N-2; 2,C)


##########################################################################################
def play_renju(dir='renju', N=15, sp=30):
    pygame.init(); renju = Renju(N, sp)
    while True:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: return
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_ESCAPE: return
                elif e.key==pygame.K_UP: renju.to_last()
                elif e.key==pygame.K_DOWN: renju.to_first()
                elif e.key==pygame.K_LEFT: renju.backward()
                elif e.key==pygame.K_RIGHT: renju.forward()
                elif e.key==pygame.K_c: renju.capture(dir)
                elif e.key==pygame.K_s: renju.save(dir)
                elif e.key==pygame.K_l: renju.load(dir)
                elif e.key==pygame.K_n: renju.reset()
                elif e.key==pygame.K_a: renju.about()
            elif e.type==pygame.MOUSEBUTTONDOWN:
                if e.button==1: renju.move(e.pos) # left_key
                elif e.button==3: renju.backward() # right_key
            elif e.type==pygame.MOUSEMOTION: renju.hover(e.pos)
            #if pygame.mouse.get_focused():
            #   renju.hover(pygame.mouse.get_pos())
        #if renju.is_end()!=0: renju.reset()
    pygame.quit() # uninitialize pygame


##########################################################################################
if __name__ == '__main__':
    play_renju(dir='renju', N=15, sp=35)

