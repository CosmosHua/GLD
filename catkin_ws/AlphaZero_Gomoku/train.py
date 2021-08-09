#!/usr/bin/python3
# coding: utf-8
'''
An implementation of the training pipeline of AlphaZero
@author: Junxiao Song
'''

import random
import numpy as np
from glob import glob
from Gomoku import Renju, C
from MCTS_Pure import MCTSPlayer_Pure
from MCTS_Alpha import MCTSPlayer_Alpha
from collections import defaultdict, deque
from PVNet_pytorch import PolicyValueNet
import torch.multiprocessing as mp
from tqdm import trange


##########################################################################################
class TrainPipeline():
    def __init__(self, width=15, height=15, C=C, n=5, model=None):
        self.w = width; self.h = height; self.n = n
        self.renju = Renju(self.w, self.h, self.n)
        # training params
        self.c_puct = 5
        self.kl_targ = 0.02
        self.check_freq = 50
        self.learn_rate = 2E-3
        self.batch_size = 1024
        self.temp = 1 # temperature param
        self.epochs = 5 # train_steps for each update
        self.playout = 800 # simulation num for each move
        self.data_buffer = deque(maxlen=self.batch_size*100)
        self.lr_factor = 1 # adaptively adjust learning rate based on KL
        # simulation num of pure mcts: as opponent to evaluate PVNet
        self.playout_pure = 2*self.playout # initial value
        mod = glob('*.model') # train PVNet from model/scratch
        if not model and mod: model = sorted(mod)[-1] # latest
        self.pvnet = PolicyValueNet(self.w, self.h, C, model=model)
        self.mcts_player = MCTSPlayer_Alpha(self.pvnet.policy_value_fn,
            c_puct=self.c_puct, playout=self.playout, is_selfplay=1)
        self.ps = 4; mp.set_start_method('spawn') # multi-process num
        print(f'board=({self.h}*{self.w}), process={self.ps}, {model}')


    def get_equi_data(self, data):
        '''augment the data set by rotation and flipping
        data: [(state, mcts_prob, winner_z), ..., ...]'''
        extend_data = []
        for state, mcts_porb, winner in data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(self.h, self.w)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).ravel(), winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).ravel(), winner))
        return extend_data


    def collect_selfplay_data(self, N=1):
        renju = Renju(self.w, self.h, self.n) # self.renju
        winner, data = renju.self_play(self.mcts_player, self.temp)
        return self.get_equi_data(data) # augment self_play data


    def policy_update(self):
        '''update the policy-value net'''
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        states, mcts_probs, winners = list(zip(*mini_batch))
        old_probs, old_v = self.pvnet.policy_value(states)
        for i in range(self.epochs):
            loss, entropy = self.pvnet.train_step(states,
                mcts_probs, winners, self.learn_rate*self.lr_factor)
            new_probs, new_v = self.pvnet.policy_value(states)
            kl = np.log((old_probs+1E-100)/(new_probs+1E-100))
            kl = np.mean(np.sum(kl*old_probs, axis=1))
            if kl > 4*self.kl_targ: break # early stopping if D_KL diverges badly
        # adaptively adjust the learning rate based on KL
        if kl > 2*self.kl_targ and self.lr_factor > 0.1: self.lr_factor /= 1.5
        elif kl < self.kl_targ/2 and self.lr_factor < 10: self.lr_factor *= 1.5
        winners = np.array(winners)
        var_old = 1 - np.var(winners-old_v.ravel())/np.var(winners)
        var_new = 1 - np.var(winners-new_v.ravel())/np.var(winners)
        print('KL=%2.5f LR_factor=%2.3f Loss=%2.6f Entropy=%2.6f Var_old=%2.3f Var_new=%2.3f'
            %(kl, self.lr_factor, loss, entropy, var_old, var_new))
        return loss, entropy


    def policy_evaluate(self, N=10):
        '''Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training'''
        mcts_player = MCTSPlayer_Alpha(self.pvnet.policy_value_fn,
            c_puct=self.c_puct, playout=self.playout, is_selfplay=0)
        mcts_player_pure = MCTSPlayer_Pure(c_puct=5, playout=self.playout_pure)

        res = []; ps = self.ps; '''pool = mp.Pool(self.ps)
        param = [(mcts_player, mcts_player_pure, i%2) for i in range(N)]
        res = pool.starmap(Renju(self.w, self.h, self.n).play, param)
        pool.close(); pool.join() # block until pool finish'''
        for k in trange(0,N,ps):
            pool = mp.Pool(ps)
            param = [(mcts_player, mcts_player_pure, (k+i)%2) for i in range(ps)]
            rt = pool.starmap(Renju(self.w, self.h, self.n).play, param)
            pool.close(); pool.join(); res += rt

        # p1=mcts_player, p2=mcts_player_pure=p2, -1=tie/draw
        rt = [res.count(i) for i in (*self.renju.board.players,-1)]
        print('current_policy vs pure_mcts(n=%d): win=%d, lose=%d, tie=%d.'
            %(self.playout_pure, *rt))
        return (rt[0] + 0.5*rt[2])/len(res) # win_ratio


    def run(self, N=1E6): # run training
        i = best_win_ratio = 0; ps = self.ps
        while self.playout_pure<N:
            if i % ps == 0:
                pool = mp.Pool(ps)
                res = pool.map(self.collect_selfplay_data, range(ps))
                pool.close(); pool.join() # block until pool finish
                print(f'batch-{i}: rollout={[len(x) for x in res]}')
                for x in res: self.data_buffer += x # extend
            if len(self.data_buffer)<self.batch_size: continue

            loss, entropy = self.policy_update(); i += 1
            self.pvnet.save_model('./current_policy.model')
            # check the performance & save net params
            if i % self.check_freq == 0:
                win_ratio = self.policy_evaluate(max(3*ps,10))
                if win_ratio > best_win_ratio: # update best_policy
                    best_win_ratio = win_ratio; print('=> best_policy!')
                    self.pvnet.save_model('./best_policy.model')
                if best_win_ratio >= 1: # update playout_pure
                    best_win_ratio = 0; self.playout_pure += 1000


##########################################################################################
if __name__ == '__main__':
    training_pipeline = TrainPipeline(15,15)
    try: training_pipeline.run()
    except KeyboardInterrupt: print('quit')

