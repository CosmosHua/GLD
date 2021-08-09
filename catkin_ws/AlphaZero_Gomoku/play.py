#!/usr/bin/python3
# coding: utf-8

import os, sys, pickle
from Gomoku import Renju, yx2i, C
from MCTS_Pure import MCTSPlayer_Pure
from MCTS_Alpha import MCTSPlayer_Alpha
from PVNet_numpy import PolicyValueNetNumpy
from PVNet_pytorch import PolicyValueNet


##########################################################################################
class HumanPlayer(object):
    def __init__(self): pass
    def __str__(self): return 'HumanPlayer'


    def get_action(self, board):
        while True:
            try:
                x = input('Your move: ').strip().upper()
                if x[-1].isalpha(): h,w = x[:-1], x[-1] # 10H
                elif x[0].isalpha(): h,w = x[1:], x[0] # H10
                move = yx2i((int(h)-1, ord(w)-65), board.w)
            except KeyboardInterrupt: print('quit')
            except: move = -1
            if move in board.availables: return move
            else: print('=> Invalid move:', x)


##########################################################################################
def Human_AI(model, AI_first, w=15, h=15, C=C, n=5):
    renju = Renju(w, h, n); human = HumanPlayer()
    npo = 800; print(f'load: {model}, playout={npo}')
    
    # load the trained policy_value_net in either PyTorch/TensorFlow
    best_policy = PolicyValueNet(w, h, C, model=model)
    # load the provided model into a MCTS_Alpha_Player written in pure numpy
    '''policy_param = pickle.load(open(model, 'rb'), encoding='bytes')
    best_policy = PolicyValueNetNumpy(w, h, policy_param)'''

    # set larger playout for better performance
    mcts_player = MCTSPlayer_Alpha(best_policy.policy_value_fn, c_puct=5, playout=npo)
    # play with pure MCTS (much weaker even with a larger playout)
    #mcts_player = MCTSPlayer_Pure(c_puct=5, playout=npo*5)

    if not AI_first: renju.board.show() # human first
    renju.play(human, mcts_player, AI_first+0, show=1)


##########################################################################################
if __name__ == '__main__':
    from glob import glob
    os.chdir((os.path.dirname(os.path.abspath(__file__))))
    model = sorted(glob('**.model'))[0]
    Human_AI(model, False, w=15, h=15, n=5)

