#!/usr/bin/python3
# coding: utf-8

import numpy as np


C = 4 # state dim
i2yx = lambda i,w: (i//w, i%w) # y,x
yx2i = lambda x,w: x[0]*w + x[1] # 0-index
cvt = lambda x: str(np.asarray(x,int)).strip('[]')
##########################################################################################
def line_win(x, val, K=5):
    x = str(np.asarray(x,int).ravel())
    return cvt([val]*K) in x


##########################################################################################
class Board(object): # board for the Renju
    def __init__(self, width=15, height=15, **kwargs):
        self.w, self.h = width, height
        # how many pieces in a row to win
        self.n = int(kwargs.get('n', 5))
        self.players = [1,2]; self.init_board()


    def init_board(self, start_player=0):
        assert self.w > self.n and self.h > self.n
        self.player_cur = self.players[start_player]
        # keep available moves in a list
        self.availables = list(range(self.w * self.h))
        # seq: {move location : player type}
        self.seq = {}; self.last_move = -1
        self.bd = np.zeros((self.h, self.w))


    def do_move(self, move):
        self.seq[move] = self.player_cur
        self.bd[i2yx(move, self.w)] = self.player_cur
        self.last_move = move; self.availables.remove(move)
        self.player_cur = sum(self.players)-self.player_cur


    def is_end(self): return self.is_win()
    def is_win(self):
        bd = self.bd; H,W = bd.shape; bd2 = np.fliplr(bd); K = self.n
        # 0=continue, player=end+winner, -1=end+tie
        if len(self.availables)<1: return -1 # end+tie
        elif len(self.seq)<2*K-1: return 0 # continue
        if len(self.seq)<3*(W+H)-2: # scan all moves
            for move, player in self.seq.items():
                y, x = i2yx(move, self.w) # max(seq)=H*W
                if line_win(bd[y,:], player, K): return player # vertical
                if line_win(bd[:,x], player, K): return player # horizontal
                if line_win(bd.diagonal(x-y), player, K): return player # diagonal
                if line_win(np.diag(bd2,(W-1-x)-y), player, K): return player # anti-diag
        else: # scan all lines: 3*(W+H)-2
            player = sum(self.players)-self.player_cur
            for i in range(W): # crosswise: W+H
                if line_win(bd[:,i], player, K): return player # vertical
            for i in range(H): # crosswise: W+H
                if line_win(bd[i,:], player, K): return player # horizontal
            for i in range(1-H,W): # oblique: 2*(W+H-1)
                if line_win(bd.diagonal(i), player, K): return player # diagonal
                if line_win(np.diag(bd2,i), player, K): return player # anti-diag
        return -1 if len(self.availables)<1 else 0 # end+tie or continue


    def show(self, k=2):
        p1, p2 = self.players; wd = self.w
        mk = {p1: 'X', p2: 'O', 0:'-'}
        print(f'Player1={mk[p1]}, Player2={mk[p2]}')
        for i in range(self.h, -1, -1):
            print(f'%{k}s'%(i if i>0 else ''), end=' ')
            for j in range(wd):
                m = self.seq.get((i-1)*wd+j, 0)
                m = chr(j+65) if i==0 else mk[m]
                print(m.center(k), end='')
            print()


    def state(self): # the input of policy_value_net
        '''return the board state from the current player view'''
        stat = np.zeros((C, self.h, self.w))
        if self.seq:
            moves, players = np.array(list(zip(*self.seq.items())))
            move_cur = moves[players == self.player_cur]
            move_opp = moves[players != self.player_cur]
            stat[0][i2yx(move_cur, self.w)] = 1
            stat[1][i2yx(move_opp, self.w)] = 1
            stat[2][i2yx(self.last_move, self.w)] = 1 # last loc
        if len(self.seq)%2==0: stat[3][:] = 1 # who to play
        return stat[:, ::-1] # flip height(axis=1)


##########################################################################################
class Renju(object): # Renju server
    def __init__(self, width=15, height=15, n=5, **kwargs):
        self.board = Board(width, height, n=n, **kwargs)


    def play(self, player1, player2, start_player=0, show=0):
        '''start a Renju between two players'''
        assert start_player in (0,1), 'start_player != 0/1'
        self.board.init_board(start_player)
        p1, p2 = self.board.players; win = 0
        players = {p1: player1, p2: player2}
        while not win: # keep playing
            player = players[self.board.player_cur]
            move = player.get_action(self.board)
            self.board.do_move(move) # perform a move
            if show: self.board.show()
            win = self.board.is_win()
        if show: print('Tie' if win<0 else f'Player{win} Win')
        return win


    def self_play(self, player, temp=1E-3, show=0):
        ''' start a self-play Renju using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training'''
        states, mcts_probs, seq_player = [], [], []; win = 0
        self.board.init_board(); p1, p2 = self.board.players
        while not win: # keep playing, temp=temperature
            move, move_probs = player.get_action(self.board, temp, True)
            states.append(self.board.state())
            mcts_probs.append(move_probs)
            seq_player.append(self.board.player_cur)
            self.board.do_move(move) # perform a move
            if show: self.board.show()
            win = self.board.is_win()
        seq_player = np.array(seq_player)
        seq_winner = np.zeros(seq_player.shape)
        if win>0: # winner: from current player view
            seq_winner[seq_player==win] = 1
            seq_winner[seq_player!=win] = -1
        player.reset() # reset MCTS root node
        if show: print('Tie' if win<0 else f'Player{win} Win')
        return win, list(zip(states, mcts_probs, seq_winner))


##########################################################################################

