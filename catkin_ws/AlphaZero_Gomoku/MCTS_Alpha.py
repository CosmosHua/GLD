#!/usr/bin/python3
# coding: utf-8
'''
An AlphaGo Zero style implementation of the Monte Carlo Tree Search (MCTS), which
uses a policy-value network to guide the tree search and evaluate the leaf nodes.
@author: Junxiao Song
'''

import copy
import numpy as np


##########################################################################################
class TreeNode(object):
    '''A node in the MCTS_Tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.'''
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {} # map from action to TreeNode
        self._n_visits = 0
        self._P = prior_p
        self._Q = 0
        self._u = 0


    def expand(self, action_priors):
        '''Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.'''
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)


    def select(self, c_puct):
        '''Select action among children that gives maximum action value Q plus bonus u(P).
        Return: A tuple of (action, next_node)'''
        return max(self._children.items(), key=lambda x: x[1].get_value(c_puct))


    def update(self, leaf_value):
        '''Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's perspective.'''
        self._n_visits += 1 # Count visit.
        # Update Q, a running average of values for all visits.
        self._Q += (leaf_value - self._Q) / self._n_visits


    def update_recursive(self, leaf_value):
        '''Like a call to update(), but applied recursively for all ancestors.'''
        # If it is not root, this node's parent should be updated first.
        if self._parent: self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)


    def get_value(self, c_puct):
        '''Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.'''
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1+self._n_visits)
        return self._Q + self._u


    def is_leaf(self):
        '''Check if leaf node (i.e. no nodes below this have been expanded).'''
        return self._children == {}


    def is_root(self): return self._parent is None


def softmax(x): p = np.exp(x-np.max(x)); return p/np.sum(p)
##########################################################################################
class MCTS_Alpha(object):
    '''An AlphaGo implementation of Monte Carlo Tree Search.'''
    def __init__(self, policy_value_fn, c_puct=5, playout=10000):
        '''policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.'''
        self._root = TreeNode(None, 1)
        self._policy = policy_value_fn
        self._playout = playout
        self._c_puct = c_puct


    def playout(self, state):
        '''Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.'''
        node = self._root
        while True:
            if node.is_leaf(): break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)
        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        win = state.is_win()
        if not win: node.expand(action_probs)
        else: # for end state, return the 'true' leaf_value
            if win<0: leaf_value = 0 # tie
            else: leaf_value = 1 if win==state.player_cur else -1
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)


    def get_move_probs(self, state, temp=1E-3):
        '''Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state.
        temp: temperature parameter in (0,1] controls the level of exploration.'''
        for n in range(self._playout):
            state_copy = copy.deepcopy(state)
            self.playout(state_copy)
        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(np.log(np.array(visits)+1E-100)/temp)
        return acts, act_probs


    def update_with_move(self, last_move):
        '''Step forward in the tree, keeping everything we already know
        about the subtree.'''
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else: self._root = TreeNode(None, 1)


    def __str__(self): return 'MCTS_Alpha'


##########################################################################################
class MCTSPlayer_Alpha(object):
    '''AI player based on MCTS_Alpha'''
    def __init__(self, policy_value_fn, c_puct=5, playout=2000, is_selfplay=0):
        self.mcts = MCTS_Alpha(policy_value_fn, c_puct, playout)
        self._is_selfplay = is_selfplay


    def get_action(self, board, temp=1E-3, return_prob=False):
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.w*board.h)
        if len(board.availables) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for self-play training)
                move = np.random.choice(acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))) )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1E-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
                #loc = i2yx(move, board.w)
                #print('AI move: %d,%d\n' % (loc[0], loc[1]))
            if return_prob: return move, move_probs
            else: return move
        else: print('WARNING: the board is full')


    def reset(self): self.mcts.update_with_move(-1)
    def __str__(self): return 'MCTSPlayer_Alpha'


##########################################################################################

