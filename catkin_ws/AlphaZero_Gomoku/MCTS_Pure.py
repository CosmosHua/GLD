#!/usr/bin/python3
# coding: utf-8
'''
A pure implementation of the Monte Carlo Tree Search (MCTS)
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


##########################################################################################
def rollout_policy_fn(board):
    '''a coarse, fast version of policy_fn used in the rollout phase.'''
    avail = board.availables; action_probs = np.random.rand(len(avail))
    return zip(avail, action_probs) # rollout randomly


def policy_value_fn(board):
    '''a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state'''
    avail = board.availables; action_probs = np.ones(len(avail))/len(avail)
    return zip(avail, action_probs), 0 # uniform prob & 0-score for MCTS_Pure


##########################################################################################
class MCTS_Pure(object):
    '''A simple implementation of Monte Carlo Tree Search.'''
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
        action_probs, leaf_value = self._policy(state)
        win = state.is_win()
        if not win: node.expand(action_probs)
        # Evaluate the leaf node by random rollout
        leaf_value = self.evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)


    def evaluate_rollout(self, state, limit=1000):
        '''Use the rollout policy to play until the end of the game,
        return +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.'''
        player = state.player_cur
        for i in range(limit):
            win = state.is_win()
            if win: break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=lambda x:x[1])[0]
            state.do_move(max_action)
        else: # If no break from the loop, issue a warning.
            print('WARNING: rollout reached move limit')
        return 0 if win<0 else 1 if win==player else -1


    def get_move(self, state):
        '''Runs all playouts sequentially and returns the most visited action.
        state: the current game state
        Return: the selected action.'''
        for n in range(self._playout):
            state_copy = copy.deepcopy(state)
            self.playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]


    def update_with_move(self, last_move):
        '''Step forward in the tree, keeping everything we already know
        about the subtree.'''
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else: self._root = TreeNode(None, 1)


    def __str__(self): return 'MCTS_Pure'


##########################################################################################
class MCTSPlayer_Pure(object):
    '''AI player based on MCTS_Pure'''
    def __init__(self, c_puct=5, playout=2000):
        self.mcts = MCTS_Pure(policy_value_fn, c_puct, playout)


    def get_action(self, board):
        if len(board.availables) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else: print('WARNING: the board is full')


    def reset(self): self.mcts.update_with_move(-1)
    def __str__(self): return 'MCTSPlayer_Pure'


##########################################################################################

