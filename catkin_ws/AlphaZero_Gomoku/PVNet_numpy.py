#!/usr/bin/python3
# coding: utf-8
'''
Implement the policy value network using numpy, so that we can play with the
trained AI model without installing any DL framwork
@author: Junxiao Song
'''

import numpy as np


# some utility functions
##########################################################################################
def relu(X): return np.maximum(X, 0)
def fc_forward(X, W, b): return np.dot(X,W) + b
def softmax(x): p = np.exp(x-np.max(x)); return p/np.sum(p)


def conv_forward(X, W, b, stride=1, padding=1):
    n_filters, d_filter, h_filter, w_filter = W.shape
    # theano conv2d flips the filters (rotate 180 degree) first
    # while doing the calculation
    W = W[:, :, ::-1, ::-1]
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    h_out, w_out = int(h_out), int(w_out)
    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)
    out = (np.dot(W_col, X_col).T + b).T
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)
    return out


def get_im2col_indices(x_shape, height, width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - height) % stride == 0
    assert (W + 2 * padding - height) % stride == 0
    out_height = int((H + 2 * padding - height) / stride + 1)
    out_width = int((W + 2 * padding - width) / stride + 1)

    i0 = np.repeat(np.arange(height), width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(width), height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), height * width).reshape(-1, 1)
    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, height, width, padding=1, stride=1):
    ''' An implementation of im2col based on some fancy indexing '''
    p = padding # Zero-pad the input
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, height, width, padding, stride)
    cols = x_padded[:, k, i, j]; C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(height * width * C, -1)
    return cols


##########################################################################################
class PolicyValueNetNumpy():
    '''policy-value network in numpy '''
    def __init__(self, width, height, net_params):
        self.params = net_params
        self.w = width; self.h = height


    def policy_value_fn(self, board):
        '''
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        '''
        X = board.state()[None,...]
        availables = board.availables
        # first 3 conv layers with ReLu nonlinearity
        for i in [0,2,4]: X = relu(conv_forward(X, self.params[i], self.params[i+1]))
        # policy head
        X_p = relu(conv_forward(X, self.params[6], self.params[7], padding=0))
        X_p = fc_forward(X_p.ravel(), self.params[8], self.params[9])
        act_probs = softmax(X_p).ravel()
        # value head
        X_v = relu(conv_forward(X, self.params[10], self.params[11], padding=0))
        X_v = relu(fc_forward(X_v.ravel(), self.params[12], self.params[13]))
        value = np.tanh(fc_forward(X_v, self.params[14], self.params[15]))[0]
        act_probs = zip(availables, act_probs[availables])
        return act_probs, value


##########################################################################################

