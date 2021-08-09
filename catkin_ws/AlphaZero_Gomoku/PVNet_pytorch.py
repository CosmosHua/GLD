#!/usr/bin/python3
# coding: utf-8
'''
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0
@author: Junxiao Song
'''

import torch, os
import torch.nn as nn
import torch.optim as optim
import numpy as np


##########################################################################################
class Net(nn.Module): # policy-value network module
    def __init__(self, width, height, C):
        super(Net, self).__init__()
        self.w = width; self.h = height
        # common layers
        self.conv1 = nn.Conv2d(C, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*self.w*self.h, self.w*self.h)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*self.w*self.h, 64)
        self.val_fc2 = nn.Linear(64, 1)


    def forward(self, x):
        # common layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # action policy layers
        x_act = torch.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.w*self.h)
        x_act = torch.log_softmax(self.act_fc1(x_act), dim=-1)
        # state value layers
        x_val = torch.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.w*self.h)
        x_val = torch.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


##########################################################################################
class PolicyValueNet(): # policy-value network
    def __init__(self, width, height, C, model=None):
        print(f'PolicyValueNet: {torch.cuda.get_device_name()}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.w = width; self.h = height; l2_coef = 1E-4 # L2 penalty

        self.net = Net(self.w, self.h, C).to(self.device) # policy_value_net
        self.optimizer = optim.Adam(self.net.parameters(), weight_decay=l2_coef)

        if type(model)==str and os.path.isfile(model):
            self.net.load_state_dict(torch.load(model))


    def policy_value(self, states):
        '''input: a batch of states
        output: a batch of action probabilities and state values'''
        states = torch.Tensor(states).to(self.device)
        log_act_probs, value = self.net(states)
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        return act_probs, value.data.cpu().numpy()


    def policy_value_fn(self, board):
        '''input: board
        output: a list of (action, probability) tuples for each available
            action and the score of the board state'''
        availables = board.availables
        state = board.state()[None,...]
        state = torch.Tensor(np.ascontiguousarray(state)).to(self.device)
        log_act_probs, value = self.net(state)
        act_probs = np.exp(log_act_probs.data.cpu().numpy().ravel())
        act_probs = zip(availables, act_probs[availables])
        return act_probs, value.data[0][0]


    def train_step(self, states, mcts_probs, winners, lr):
        '''perform a training step'''
        states = torch.Tensor(states).to(self.device)
        mcts_probs = torch.Tensor(mcts_probs).to(self.device)
        winners = torch.Tensor(winners).to(self.device)

        self.optimizer.zero_grad() # zero gradients
        for x in self.optimizer.param_groups: x['lr'] = lr

        log_act_probs, value = self.net(states) # forward
        # Loss = (z-v)^2 - pi^T * log(p) + c*||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = nn.MSELoss()(value.view(-1), winners)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss

        loss.backward() # backward
        self.optimizer.step() # optimize

        # calc policy entropy, for monitoring only
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item() # for pytorch>=0.5


    def save_model(self, model):
        torch.save(self.net.state_dict(), model)


##########################################################################################

