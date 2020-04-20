import torch
from torch.optim.optimizer import Optimizer

class SGD_Adam():
    '''
    applies SGD to the first group of parameters, Adam - to the second
    '''
    def __init__(self, sgd, adam):
        self.sgd = sgd
        self.adam = adam
        self.param_groups = sgd.param_groups + adam.param_groups

    def step(self, closure=None):
        self.sgd.step()
        self.adam.step()

    def zero_grad(self):
        self.sgd.zero_grad()
        self.adam.zero_grad()

    def state_dict(self):
        return (self.sgd.state_dict(), self.adam.state_dict())