from __future__ import absolute_import
import torch as th


def compute_gradients(opt, v):
    opt.zero_grad()
    v.backward()


def apply_gradients(opt, grads):
    return opt.step()


def MomentumOptimizer(lr, var_list=None, momentum=0.9, use_nesterov=False):
    return th.optim.SGD(var_list, lr, momentum, nesterov=use_nesterov)


def GradientDescentOptimizer(lr, var_list=None, momentum=0, use_nesterov=False):
    return th.optim.SGD(var_list, lr, momentum, nesterov=use_nesterov)
