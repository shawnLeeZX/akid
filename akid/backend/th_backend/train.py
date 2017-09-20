import torch as th


def compute_gradients(opt, v):
    v.backward()


def apply_gradients(opt, grads):
    return opt.step()


def MomentumOptimizer(lr, var_list=None, momentum=0.9, use_nesterov=False):
    return th.optim.SGD(var_list, lr, momentum, nesterov=use_nesterov)
