import torch
import torch.nn as nn
import datetime
import numpy as np
import os
import errno
import os.path as osp


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def minmaxscaler(data):
    output=torch.cat([data,torch.zeros(1,1)],dim=-1)
    output[...,-1] = 0-output[...,:-1].sum()
    return output


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


def defappendTensorToTensor(old_tensor, new_tensor, dim=None, new_axis=None):
    # type: (tensor, tenor, int, int) -> tensor
    if new_axis is not None:
        assert dim is None
        if old_tensor is None:
            old_tensor = new_tensor.unsqueeze(new_axis)
        else:
            old_tensor = torch.cat([old_tensor, new_tensor.unsqueeze(new_axis)], dim=new_axis)
    else:
        if old_tensor is None:
            old_tensor = new_tensor
        else:
            old_tensor = torch.cat([old_tensor, new_tensor], dim=dim)
    return old_tensor


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_checkpoint(model, fpath='checkpoint.pth.tar'):
    state = {'state_dict': model.state_dict()}
    print('Save checkpoint. {}'.format(fpath))
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)


def load_checkpoint(model, fpath):
    if osp.isfile(fpath):
        ct = torch.load(fpath)
        model.load_state_dict(ct['state_dict'])
        print("=> Loaded checkpoint '{}'".format(fpath))
        return model
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))