import os
import collections
import random

import torch


def rename_files(directory, from_str, to_str):
    for root, _, files in os.walk(directory):
        for file in files:
            if from_str in file:
                old_file_path = os.path.join(root, file)
                new_file_name = file.replace(from_str, to_str)
                new_file_path = os.path.join(root, new_file_name)
                # print(f'renaming {old_file_path} to {new_file_path}')
                os.rename(old_file_path, new_file_path)


def wrapped_getattr(self, name, default=None, wrapped_member_name='model'):
    ''' 
    should be called from wrappers of model classes such as ClassifierFreeSampleModel
    '''
    if isinstance(self, torch.nn.Module):
        # for descendants of nn.Module, 
        # name may be in self.__dict__[_parameters/_buffers/_modules] 
        # so we activate nn.Module.__getattr__ first.
        # otherwise, we might encounter an infinite loop
        try:
            attr = torch.nn.Module.__getattr__(self, name)
        except AttributeError:
            wrapped_member = torch.nn.Module.__getattr__(self, wrapped_member_name)
            attr = getattr(wrapped_member, name, default)
    else:
        # the easy case, where self is not derived from nn.Module
        wrapped_member = getattr(self, wrapped_member_name)
        attr = getattr(wrapped_member, name, default)
    return attr


def recursive_op2(x, y, op):

    assert type(x) == type(y)

    if isinstance(x, collections.Mapping):
        assert x.keys() == y.keys()
        return {k: recursive_op2(v1, v2, op) for (k, v1), v2 in zip(x.items(), y.values())}

    elif isinstance(x, collections.Sequence) and not isinstance(x, str):
        Warning('recursive_op2 on a sequence has never been tested')
        return [recursive_op2(v1, v2, op) for v1, v2 in zip(x, y)]

    elif isinstance(x, tuple):
        Warning('recursive_op2 on a tuple has never been tested')
        return tuple([recursive_op2(v1, v2, op) for v1, v2 in zip(x, y)])

    else:
        return op(x, y)


def recursive_op1(x, op, **kwargs):

    if isinstance(x, collections.Mapping):
        return {k: recursive_op1(v, op, **kwargs) for (k, v) in x.items()}

    elif isinstance(x, collections.Sequence) and not isinstance(x, str):
        return [recursive_op1(v, op, **kwargs) for v in x]

    elif isinstance(x, tuple):
        Warning('recursive_op1 on a tuple has never been tested')
        return tuple([recursive_op1(v, op, **kwargs) for v in x])

    else:
        return op(x, **kwargs)
