""" Layer/Module Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
from itertools import repeat
import collections.abc
from typing import Any


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def convert_padding(p):
    """Map timm-style pad_type to something Conv2d understands."""
    if p in ('same', 'SAME'):
        return 'same'
    if p in ('valid', 'VALID'):
        return 'valid'
    if p in ('', None):
        return 0
    return p  # assume int / tuple / already valid

def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def extend_tuple(x, n):
    # pads a tuple to specified n by padding with last value
    if not isinstance(x, (tuple, list)):
        x = (x,)
    else:
        x = tuple(x)
    pad_n = n - len(x)
    if pad_n <= 0:
        return x[:n]
    return x + (x[-1],) * pad_n

def Cls_parse(v: Any, cls_dict: dict[str, type]) -> Any:
    if isinstance(v, tuple(cls_dict.values())):
        return v
    if not isinstance(v, dict):
        raise TypeError(f"expected a module instance or serialized module dict. Got {v}")
    raw_uuid = v.get("uuid")
    if not isinstance(raw_uuid, str) or ":" not in raw_uuid:
        raise ValueError("serialized module must include uuid like 'ClassName:...'")
    kind = raw_uuid.split(":", 1)[0]
    module_cls = cls_dict.get(kind)
    if module_cls is None:
        raise ValueError(f"unknown module type: {kind}")
    return module_cls.model_validate(v)
