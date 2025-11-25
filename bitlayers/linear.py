from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, field_validator
from torch import nn
import torch

from .acts import ActModels
from .bit import Bit
from .padding import PadSame
from timmlayers.mixed_conv2d import MixedConv2d as _MixedConv2d
from timmlayers.separable_conv import SeparableConv2d as _SeparableConv2d
from timmlayers.split_batchnorm import SplitBatchNorm2d as _SplitBatchNorm2d
from timmlayers.std_conv import (
    ScaledStdConv2d as _ScaledStdConv2d,
    ScaledStdConv2dSame as _ScaledStdConv2dSame,
    StdConv2d as _StdConv2d,
    StdConv2dSame as _StdConv2dSame,
)

IntOrPair = Union[int, Tuple[int, int]]
KernelSizeArg = Union[int, Tuple[int, int], Sequence[int]]
PadArg = Union[str, int, Tuple[int, int]]

def shortcut_prefix_fields(json_schema:Dict,specifics:Dict[str,str]={},le=1,alow=['integer','boolean']):
    ts = json_schema.items()#cls.model_json_schema()['properties'].items()
    ts = [([v.get('type')] or [i['type'] for i in v['anyOf']]) for k,v in ts]
    ts = [set(i).intersection(set(alow)) for i in ts]

    ms = [i for i,j in zip(json_schema.keys(),ts) if len(i)>0]
    ks = {i:i[:le] for i in ms if i not in specifics}
    ks.update(specifics)
    ks = {v:k for k,v in ks.items() if v is not None}
    if len(set(ks.keys()))!=len(set(ks.values())):
        raise ValueError(f'short keys is duplicat, {list(ks.keys())} <=> {list(ks.values())}')
    return ks

def parse_shortcut_kwargs(cmd: str, prefix: str, json_schema:Dict,
                            specifics:Dict[str,str], le=1, alow=['integer','boolean']):
    if not cmd.startswith(prefix): return None
    cmds = cmd.split('_')[1:]
    sf = shortcut_prefix_fields(json_schema,specifics,le,alow)
    ks = list(sf.keys())
    ks = sorted(ks, key=len, reverse=True)
    args_tmp = {}
    for k in ks:
        for i,c in enumerate(cmds):
            if c.startswith(k):
                args_tmp[sf[k]] = c[len(k):]
                cmds.pop(i)
                break
            
    args = {}
    for k,v in args_tmp.items():
        try:
            v = int(v)
        except: 
            v = True
        args[k] = v
    return args

class LinearModels:

    class BasicModel(BaseModel):
        bit: bool = True
        scale_op: str = "median"

        def build(self):
            mod = LinearModules
            return mod.__dict__[f'{self.__class__.__name__}'](self)
        
    class Linear(BasicModel):
        in_channels: int
        out_channels: int
        kernel_size: IntOrPair = 3
        stride: IntOrPair = 1
        padding: PadArg = 0
        dilation: IntOrPair = 1
        groups: int = 1
        bias: bool = True
        padding_mode: str = 'zeros'
        
        @classmethod
        def specifics(cls):
            return {
                'bias': 'bs',
                'bit': 'bit',
                'padding_mode': None,
                'scale_op': None,
            }

        @classmethod
        def shortcut(cls,cmd: str = "conv_i3_o32_k3_s1_p1_d1_bs_bit",
                     prefix: str = None, specifics:Dict[str,str]=None,
                     json_schema: Dict=None,
                     le=1, alow=['integer','boolean']):
            if prefix is None: prefix = "conv"
            if specifics is None: specifics = cls.specifics()
            if json_schema is None: json_schema = cls.model_json_schema()['properties']
            kwargs = parse_shortcut_kwargs(
                            cmd=cmd, prefix=prefix, specifics=specifics,le=le, alow=alow,
                            json_schema=json_schema)
            if not kwargs:return None
            return cls(**kwargs)

class LinearModules:
    class Linear(nn.Module):
        def __init__(self,para:LinearModels.Linear,para_cls=LinearModels.Linear):
            if type(para) is dict: para = para_cls(**para)
            self.para = json.loads(para.model_dump_json())

            super().__init__()
            if para.bit:
                self.conv = Bit.Conv2d(**para.model_dump())
            else:                
                self.conv = nn.Conv2d(**para.model_dump(exclude=['scale_op']))

            self.para:LinearModels.Linear = self.para

        def forward(self,x):
            return self.conv(x)

        @torch.no_grad()
        def to_ternary(self):
            if hasattr(self.conv,'to_ternary'):return self.conv.to_ternary()
            print('to_ternary is no support!')
            return self.conv.to_ternary()
