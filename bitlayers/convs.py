from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import nn
import torch

from bitlayers.helpers import make_divisible
from bitlayers.pool import PoolModels

from .norms import NormModels
from .drop import DropPath
from .acts import ActModels
from .bit import Bit
from .padding import PadSame

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

class Conv2dModels:
    """Lightweight Pydantic wrappers for key convolutional primitives."""

    class BasicModel(BaseModel):
        bit: bool = True
        scale_op: str = "median"

        def build(self):
            mod = Conv2dModules
            return mod.__dict__[f'{self.__class__.__name__}'](self)
        
    class Conv2d(BasicModel):
        in_channels: int
        out_channels: int = -1 # just a place holder not valid
        kernel_size: IntOrPair = 3
        stride: IntOrPair = 1
        padding: PadArg = 0
        dilation: IntOrPair = 1
        groups: int = 1
        bias: bool = True
        padding_mode: str = 'zeros'
        
        bit: bool = True
        scale_op: str = "median"

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

    class Conv2dSame(Conv2d):
        @classmethod
        def shortcut(cls, cmd = "cons_i3_o32_k3_s1_p1_d1_bs_bit"):
            return super().shortcut(cmd,prefix="cons")

    class Conv2dBn(Conv2d):
        bn: NormModels.type

    class Conv2dAct(Conv2d):
        act: ActModels.type

    class Conv2dBnAct(Conv2d):
        bn: NormModels.type
        act: ActModels.type

    class SqueezeExcite(BasicModel):
        in_channels: int

        conv_reduce_layer: Union['Conv2dModels.Conv2dAct'] = Field(
                default_factory=lambda:Conv2dModels.Conv2dAct(
                                            in_channels=-1, act=ActModels.ReLU(),))
        
        conv_expand_layer: Union['Conv2dModels.Conv2dAct'] = Field(
                default_factory=lambda:Conv2dModels.Conv2dAct(
                                            in_channels=-1, act=ActModels.Sigmoid(),))
        rd_ratio: float = 0.25
        rd_channels: Optional[int] = None
        
        @model_validator(mode='after')
        def valid_model(self):
            if self.rd_channels is None:
                self.rd_channels = round(self.in_channels * self.rd_ratio)
            
            self.conv_reduce_layer.in_channels  = self.in_channels
            self.conv_reduce_layer.out_channels = self.rd_channels

            self.conv_expand_layer.in_channels  = self.rd_channels
            self.conv_expand_layer.out_channels = self.in_channels
            return self
        
    class DepthwiseSeparableConv(Conv2d):
        in_channels: int
        out_channels: int

        kernel_size: Optional[IntOrPair] = None
        dw_kernel_size: int = 3
        pw_kernel_size: int = 1

        group_size: int = 1
        padding: PadArg = 'same'

        stride: IntOrPair = 1
        noskip: bool = False

        drop_path_rate: float = 0.0

        conv_s2d_layer: Optional[Union['Conv2dModels.Conv2dBnAct']] = Field(default=None)

        conv_dw_layer: Union['Conv2dModels.Conv2dBnAct'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dBnAct(
                in_channels=-1,
                bn=NormModels.BatchNorm2d(),
                act=ActModels.ReLU(),
            )
        )

        se_layer: Optional['Conv2dModels.SqueezeExcite'] = None
        aa_layer: Optional[PoolModels.type] = None

        conv_pw_layer: Union['Conv2dModels.Conv2dBnAct'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dBnAct(
                in_channels=-1,
                bn=NormModels.BatchNorm2d(),
                act=ActModels.ReLU(),
            )
        )

        @staticmethod        
        def _convert_padding(p):
            """Map timm-style pad_type to something Conv2d understands."""
            if p in ('same', 'SAME'):
                return 'same'
            if p in ('valid', 'VALID'):
                return 'valid'
            if p in ('', None):
                return 0
            return p  # assume int / tuple / already valid

        @model_validator(mode='after')
        def valid_model(self):
            self.kernel_size = None
            self.padding = self.__class__._convert_padding(self.padding)

            in_chs_local = self.in_channels
            dw_kernel_local = self.dw_kernel_size
            pad_type_local = self.padding


            if self.conv_s2d_layer is not None:
                sd_chs = int(in_chs_local * 4)
                self.conv_s2d_layer.in_channels = in_chs_local
                self.conv_s2d_layer.out_channels = sd_chs
                self.conv_s2d_layer.kernel_size = 2
                self.conv_s2d_layer.stride = 2
                self.conv_s2d_layer.padding = 'same'
                self.conv_s2d_layer.bias = False

                dw_kernel_local = (dw_kernel_local + 1) // 2
                dw_pad_type = 'same' if dw_kernel_local == 2 else pad_type_local
                in_chs_local = sd_chs
                use_aa = False  # we already downsampled
                self.aa_layer = None
            else:
                dw_pad_type = pad_type_local

            use_aa = (self.aa_layer is None)

            def num_groups(group_size: Optional[int], channels: int):
                if not group_size:  # 0 or None
                    return 1  # normal conv with 1 group
                else:
                    # NOTE group_size == 1 -> depthwise conv
                    assert channels % group_size == 0
                    return channels // group_size
                
            mid_chs = in_chs_local
            groups = num_groups(self.group_size, in_chs_local)

            self.conv_dw_layer.in_channels = mid_chs
            self.conv_dw_layer.out_channels = mid_chs
            self.conv_dw_layer.kernel_size = dw_kernel_local
            self.conv_dw_layer.stride = 1 if use_aa else self.stride
            self.conv_dw_layer.padding = dw_pad_type
            self.conv_dw_layer.groups = groups
            self.conv_dw_layer.bias = False

            # Pointwise conv
            self.conv_pw_layer.in_channels = mid_chs
            self.conv_pw_layer.out_channels = self.out_channels
            self.conv_pw_layer.kernel_size = self.pw_kernel_size
            self.conv_pw_layer.stride = 1
            self.conv_pw_layer.padding = self.padding
            self.conv_pw_layer.bias = False

            return self
        
    class InvertedResidual(DepthwiseSeparableConv):
        """InvertedResidual (MBConv) block using string layer names."""
        exp_ratio: float = 1.0
        exp_kernel_size: int = 1

        conv_pwl_layer: Union['Conv2dModels.Conv2dBnAct'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dBnAct(
                in_channels=-1,
                bn=NormModels.BatchNorm2d(),
                act=ActModels.Identity(),
            )
        )

        @model_validator(mode='after')
        def valid_model(self):
            self.kernel_size = None
            self.padding = self.__class__._convert_padding(self.padding)

            in_chs_local = self.in_channels
            dw_kernel_local = self.dw_kernel_size
            pad_type_local = self.padding

            if self.conv_s2d_layer is not None:
                sd_chs = int(in_chs_local * 4)
                self.conv_s2d_layer.in_channels = in_chs_local
                self.conv_s2d_layer.out_channels = sd_chs
                self.conv_s2d_layer.kernel_size = 2
                self.conv_s2d_layer.stride = 2
                self.conv_s2d_layer.padding = 'same'
                self.conv_s2d_layer.bias = False

                dw_kernel_local = (dw_kernel_local + 1) // 2
                dw_pad_type = 'same' if dw_kernel_local == 2 else pad_type_local
                in_chs_local = sd_chs
                use_aa = False  # we already downsampled
                self.aa_layer = None
            else:
                dw_pad_type = pad_type_local

            use_aa = (self.aa_layer is None)

            def num_groups(group_size: Optional[int], channels: int):
                if not group_size:  # 0 or None
                    return 1  # normal conv with 1 group
                else:
                    # NOTE group_size == 1 -> depthwise conv
                    assert channels % group_size == 0
                    return channels // group_size
                
            mid_chs = make_divisible(in_chs_local * self.exp_ratio)
            groups = num_groups(self.group_size, in_chs_local)

            # Point-wise expansion
            self.conv_pw_layer.in_channels = in_chs_local
            self.conv_pw_layer.out_channels = mid_chs
            self.conv_pw_layer.kernel_size = self.exp_kernel_size
            self.conv_pw_layer.padding = self.padding

            # Depth-wise convolution
            self.conv_dw_layer.in_channels = mid_chs
            self.conv_dw_layer.out_channels = mid_chs
            self.conv_dw_layer.kernel_size = dw_kernel_local
            self.conv_dw_layer.stride = 1 if use_aa else self.stride
            self.conv_dw_layer.padding = dw_pad_type
            self.conv_dw_layer.groups = groups            
            
            # Point-wise linear projection
            self.conv_pwl_layer.in_channels = mid_chs
            self.conv_pwl_layer.out_channels = self.out_channels
            self.conv_pwl_layer.kernel_size = self.pw_kernel_size
            self.conv_pwl_layer.padding = self.padding
            self.conv_pwl_layer.act = ActModels.Identity()
            return self
        
class Conv2dModules:
    class Module(nn.Module):
        def __init__(self,para:BaseModel,para_cls):
            if type(para) is dict: para = para_cls(**para)
            self.para = json.loads(para.model_dump_json())
            super().__init__()
        
        @torch.no_grad()
        def to_ternary(self,mods=[]):
            for m in mods:
                if self.__dict__[m] and hasattr(self.__dict__[m],'to_ternary'):
                    setattr(self,m,self.__dict__[m].to_ternary())
            return self

    class Conv2d(Module):
        def __init__(self,para:Conv2dModels.Conv2d,para_cls=Conv2dModels.Conv2d):
            super().__init__(para,para_cls)
            self.para:Conv2dModels.Conv2d = self.para

            if para.bit:
                self.conv = Bit.Conv2d(**para.model_dump())
            else:                
                self.conv = nn.Conv2d(**para.model_dump(exclude=['scale_op']))

        def forward(self,x):
            return self.conv(x)

        @torch.no_grad()
        def to_ternary(self):
            if hasattr(self.conv,'to_ternary'):return self.conv.to_ternary()
            print('to_ternary is no support!')

    class Conv2dSame(Conv2d):
        """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
        """
        def __init__(self,para:Conv2dModels.Conv2dSame,para_cls=Conv2dModels.Conv2dSame):
            super().__init__(para,para_cls)
            w,s,d = self.conv.weight.shape[-2:], self.conv.stride, self.conv.dilation
            self.pad = PadSame(w,s,d)
            self.para:Conv2dModels.Conv2dSame = self.para

        def forward(self, x):
            return super().forward(self.pad(x))
        
        @torch.no_grad()
        def to_ternary(self):
            return nn.Sequential(self.pad,super().to_ternary())
    class Conv2dBn(Conv2d):
        def __init__(self,para:Conv2dModels.Conv2dBn,para_cls=Conv2dModels.Conv2dBn):
            super().__init__(para,para_cls)
            self.bn = para.bn.build()           

        def forward(self, x):
            return self.bn(super().forward(x))
        
        @torch.no_grad()
        def to_ternary(self):
            return nn.Sequential(super().to_ternary(),self.bn)
        
    class Conv2dAct(Conv2d):
        def __init__(self,para:Conv2dModels.Conv2dAct,para_cls=Conv2dModels.Conv2dAct):
            super().__init__(para,para_cls)
            self.act = para.act.build()

        def forward(self, x):
            return self.act(super().forward(x))
        
        @torch.no_grad()
        def to_ternary(self):
            return nn.Sequential(super().to_ternary(),self.act)
        
    class Conv2dBnAct(Conv2d):
        def __init__(self,para:Conv2dModels.Conv2dBnAct,para_cls=Conv2dModels.Conv2dBnAct):
            super().__init__(para,para_cls)
            self.bn = para.bn.build()
            self.act = para.act.build()

        def forward(self, x):
            return self.act(self.bn(super().forward(x)))
        
        @torch.no_grad()
        def to_ternary(self):
            return nn.Sequential(super().to_ternary(),self.bn,self.act)
        
    class SqueezeExcite(Module):

        def __init__(self,para,para_cls=Conv2dModels.SqueezeExcite):
            super().__init__(para,para_cls)
            self.para:Conv2dModels.SqueezeExcite=self.para
            self.conv_reduce:Conv2dModules.Conv2dAct = self.para.conv_reduce_layer.build()
            self.conv_expand:Conv2dModules.Conv2dAct = self.para.conv_expand_layer.build()
            
        def forward(self, x:torch.Tensor):
            x_se = x.mean((2, 3), keepdim=True)
            x_se = self.conv_reduce(x_se)
            x_se = self.conv_expand(x_se)
            return x * x_se
        
        @torch.no_grad()
        def to_ternary(self,mods=['conv_reduce','conv_reduce']):
            self.drop_path = nn.Identity()
            return super().to_ternary(mods)

    class DepthwiseSeparableConv(Module):
        """Depthwise-separable block with Pydantic config and string layer names."""

        def __init__(self,para,para_cls=Conv2dModels.DepthwiseSeparableConv):
            super().__init__(para,para_cls)
            self.para:Conv2dModels.DepthwiseSeparableConv=self.para
            self.conv_s2d = self.para.conv_s2d_layer.build() if self.para.conv_s2d_layer else nn.Identity()            
            self.conv_dw = self.para.conv_dw_layer.build()
            self.se = self.para.se_layer.build() if self.para.se_layer else nn.Identity()
            self.aa = self.para.aa_layer.build() if self.para.aa_layer else nn.Identity()
            self.conv_pw = self.para.conv_pw_layer.build()
            # ---- DropPath / stochastic depth ----
            self.drop_path = DropPath(self.para.drop_path_rate) if self.para.drop_path_rate > 0 else nn.Identity()

        def feature_info(self, location):
            if location == 'expansion':  # after SE, before PW
                return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.para.in_channels)
            else:  # 'bottleneck'
                return dict(module='', num_chs=self.para.out_channels)

        def forward(self, x):
            shortcut = x
            if self.para.conv_s2d_layer is not None:
                x = self.conv_s2d(x) # with bn and act

            x = self.conv_dw(x) # with bn and act

            if self.para.aa_layer is not None:
                x = self.aa(x)
                
            if self.para.se_layer is not None:
                x = self.se(x)
                
            x = self.conv_pw(x) # with bn and act            
            if (self.para.in_channels == self.para.out_channels and self.para.stride == 1) and (not self.para.noskip):                
                x = self.drop_path(x) + shortcut
            return x
        
        @torch.no_grad()
        def to_ternary(self,mods=['conv_s2d','conv_dw','se','aa','conv_pw']):
            self.drop_path = nn.Identity()
            return super().to_ternary(mods)

    class InvertedResidual(DepthwiseSeparableConv):
        def __init__(self, para, para_cls=Conv2dModels.InvertedResidual):
            super().__init__(para, para_cls)
            self.para:Conv2dModels.InvertedResidual=self.para
            self.conv_pwl = self.para.conv_pwl_layer.build()
        
        def forward(self, x):
            shortcut = x
            if self.para.conv_s2d_layer is not None:
                x = self.conv_s2d(x) # with bn and act

            x = self.conv_dw(x) # with bn and act

            if self.para.aa_layer is not None:
                x = self.aa(x)
                
            if self.para.se_layer is not None:
                x = self.se(x)
                
            x = self.conv_pw(x) # with bn and act            
            if (self.para.in_channels == self.para.out_channels and self.para.stride == 1) and (not self.para.noskip):                
                x = self.drop_path(x) + shortcut
            return x
        
        def forward(self, x):
            shortcut = x
            if self.conv_s2d is not None:
                x = self.conv_s2d(x)

            x = self.conv_pw(x)
            x = self.conv_dw(x)

            if self.para.aa_layer is not None:
                x = self.aa(x)
            if self.para.se_layer is not None:
                x = self.se(x)

            x = self.conv_pwl(x)

            if (self.para.in_channels == self.para.out_channels and self.para.stride == 1) and (not self.para.noskip):                
                x = self.drop_path(x) + shortcut
            return x
        
        @torch.no_grad()
        def to_ternary(self,mods=['conv_s2d','conv_dw','se','aa','conv_pw']):
            return super().to_ternary(mods)



























#...