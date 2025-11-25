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

    class Conv2dSame(Conv2d):
        @classmethod
        def shortcut(cls, cmd = "cons_i3_o32_k3_s1_p1_d1_bs_bit"):
            return super().shortcut(cmd,prefix="cons")

    class SqueezeExcite(BasicModel):
        in_chs: int
        rd_ratio: float = 0.25
        rd_channels: Optional[int] = None
        act_layer: str = 'ReLU'
        gate_layer: str = 'Sigmoid'
        force_act_layer: Optional[str] = None
        # rd_round_fn: Optional[Callable] = None
        
        @field_validator("rd_channels", always=True)
        def compute_rd_channels(cls, v, values):
            # If rd_channels is explicitly given, keep it
            if v is not None: return v
            in_chs = values.get("in_chs")
            rd_ratio = values.get("rd_ratio", 0.25)
            rd_round_fn = values.get("rd_round_fn") or round
            if in_chs is None:
                raise ValueError("`in_chs` must be set to compute `rd_channels`.")
            return rd_round_fn(in_chs * rd_ratio)

    class DepthwiseSeparableConv(BaseModel):
        in_chs: int
        out_chs: int
        dw_kernel_size: int = 3
        stride: int = 1
        dilation: int = 1
        group_size: int = 1
        pad_type: str = ''
        noskip: bool = False
        pw_kernel_size: int = 1
        pw_act: bool = False
        s2d: int = 0

        act_layer: str = 'Relu'
        norm_layer: str = 'Batchnorm2d'
        aa_layer: Optional[str] = None
        se_layer: Optional[str] = None

        drop_path_rate: float = 0.0

    # class MixedConv2d(Conv2d):
    #     padding: PadArg = ''
    #     depthwise: bool = False
    #     padding_mode: None = None

    #     def build(self) -> nn.Module:
    #         return _MixedConv2d(**self.model_dump())

    #     @classmethod
    #     def specifics(cls):return {**super().specifics(),'depthwise': 'dw'}
        
    #     @classmethod
    #     def shortcut(cls, cmd = "mcon_i3_o32_k3_s1_p1_d1_bs_bit_dw"):
    #         return super().shortcut(cmd=cmd,prefix="mcon",specifics=cls.specifics(),
    #                                 json_schema=cls.model_json_schema()['properties'])
        
    # class SeparableConv2d(Conv2d):
    #     channel_multiplier: float = 1.0
    #     pw_kernel_size: int = 1

    #     def build(self) -> nn.Module:
    #         return _SeparableConv2d(**self.model_dump())
        
    #     @classmethod
    #     def specifics(cls):return {**super().specifics(),'pw_kernel_size': 'pk'}
        
    #     @classmethod
    #     def shortcut(cls, cmd = "scon_i3_o32_k3_s1_p1_d1_bs_bit_dw"):
    #         return super().shortcut(cmd=cmd,prefix="scon",specifics=cls.specifics(),
    #                                 json_schema=cls.model_json_schema()['properties'])

    # class SplitBatchNorm2d(Conv2d):
    #     # This one doesn’t actually use the Conv2d fields, but if you rely on it
    #     # being a Conv2d subclass, keep it that way and just define BN-specific ones.
    #     num_features: int
    #     eps: float = 1e-5
    #     momentum: float = 0.1
    #     affine: bool = True
    #     track_running_stats: bool = True
    #     num_splits: int = 2

    #     def build(self) -> nn.Module:
    #         return _SplitBatchNorm2d(**self.model_dump())
        
    #     def model_post_init(self, context):
    #         raise NotImplementedError()

    # class _BaseStdConv2d(Conv2d):
    #     """Shared config for StdConv2d variants."""
    #     in_channel: int                  # as in your original code
    #     out_channels: int
    #     kernel_size: IntOrPair
    #     stride: int = 1
    #     padding: Optional[PadArg] = None
    #     dilation: int = 1
    #     groups: int = 1
    #     bias: bool = False
    #     eps: float = 1e-6

    # class StdConv2d(_BaseStdConv2d):
    #     def model_post_init(self, context):
    #         raise NotImplementedError()

    # class StdConv2dSame(_BaseStdConv2d):
    #     # Only difference vs StdConv2d is padding default.
    #     padding: PadArg = 'SAME'

    #     def model_post_init(self, context):
    #         raise NotImplementedError()
        
    # class _BaseScaledStdConv2d(Conv2d):
    #     """Shared config for ScaledStdConv2d variants."""
    #     in_channels: int
    #     out_channels: int
    #     kernel_size: IntOrPair
    #     stride: int = 1
    #     padding: Optional[PadArg] = None
    #     dilation: int = 1
    #     groups: int = 1
    #     bias: bool = True
    #     gamma: float = 1.0
    #     eps: float = 1e-6
    #     gain_init: float = 1.0

    #     def model_post_init(self, context):
    #         raise NotImplementedError()
        
    # class ScaledStdConv2d(_BaseScaledStdConv2d):
    #     def model_post_init(self, context):
    #         raise NotImplementedError()
        
    # class ScaledStdConv2dSame(_BaseScaledStdConv2d):
    #     # Only difference vs ScaledStdConv2d is padding default.
    #     padding: PadArg = 'SAME'
    #     def model_post_init(self, context):
    #         raise NotImplementedError()
    

class Conv2dModules:
    class Module(nn.Module):
        def __init__(self,para:BaseModel,para_cls):
            if type(para) is dict: para = para_cls(**para)
            self.para = json.loads(para.model_dump_json())
            super().__init__()

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
            return self.conv.to_ternary()

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
            return nn.Sequential(
                    self.pad,
                    super().to_ternary())

    class SqueezeExcite(nn.Module):

        def __init__(self,para,para_cls=Conv2dModels.SqueezeExcite):
            super().__init__(para,para_cls)
            self.para:Conv2dModels.SqueezeExcite=self.para

            act_layer = self.para.force_act_layer or self.para.act_layer
            self.conv_reduce = Conv2dModels.Conv2d(in_channels=self.para.in_chs,
                                                   out_channels=self.para.rd_channels,
                                                   kernel_size=1,
                                                   bias=True).build()        
            self.act1 = ActModels.__dict__[act_layer](inplace=True)
            
            self.conv_expand = Conv2dModels.Conv2d(in_channels=self.para.rd_channels,
                                        out_channels=self.para.in_chs,
                                        kernel_size=1,
                                        bias=True).build()        
            self.gate = ActModels.__dict__[self.para.gate_layer]()

        def forward(self, x:torch.Tensor):
            x_se = x.mean((2, 3), keepdim=True)
            x_se = self.conv_reduce(x_se)
            x_se = self.act1(x_se)
            x_se = self.conv_expand(x_se)
            return x * self.gate(x_se)


        @torch.no_grad()
        def to_ternary(self):
            if hasattr(self.conv_reduce,'to_ternary') and hasattr(self.conv_expand,'to_ternary'):
                self.conv_reduce = self.conv_reduce.to_ternary()
                self.conv_expand = self.conv_expand.to_ternary()
                return self
            print('to_ternary is no support!')

    class DepthwiseSeparableConv(nn.Module):
        """Depthwise-separable block with Pydantic config and string layer names."""

        def __init__(self,para,para_cls=Conv2dModels.DepthwiseSeparableConv):
            super().__init__(para,para_cls)
            self.para:Conv2dModels.DepthwiseSeparableConv=self.para
            
            def _convert_padding(p):
                """Map timm-style pad_type to something Conv2d understands."""
                if p in ('same', 'SAME'):
                    return 'same'
                if p in ('valid', 'VALID'):
                    return 'valid'
                if p in ('', None):
                    return 0
                return p  # assume int / tuple / already valid

            para = self.para

            # your existing helper should know how to handle string layer names
            norm_act_layer = get_norm_act_layer(para.norm_layer, para.act_layer)

            self.has_skip = (para.stride == 1 and para.in_chs == para.out_chs) and not para.noskip
            self.has_pw_act = para.pw_act
            use_aa = para.aa_layer is not None and para.stride > 1  # AA only when downsampling

            # ---- optional space-to-depth pre-conv ----
            in_chs_local = para.in_chs
            dw_kernel_local = para.dw_kernel_size
            pad_type_local = para.pad_type

            if para.s2d == 1:
                sd_chs = int(in_chs_local * 4)
                self.conv_s2d = nn.Conv2d(
                    in_chs_local,
                    sd_chs,
                    kernel_size=2,
                    stride=2,
                    padding=_convert_padding('same'),
                    bias=False,
                )
                self.bn_s2d = norm_act_layer(sd_chs, sd_chs)
                dw_kernel_local = (dw_kernel_local + 1) // 2
                dw_pad_type = 'same' if dw_kernel_local == 2 else pad_type_local
                in_chs_local = sd_chs
                use_aa = False  # we already downsampled
            else:
                self.conv_s2d = None
                self.bn_s2d = None
                dw_pad_type = pad_type_local

            groups = num_groups(para.group_size, in_chs_local)

            # ---- depthwise conv ----
            self.conv_dw = nn.Conv2d(
                in_chs_local,
                in_chs_local,
                kernel_size=dw_kernel_local,
                stride=1 if use_aa else para.stride,
                dilation=para.dilation,
                padding=_convert_padding(dw_pad_type),
                groups=groups,
                bias=False,
            )
            self.bn1 = norm_act_layer(in_chs_local, inplace=True)

            # ---- anti-aliasing (delegated to your create_aa helper) ----
            self.aa = create_aa(para.aa_layer, channels=para.out_chs, stride=para.stride, enable=use_aa)

            # ---- squeeze-and-excitation layer, SE (you can plug in your own string->module factory here) ----
            if para.se_layer is not None:
                raise NotImplementedError(
                    "String-based se_layer resolution is not implemented here. "
                    "Resolve cfg.se_layer (str) to a module in this spot."
                )
            else:
                self.se = nn.Identity()

            # ---- pointwise conv ----
            self.conv_pw = nn.Conv2d(
                in_chs_local,
                para.out_chs,
                kernel_size=para.pw_kernel_size,
                stride=1,
                padding=_convert_padding(para.pad_type),
                bias=False,
            )
            self.bn2 = norm_act_layer(para.out_chs, inplace=True, apply_act=self.has_pw_act)

            # ---- DropPath / stochastic depth ----
            self.drop_path = DropPath(para.drop_path_rate) if para.drop_path_rate > 0 else nn.Identity()

        def feature_info(self, location):
            if location == 'expansion':  # after SE, before PW
                return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
            else:  # 'bottleneck'
                return dict(module='', num_chs=self.conv_pw.out_channels)

        def forward(self, x):
            shortcut = x
            if self.conv_s2d is not None:
                x = self.conv_s2d(x)
                x = self.bn_s2d(x)
            x = self.conv_dw(x)
            x = self.bn1(x)
            x = self.aa(x)
            x = self.se(x)
            x = self.conv_pw(x)
            x = self.bn2(x)
            if self.has_skip:
                x = self.drop_path(x) + shortcut
            return x
        

























#...