from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union, Unpack

from pydantic import BaseModel, Field, model_validator
from torch import nn
import torch
import torch.nn.functional as F


from .helpers import make_divisible, to_2tuple
from .pool import PoolModels
from .norms import NormModels
from .drop import DropPath
from .acts import ActModels
from .bit import Bit
from .base import CommonModel, CommonModule
from .linear import LinearModels, LinearModules

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
    class BasicModel(CommonModel):
        def build(self): return self._build(self,Conv2dModules)

        
    class Conv2d(BasicModel):
        in_channels: int
        out_channels: int = -1 # just a place holder not valid
        kernel_size: IntOrPair = 3
        stride: IntOrPair = 1
        padding: PadArg = 'same'
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

    class Conv2dDepthwise(Conv2d):
        group_size: int = Field(default=1, gt=0, exclude=True)
        # If group_size = 1:
        # You get true depthwise convolution (each channel independent).

        # If group_size = in_channels:
        # You effectively get a regular convolution (no grouping).

        # If 1 < group_size < in_channels:
        # You get something in between: grouped depthwise-ish conv, where each group has group_size channels.

        @model_validator(mode='after')
        def valid_model(self):
            if self.in_channels % self.group_size != 0:
                raise ValueError("in_channels must be a multiple of group_size.")
            
            self.groups = self.in_channels // self.group_size

            if self.out_channels % self.groups != 0:
                raise ValueError("out_channels must be a multiple of groups.")
            return self

    class Conv2dPointwise(Conv2d):
        @model_validator(mode='after')
        def valid_model(self):
            self.kernel_size = 1
            self.padding = 0
            self.groups = 1
            return self

    class Conv2dNorm(Conv2d):
        norm: NormModels.type

    class Conv2dAct(Conv2d):
        act: ActModels.type

    class Conv2dNormAct(Conv2d):
        norm: NormModels.type
        act: ActModels.type

    class Conv2dDepthwiseNorm(Conv2dDepthwise,Conv2dNorm):pass
    class Conv2dDepthwiseAct(Conv2dDepthwise,Conv2dAct):pass
    class Conv2dDepthwiseNormAct(Conv2dDepthwise,Conv2dNormAct):pass

    class Conv2dPointwiseNorm(Conv2dPointwise,Conv2dNorm):pass
    class Conv2dPointwiseAct(Conv2dPointwise,Conv2dAct):pass
    class Conv2dPointwiseNormAct(Conv2dPointwise,Conv2dNormAct):pass


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
            
            self.conv_reduce_layer.update(
                    in_channels  = self.in_channels,
                    out_channels = self.rd_channels
            )

            self.conv_expand_layer.update(
                    in_channels  = self.rd_channels,
                    out_channels = self.in_channels
            )
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
        bias: bool = False

        noskip: bool = False
        drop_path_rate: float = 0.0

        conv_s2d_layer: Optional[Union['Conv2dModels.Conv2dNormAct']] = Field(default=None)

        conv_dw_layer: Union['Conv2dModels.Conv2dDepthwiseNormAct'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dDepthwiseNormAct(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
                act=ActModels.ReLU(),
            )
        )

        se_layer: Optional['Conv2dModels.SqueezeExcite'] = None
        aa_layer: Optional[PoolModels.type] = None

        conv_pw_layer: Union['Conv2dModels.Conv2dPointwiseNormAct'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dPointwiseNormAct(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
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
            dw_kernel_local = self.dw_kernel_size
            dw_pad_type = self.padding

            if self.conv_s2d_layer is not None:
                self.conv_s2d_layer.in_channels = self.in_channels
                self.conv_s2d_layer.out_channels = int(self.in_channels * 4)
                self.conv_s2d_layer.kernel_size = 2
                self.conv_s2d_layer.stride = 2
                self.conv_s2d_layer.padding = 'same'
                self.conv_s2d_layer.bias = self.bias
                if dw_kernel_local in (3,4):
                    dw_pad_type = 'same'
                # we already downsampled
                self.aa_layer = None

            use_aa = (self.aa_layer is None)

            in_channels = self.conv_s2d_layer.in_channels if self.conv_s2d_layer else self.in_channels

            # Depthwise conv
            self.conv_dw_layer.in_channels = in_channels
            self.conv_dw_layer.out_channels = in_channels
            self.conv_dw_layer.kernel_size = dw_kernel_local
            self.conv_dw_layer.stride = 1 if use_aa else self.stride
            self.conv_dw_layer.padding = 1 if dw_kernel_local==3 else dw_pad_type
            self.conv_dw_layer.group_size = self.group_size
            self.conv_dw_layer.bias = self.bias

            # Pointwise conv
            self.conv_pw_layer.in_channels = in_channels
            self.conv_pw_layer.out_channels = self.out_channels
            self.conv_pw_layer.kernel_size = self.pw_kernel_size # always 1
            self.conv_pw_layer.stride = 1
            self.conv_pw_layer.padding = self.padding
            self.conv_pw_layer.groups = 1
            self.conv_pw_layer.bias = self.bias
            return self
        
    class InvertedResidual(DepthwiseSeparableConv):
        """InvertedResidual (MBConv) block using string layer names."""
        exp_ratio: float = 1.0
        exp_kernel_size: int = 1

        conv_pw_exp_layer: Union['Conv2dModels.Conv2dPointwiseNormAct'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dPointwiseNormAct(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
                act=ActModels.ReLU(),
            )
        )

        conv_pw_layer: Union['Conv2dModels.Conv2dPointwiseNorm'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dPointwiseNorm(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
            )
        )

        @model_validator(mode='after')
        def valid_model(self):
            super().valid_model()

            in_channels = self.conv_s2d_layer.in_channels if self.conv_s2d_layer else self.in_channels
            mid_chs = make_divisible(in_channels * self.exp_ratio)

            # Point-wise expansion
            self.conv_pw_exp_layer.in_channels = in_channels
            self.conv_pw_exp_layer.out_channels = mid_chs
            self.conv_pw_exp_layer.kernel_size = self.exp_kernel_size
            self.conv_pw_exp_layer.stride = 1 if self.aa_layer else self.stride
            self.conv_pw_exp_layer.padding = self.padding
            self.conv_pw_exp_layer.dilation = self.dilation
            self.conv_pw_exp_layer.bias = self.bias

            # Depth-wise convolution
            self.conv_dw_layer.in_channels = mid_chs
            self.conv_dw_layer.out_channels = mid_chs
            
            # Point-wise linear projection
            self.conv_pw_layer.in_channels = mid_chs
            self.conv_pw_layer.padding = self.padding
            self.conv_pw_layer.bias = self.bias
            return self

    class CondConvResidual(InvertedResidual):
        num_experts:int
        routing_layer: LinearModels.Linear =  Field(
            default_factory=lambda: LinearModels.Linear(in_features=-1)
        )
        
        @model_validator(mode='after')
        def valid_model(self):
            self.routing_layer.in_features = self.in_channels
            self.routing_layer.out_features = self.num_experts
        
    class EdgeResidual(InvertedResidual):
        """Fused MBConv / EdgeResidual block configured via Pydantic model."""
        force_in_channels: int = 0
        bias: bool = False

        @staticmethod
        def _num_groups(group_size: int, channels: int) -> int:
            if not group_size:
                return 1
            if channels % group_size != 0:
                raise ValueError("channels must be divisible by group_size.")
            return channels // group_size
        
        @model_validator(mode='after')
        def valid_model(self):
            super().valid_model()

            mid_base = self.force_in_channels if self.force_in_channels > 0 else self.in_channels
            mid_chs = make_divisible(mid_base * self.exp_ratio)
            groups = self._num_groups(self.group_size, mid_chs)

            self.conv_pw_exp_layer.in_channels = self.in_channels
            self.conv_pw_exp_layer.out_channels = mid_chs
            self.conv_pw_exp_layer.groups = groups

            self.conv_pw_layer.in_channels = mid_chs
            self.conv_pw_layer.out_channels = self.out_channels
            return self
    
    class UniversalInvertedResidual(BasicModel):
        """Universal Inverted Bottleneck with configurable depthwise stages."""

        in_channels: int
        out_channels: int

        dw_kernel_size_start: int = 0
        dw_kernel_size_mid: int = 3
        dw_kernel_size_end: int = 0

        stride: int = 1
        dilation: int = 1
        group_size: int = 1
        padding: PadArg = 'same'
        bias: bool = False

        noskip: bool = False
        exp_ratio: float = 1.0

        aa_layer: Optional[PoolModels.type] = None
        se_layer: Optional['Conv2dModels.SqueezeExcite'] = None

        conv_dw_start_layer: Optional[Union['Conv2dModels.Conv2dDepthwiseNorm']] = Field(
            default_factory=lambda: Conv2dModels.Conv2dDepthwiseNorm(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
            )
        )
        conv_pw_exp_layer: Union['Conv2dModels.Conv2dPointwiseNormAct'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dPointwiseNormAct(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
                act=ActModels.ReLU(),
            )
        )
        conv_dw_mid_layer: Optional[Union['Conv2dModels.Conv2dDepthwiseNormAct']] = Field(
            default_factory=lambda: Conv2dModels.Conv2dDepthwiseNormAct(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
                act=ActModels.ReLU(),
            )
        )
        conv_pw_proj_layer: Union['Conv2dModels.Conv2dPointwiseNorm'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dPointwiseNorm(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
            )
        )
        
        conv_dw_end_layer: Optional[Union['Conv2dModels.Conv2dNorm']] = Field(
            default_factory=lambda: Conv2dModels.Conv2dDepthwiseNorm(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
            )
        )

        drop_path_rate: float = 0.0
        layer_scale_init_value: Optional[float] = 1e-5

        @model_validator(mode='after')
        def valid_model(self):
            if self.stride > 1 and not (
                self.dw_kernel_size_start or self.dw_kernel_size_mid or self.dw_kernel_size_end
            ):
                raise ValueError("UniversalInvertedResidual needs a depthwise kernel when stride > 1.")

            self.padding = Conv2dModels.DepthwiseSeparableConv._convert_padding(self.padding)

            def num_groups(group_size: Optional[int], channels: int) -> int:
                if not group_size:
                    return 1
                if channels % group_size != 0:
                    raise ValueError("channels must be divisible by group_size.")
                return channels // group_size

            mid_chs = make_divisible(self.in_channels * self.exp_ratio)

            if self.dw_kernel_size_start:
                dw_start_stride = self.stride if not self.dw_kernel_size_mid else 1
                use_aa_start = self.aa_layer is not None and dw_start_stride > 1
                self.conv_dw_start_layer.in_channels = self.in_channels
                self.conv_dw_start_layer.out_channels = self.in_channels
                self.conv_dw_start_layer.kernel_size = self.dw_kernel_size_start
                self.conv_dw_start_layer.stride = 1 if use_aa_start else dw_start_stride
                self.conv_dw_start_layer.padding = self.padding
                self.conv_dw_start_layer.group_size = self.group_size
                self.conv_dw_start_layer.bias = self.bias
                self.conv_dw_start_layer.dilation = self.dilation
            else:
                self.conv_dw_start_layer = None

            self.conv_pw_exp_layer.in_channels = self.in_channels
            self.conv_pw_exp_layer.out_channels = mid_chs
            self.conv_pw_exp_layer.kernel_size = 1
            self.conv_pw_exp_layer.stride = 1
            self.conv_pw_exp_layer.padding = self.padding
            self.conv_pw_exp_layer.bias = self.bias

            if self.dw_kernel_size_mid:
                use_aa_mid = self.aa_layer is not None and self.stride > 1
                self.conv_dw_mid_layer.in_channels = mid_chs
                self.conv_dw_mid_layer.out_channels = mid_chs
                self.conv_dw_mid_layer.kernel_size = self.dw_kernel_size_mid
                self.conv_dw_mid_layer.stride = 1 if use_aa_mid else self.stride
                self.conv_dw_mid_layer.padding = self.padding
                self.conv_dw_mid_layer.group_size = self.group_size
                self.conv_dw_mid_layer.bias = self.bias
                self.conv_dw_mid_layer.dilation = self.dilation
            else:
                self.conv_dw_mid_layer = None

            if self.se_layer is not None:
                self.se_layer.in_channels = mid_chs

            self.conv_pw_proj_layer.in_channels = mid_chs
            self.conv_pw_proj_layer.out_channels = self.out_channels
            self.conv_pw_proj_layer.kernel_size = 1
            self.conv_pw_proj_layer.stride = 1
            self.conv_pw_proj_layer.padding = self.padding
            self.conv_pw_proj_layer.bias = self.bias

            if self.dw_kernel_size_end:
                dw_end_stride = self.stride if not (self.dw_kernel_size_start or self.dw_kernel_size_mid) else 1
                if self.aa_layer is not None and dw_end_stride > 1:
                    raise ValueError("Anti-aliasing on the ending depthwise stage with stride > 1 is not supported.")
                self.conv_dw_end_layer.in_channels = self.out_channels
                self.conv_dw_end_layer.out_channels = self.out_channels
                self.conv_dw_end_layer.kernel_size = self.dw_kernel_size_end
                self.conv_dw_end_layer.stride = dw_end_stride
                self.conv_dw_end_layer.padding = self.padding
                self.conv_dw_end_layer.groups = num_groups(self.group_size, self.out_channels)
                self.conv_dw_end_layer.bias = self.bias
                self.conv_dw_end_layer.dilation = self.dilation
            else:
                self.conv_dw_end_layer = None

            if self.aa_layer is not None and hasattr(self.aa_layer, "in_channels"):
                if self.dw_kernel_size_mid and self.stride > 1:
                    self.aa_layer.in_channels = mid_chs
                elif self.dw_kernel_size_start and self.stride > 1 and not self.dw_kernel_size_mid:
                    self.aa_layer.in_channels = self.in_channels

            return self

    class Attention2d(BasicModel):
        """Multi-head attention for 2D NCHW tensors."""

        in_channels: int
        out_channels: Optional[int] = None
        num_heads: int = 32
        bias: bool = True
        expand_first: bool = False
        head_first: bool = False
        attn_drop: float = 0.0
        proj_drop: float = 0.0

        qkv_layer: Union['Conv2dModels.Conv2d'] = Field(
            default_factory=lambda: Conv2dModels.Conv2d(in_channels=-1))
        proj_layer: Union['Conv2dModels.Conv2d'] = Field(
            default_factory=lambda: Conv2dModels.Conv2d(in_channels=-1))

        # derived, not part of the public schema
        dim_attn: Optional[int] = Field(default=None, exclude=True)
        dim_head: Optional[int] = Field(default=None, exclude=True)

        fused_attn:bool = False # TODO


        @model_validator(mode='after')
        def valid_model(self):
            self.out_channels = self.in_channels if self.out_channels is None else self.out_channels
            self.dim_attn = self.out_channels if self.expand_first else self.in_channels
            if self.dim_attn % self.num_heads != 0:
                raise ValueError("dim_attn must be divisible by num_heads.")
            self.dim_head = self.dim_attn // self.num_heads

            self.qkv_layer.in_channels  = self.in_channels
            self.qkv_layer.out_channels = self.dim_attn * 3
            self.qkv_layer.kernel_size  = 1
            self.qkv_layer.bias  = self.bias

            self.proj_layer.in_channels  = self.dim_attn
            self.proj_layer.out_channels = self.out_channels
            self.proj_layer.kernel_size  = 1
            self.proj_layer.bias  = self.bias
            return self


    class MultiQueryAttention2d(BasicModel):
        """Multi Query Attention with optional spatial downsampling."""
        in_channels: int
        out_channels: Optional[int] = None
        num_heads: int = 8
        key_dim: Optional[int] = None
        value_dim: Optional[int] = None
        query_strides: IntOrPair = 1
        kv_stride: int = 1
        dw_kernel_size: int = 3
        dilation: int = 1
        padding: PadArg = ''
        attn_drop: float = 0.0
        proj_drop: float = 0.0
        bias: bool = False
        
        norm_layer: NormModels.BatchNorm2d = Field(
            default_factory=lambda:NormModels.BatchNorm2d(num_features=-1))
        conv_layer: Conv2dModels.Conv2d = Field(
            default_factory=lambda:Conv2dModels.Conv2d(in_channels=-1))
        
        einsum: bool = False

        has_query_strides: bool = Field(default=False)
        fused_attn: bool = False # TODO


        @model_validator(mode='after')
        def valid_model(self):
            self.out_channels = self.in_channels if self.out_channels is None else self.out_channels
            self.key_dim = self.in_channels // self.num_heads if self.key_dim is None else self.key_dim
            self.value_dim = self.in_channels // self.num_heads if self.value_dim is None else self.value_dim
            self.query_strides = to_2tuple(self.query_strides)
            self.has_query_strides = any(s > 1 for s in self.query_strides)
            self.norm_layer.num_features = self.in_channels
            return self

    class MobileAttention(BasicModel):
        """Mobile attention configuration in line with the other Pydantic models."""
        in_channels: int
        out_channels: int
        stride: int = 1
        dw_kernel_size: int = 3
        dilation: int = 1
        group_size: int = 1
        padding: PadArg = ''
        num_heads: Optional[int] = 8
        key_dim: Optional[int] = None
        value_dim: Optional[int] = None
        use_multi_query: bool = False
        query_strides: IntOrPair = (1, 1)
        kv_stride: int = 1
        cpe_dw_kernel_size: int = 3
        noskip: bool = False
        drop_path_rate: float = 0.0
        attn_drop: float = 0.0
        proj_drop: float = 0.0
        layer_scale_init_value: Optional[float] = 1e-5
        bias: bool = Field(default=False, validation_alias='use_bias')
        use_cpe: bool = False
        fused_attn: bool = Field(default_factory=lambda: False) #use_fused_attn()) TODO

        norm_layer: Optional[NormModels.type] = Field(
            default_factory=lambda: NormModels.BatchNorm2d(num_features=-1)
        )
        attn_layer: Optional[Union['Conv2dModels.MultiQueryAttention2d', 'Conv2dModels.Attention2d']] = Field(
            default_factory=lambda:Conv2dModels.Attention2d(in_channels=0)
        )
        conv_cpe_layer: Optional['Conv2dModels.Conv2dDepthwise'] = Field(
                        default_factory=lambda:Conv2dModels.Conv2dDepthwise(in_channels=-1))        
        layer_scale_layer: Optional[LinearModels.LayerScale2d] = None

        @model_validator(mode='after')
        def valid_model(self):
            self.padding = Conv2dModels.DepthwiseSeparableConv._convert_padding(self.padding)
            self.query_strides = to_2tuple(self.query_strides)

            if self.num_heads is None and self.key_dim is None:
                raise ValueError("Either num_heads or key_dim must be set.")
            if self.num_heads is None:
                if self.in_channels % self.key_dim != 0:
                    raise ValueError("in_channels must be divisible by key_dim when num_heads is None.")
                self.num_heads = self.in_channels // self.key_dim
            key_dim = self.key_dim if self.key_dim is not None else self.in_channels // self.num_heads
            value_dim = self.value_dim if self.value_dim is not None else self.in_channels // self.num_heads
            if key_dim <= 0 or value_dim <= 0:
                raise ValueError("key_dim and value_dim must be positive.")

            if self.use_cpe:
                self.conv_cpe_layer.in_channels = self.in_channels
                self.conv_cpe_layer.out_channels = self.in_channels
                self.conv_cpe_layer.kernel_size = self.cpe_dw_kernel_size
                self.conv_cpe_layer.stride = 1
                self.conv_cpe_layer.dilation = self.dilation
                self.conv_cpe_layer.padding = 'same'
                self.conv_cpe_layer.bias = True
            else:
                self.conv_cpe_layer = None

            if self.norm_layer is not None:
                self.norm_layer.num_features = self.in_channels

            fused_attn = bool(self.fused_attn)
            if self.use_multi_query:
                self.attn_layer=Conv2dModels.MultiQueryAttention2d(in_channels=-1)
                if isinstance(self.attn_layer, Conv2dModels.MultiQueryAttention2d):
                    attn = self.attn_layer
                    attn.in_channels = self.in_channels
                    attn.out_channels = self.out_channels
                    attn.num_heads = self.num_heads
                    attn.key_dim = key_dim
                    attn.value_dim = value_dim
                    attn.query_strides = self.query_strides
                    attn.kv_stride = self.kv_stride
                    attn.dw_kernel_size = self.dw_kernel_size
                    attn.dilation = self.dilation
                    attn.padding = self.padding
                    attn.attn_drop = self.attn_drop
                    attn.proj_drop = self.proj_drop
                    attn.bias = self.bias
                    attn.norm_layer = self.norm_layer.model_copy()
                    attn.fused_attn = fused_attn
                    self.attn_layer = attn
                else:
                    raise ValueError("attn_layer must be MultiQueryAttention2d when use_multi_query is True.")
            
            elif isinstance(self.attn_layer, Conv2dModels.Attention2d):
                    attn = self.attn_layer
                    attn.in_channels = self.in_channels
                    attn.out_channels = self.out_channels
                    attn.num_heads = self.num_heads
                    attn.attn_drop = self.attn_drop
                    attn.proj_drop = self.proj_drop
                    attn.bias = self.bias
                    attn.fused_attn = fused_attn
                    self.attn_layer = attn
            else:
                raise ValueError("attn_layer must be Attention2d when use_multi_query is False.")

            if self.layer_scale_init_value is not None:
                self.layer_scale_layer = LinearModels.LayerScale2d(
                    dim=self.out_channels,
                    init_values=self.layer_scale_init_value,
                )
            else:
                self.layer_scale_layer = None

            return self


class Conv2dModules:
    class Module(CommonModule):
        def __init__(self, para, para_cls=None):
            super().__init__(para, Conv2dModels, para_cls)

    class Conv2d(Module):
        def __init__(self,para):
            super().__init__(para)
            self.para:Conv2dModels.Conv2d = self.para
            self.bit = self.para.bit

            if self.para.bit:
                self.conv_para = self.para.model_dump(exclude=['bit','norm','act'])
                self.conv = Bit.Conv2d(**self.conv_para)
            else:
                self.conv_para = self.para.model_dump(exclude=['bit','norm','act','scale_op'])
                self.conv = nn.Conv2d(**self.conv_para)
            self.weight_used = True

        def forward_weight(self,x, weight:Optional[torch.Tensor]=None):
            self.weight_used = False
            if self.bit:
                return Bit.functional.conv2d(x,weight=weight,**self.conv_para)
            return torch.nn.functional.conv2d(x,weight=weight,**self.conv_para)

        def forward(self,x):
            return self.conv(x)

        @torch.no_grad()
        def to_ternary(self):
            if not self.weight_used:
                self.conv = nn.Identity()
            if not self.bit:
                print('to_ternary is off here!')
            if self.weight_used and self.bit:
                self.conv = self.conv.to_ternary()                
            return self
                
    class Conv2dDepthwise(Conv2d):pass
    class Conv2dPointwise(Conv2d):pass

    class Conv2dNorm(Conv2d):
        def __init__(self,para):
            super().__init__(para)
            self.para:Conv2dModels.Conv2dNorm = self.para
            self.para.norm.num_features = self.para.out_channels
            self.norm = self.para.norm.build()

        def forward_weight(self, x, weight = None):
            return self.norm(super().forward_weight(x, weight))

        def forward(self, x):
            return self.norm(super().forward(x))
        
    class Conv2dDepthwiseNorm(Conv2dNorm):pass
    class Conv2dPointwiseNorm(Conv2dNorm):pass

    class Conv2dNormAct(Conv2dNorm):
        def __init__(self,para):
            super().__init__(para)
            self.para:Conv2dModels.Conv2dNormAct = self.para
            self.act = self.para.act.build()

        def forward_weight(self, x, weight = None):
            return self.act(super().forward_weight(x, weight))

        def forward(self, x):
            return self.act(super().forward(x))
        
    class Conv2dDepthwiseNormAct(Conv2dNormAct):pass
    class Conv2dPointwiseNormAct(Conv2dNormAct):pass

    class Conv2dAct(Conv2d):
        def __init__(self,para):
            super().__init__(para)
            self.para:Conv2dModels.Conv2dAct = self.para
            self.act = self.para.act.build()

        def forward_weight(self, x, weight = None):
            return self.act(super().forward_weight(x, weight))

        def forward(self, x):
            return self.act(super().forward(x))
                
    class Conv2dDepthwiseAct(Conv2dAct):pass
    class Conv2dPointwiseAct(Conv2dAct):pass
    
    class SqueezeExcite(Module):
        def __init__(self,para):
            super().__init__(para)
            self.para:Conv2dModels.SqueezeExcite=self.para
            self.conv_reduce:Conv2dModules.Conv2dAct = self.para.conv_reduce_layer.build()
            self.conv_expand:Conv2dModules.Conv2dAct = self.para.conv_expand_layer.build()
            
        def forward(self, x:torch.Tensor):
            x_se = x.mean((2, 3), keepdim=True)
            x_se = self.conv_reduce(x_se)
            x_se = self.conv_expand(x_se)
            return x * x_se
        
        def to_ternary(self,mods=['conv_reduce','conv_reduce']):
            self.drop_path = nn.Identity()
            self.convert_to_ternary(self,mods)
            return self

    class DepthwiseSeparableConv(Module):
        """Depthwise-separable block with Pydantic config and string layer names."""
        def __init__(self,para):
            super().__init__(para)
            self.para:Conv2dModels.DepthwiseSeparableConv=self.para
            self.conv_s2d = self.para.conv_s2d_layer.build() if self.para.conv_s2d_layer else nn.Identity()            
            self.conv_dw = self.para.conv_dw_layer.build()
            self.se = self.para.se_layer.build() if self.para.se_layer else nn.Identity()
            self.aa = self.para.aa_layer.build() if self.para.aa_layer else nn.Identity()
            self.conv_pw = self.para.conv_pw_layer.build()
            # ---- DropPath / stochastic depth ----
            self.drop_path = DropPath(self.para.drop_path_rate) if self.para.drop_path_rate > 0 else nn.Identity()
            self.has_skip = (self.para.in_channels == self.para.out_channels and self.para.stride == 1) and (
                not self.para.noskip
            )

        def pre_conv(self,x):
            x = self.conv_s2d(x) # with norm and act OR Identity
            return x
        
        def mid_conv(self,x):
            x = self.aa(x)
            x = self.se(x)
            return x

        def end_conv(self,x,shortcut):
            x = self.drop_path(x)           
            if self.has_skip:
                x = x + shortcut
            return x

        def forward(self, x):
            shortcut = x
            x = self.pre_conv(x)
            x = self.conv_dw(x) # with norm and act
            x = self.mid_conv(x)           
            x = self.conv_pw(x) # with norm and act
            x = self.end_conv(x,shortcut)
            return x
        
        def to_ternary(self,mods=['conv_s2d','conv_dw','se','aa','conv_pw']):
            self.drop_path = nn.Identity()
            self.convert_to_ternary(self,mods)
            return self

    class InvertedResidual(DepthwiseSeparableConv):
        def __init__(self, para):
            super().__init__(para)
            self.para:Conv2dModels.InvertedResidual=self.para
            self.conv_pw_exp = self.para.conv_pw_exp_layer.build()
        
        def pre_conv(self,x):
            if self.para.conv_s2d_layer is not None:
                x = self.conv_s2d(x) # with norm and act
            x = self.conv_pw_exp(x) # with norm and act
            return x
        
        def to_ternary(self,mods=['conv_s2d','conv_pw','conv_dw','se','aa','conv_pw_exp']):
            self.drop_path = nn.Identity()
            self.convert_to_ternary(self,mods)
            return self

    class EdgeResidual(InvertedResidual):
        """Fused MBConv / EdgeResidual block using Pydantic config."""
        def __init__(self, para):
            super().__init__(para)
            self.para: Conv2dModels.EdgeResidual = self.para

            self.conv_s2d = nn.Identity()
            self.conv_dw = nn.Identity()
            self.conv_exp = self.para.conv_pw_exp_layer.build()
            self.aa = self.para.aa_layer.build() if self.para.aa_layer else nn.Identity()
            self.se = self.para.se_layer.build() if self.para.se_layer else nn.Identity()
            self.conv_pw = self.para.conv_pw_layer.build()
            self.drop_path = DropPath(self.para.drop_path_rate) if self.para.drop_path_rate else nn.Identity()
            self.has_skip = (
                self.para.in_channels == self.para.out_channels and self.para.stride == 1 and not self.para.noskip
            )

        # def forward(self, x):
        #     shortcut = x
        #     x = self.conv_exp(x)
        #     # no conv_dw
        #     x = self.aa(x)
        #     x = self.se(x)
        #     x = self.conv_pw(x)
        #     if self.has_skip:
        #         x = self.drop_path(x) + shortcut
        #     return x

        def to_ternary(self, mods=['conv_exp', 'aa', 'se', 'conv_pw']):
            self.drop_path = nn.Identity()
            self.convert_to_ternary(self, mods)
            return self

    class UniversalInvertedResidual(Module):
        def __init__(self, para):
            super().__init__(para)
            self.para: Conv2dModels.UniversalInvertedResidual = self.para

            self.dw_start = self.para.conv_dw_start_layer.build() if self.para.conv_dw_start_layer else nn.Identity()
            self.pw_exp = self.para.conv_pw_exp_layer.build()
            self.dw_mid = self.para.conv_dw_mid_layer.build() if self.para.conv_dw_mid_layer else nn.Identity()
            self.se = self.para.se_layer.build() if self.para.se_layer else nn.Identity()
            self.pw_proj = self.para.conv_pw_proj_layer.build()
            self.dw_end = self.para.conv_dw_end_layer.build() if self.para.conv_dw_end_layer else nn.Identity()
            self.aa = self.para.aa_layer.build() if self.para.aa_layer else nn.Identity()            
            if self._aa_after == "start":
                self.aa_start = self.aa
            if self._aa_after == "mid":
                self.aa_mid = self.aa

            if self.para.layer_scale_init_value is not None:
                self.layer_scale = LinearModules.LayerScale2d(
                    LinearModels.LayerScale2d(
                        dim=self.para.out_channels,
                        init_values=self.para.layer_scale_init_value,
                    )
                )
            else:
                self.layer_scale = nn.Identity()

            self.drop_path = DropPath(self.para.drop_path_rate) if self.para.drop_path_rate else nn.Identity()
            self.has_skip = (self.para.in_channels == self.para.out_channels and self.para.stride == 1) and (
                not self.para.noskip
            )
            self._aa_after = self._resolve_aa_location()

        def _resolve_aa_location(self):
            if self.para.aa_layer is None:
                return None
            if self.para.dw_kernel_size_mid and self.para.stride > 1:
                return "mid"
            if self.para.dw_kernel_size_start and not self.para.dw_kernel_size_mid and self.para.stride > 1:
                return "start"
            return None

        def forward(self, x):
            shortcut = x
            x = self.dw_start(x)
            x = self.aa_start(x)

            x = self.pw_exp(x)
            x = self.dw_mid(x)
            x = self.aa_mid(x)
            x = self.se(x)
            x = self.pw_proj(x)

            x = self.dw_end(x)
            x = self.layer_scale(x)
            x = self.drop_path(x)
            if self.has_skip:
                x = x + shortcut
            return x

        def to_ternary(self, mods=['dw_start', 'pw_exp', 'dw_mid', 'se', 'aa', 'pw_proj', 'dw_end']):
            self.convert_to_ternary(self,mods)
            self.drop_path = nn.Identity()
            return self

    class CondConvResidual(InvertedResidual):
        """ Inverted residual block w/ CondConv routing"""
        def __init__(self, para):
            super().__init__(para)
            self.para:Conv2dModels.CondConvResidual=self.para
            self.routing_fn = self.para.routing_layer.build()

        def forward(self, x):
            shortcut = x
            # CondConv routing
            pooled_inputs = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
            routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))
            x = self.conv_pw_exp(x, routing_weights)
            x = self.conv_dw(x, routing_weights)
            x = self.se(x)
            x = self.conv_pw(x, routing_weights)
            if self.has_skip:
                x = self.drop_path(x) + shortcut
            return x
        
        def to_ternary(self, mods=['conv_pw','conv_dw','se','conv_pwl','routing_fn']):
            self.drop_path = nn.Identity()
            self.convert_to_ternary(self,mods)
            return self
        
    class MultiQueryAttention2d(Module):
        fused_attn: torch.jit.Final[bool]

        def __init__(self, para):
            super().__init__(para)
            self.para: Conv2dModels.MultiQueryAttention2d = self.para

            self.num_heads = self.para.num_heads
            self.key_dim = self.para.key_dim
            self.value_dim = self.para.value_dim
            self.query_strides = self.para.query_strides
            self.kv_stride = self.para.kv_stride
            self.has_query_strides = self.para.has_query_strides
            self.scale = self.key_dim ** -0.5
            self.fused_attn = self.para.fused_attn
            self.einsum = self.para.einsum

            def norm():
                if self.para.norm_layer is None:
                    return nn.Identity()
                return self.para.norm_layer.model_copy().build()
            
            def create_conv2d(in_channels: int, out_channels: int = -1, kernel_size: IntOrPair = 3,
                              stride: IntOrPair = 1, padding: PadArg = 0, dilation: IntOrPair = 1,
                              groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
                              bit: bool = True, scale_op: str = "median",
                              depthwise=False):
                kwargs = locals()
                # for DW out_channels must be multiple of in_channels as must have out_channels % groups == 0
                if kwargs.pop('depthwise', False): kwargs['groups'] = kwargs['in_channels']
                args = {**self.para.conv_layer.model_dump(),**kwargs}
                return self.para.conv_layer.__class__(**args).build()
            
            self.query = nn.Sequential()
            if self.has_query_strides:
                self.query.add_module('down_pool', PoolModels.AvgPool2d(
                                                    kernel_size=self.query_strides,
                                                    padding=self.para.padding).build())
                self.query.add_module('norm', norm())

            self.query.add_module('proj', create_conv2d(
                self.para.in_channels,
                self.num_heads * self.key_dim,
                kernel_size=1,
                bias=self.para.bias,
            ))

            self.key = nn.Sequential()
            if self.kv_stride > 1:
                self.key.add_module('down_conv', create_conv2d(
                    self.para.in_channels,
                    self.para.in_channels,
                    kernel_size=self.para.dw_kernel_size,
                    stride=self.kv_stride,
                    dilation=self.para.dilation,
                    padding=self.para.padding,
                    depthwise=True,
                ))
                self.key.add_module('norm', norm())

            self.key.add_module('proj', create_conv2d(
                self.para.in_channels,
                self.key_dim,
                kernel_size=1,
                padding=self.para.padding,
                bias=self.para.bias,
            ))

            self.value = nn.Sequential()
            if self.kv_stride > 1:
                self.value.add_module('down_conv', create_conv2d(
                    self.para.in_channels,
                    self.para.in_channels,
                    kernel_size=self.para.dw_kernel_size,
                    stride=self.kv_stride,
                    dilation=self.para.dilation,
                    padding=self.para.padding,
                    depthwise=True,
                ))
                self.value.add_module('norm', norm())

            self.value.add_module('proj', create_conv2d(
                self.para.in_channels,
                self.value_dim,
                kernel_size=1,
                bias=self.para.bias,
            ))

            self.attn_drop = nn.Dropout(self.para.attn_drop)

            self.output = nn.Sequential()
            if self.has_query_strides:
                self.output.add_module('upsample',
                    nn.Upsample(scale_factor=self.query_strides, mode='bilinear', align_corners=False),
                )

            self.output.add_module('proj', create_conv2d(
                self.value_dim * self.num_heads,
                self.para.out_channels,
                kernel_size=1,
                bias=self.para.bias,
            ))
            self.output.add_module('drop', nn.Dropout(self.para.proj_drop))

        def init_weights(self):
            nn.init.xavier_uniform_(self.query.proj.weight)
            nn.init.xavier_uniform_(self.key.proj.weight)
            nn.init.xavier_uniform_(self.value.proj.weight)
            if self.kv_stride > 1:
                nn.init.xavier_uniform_(self.key.down_conv.weight)
                nn.init.xavier_uniform_(self.value.down_conv.weight)
            nn.init.xavier_uniform_(self.output.proj.weight)

        def _reshape_input(self, t: torch.Tensor):
            s = t.shape
            t = t.reshape(s[0], s[1], -1).transpose(1, 2)
            if self.einsum:
                return t
            return t.unsqueeze(1).contiguous()

        def _reshape_projected_query(self, t: torch.Tensor, num_heads: int, key_dim: int):
            s = t.shape
            t = t.reshape(s[0], num_heads, key_dim, -1)
            if self.einsum:
                return t.permute(0, 3, 1, 2).contiguous()
            return t.transpose(-1, -2).contiguous()

        def _reshape_output(self, t: torch.Tensor, num_heads: int, h_px: int, w_px: int):
            s = t.shape
            feat_dim = s[-1] * num_heads
            if not self.einsum:
                t = t.transpose(1, 2)
            return t.reshape(s[0], h_px, w_px, feat_dim).permute(0, 3, 1, 2).contiguous()

        def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
            B, C, H, W = x.shape

            q = self.query(x)
            q = self._reshape_projected_query(q, self.num_heads, self.key_dim)

            k = self.key(x)
            k = self._reshape_input(k)

            v = self.value(x)
            v = self._reshape_input(v)

            if self.einsum:
                attn = torch.einsum('blhk,bpk->blhp', q, k) * self.scale
                if attn_mask is not None:
                    attn = attn + attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                o = torch.einsum('blhp,bpk->blhk', attn, v)
            else:
                if self.fused_attn:
                    o = F.scaled_dot_product_attention(
                        q,k,v, attn_mask=attn_mask,
                        dropout_p=self.attn_drop.p if self.training else 0.,
                    )
                else:
                    q = q * self.scale
                    attn = q @ k.transpose(-1, -2)
                    if attn_mask is not None:
                        attn = attn + attn_mask
                    attn = attn.softmax(dim=-1)
                    attn = self.attn_drop(attn)
                    o = attn @ v

            o = self._reshape_output(o, self.num_heads, H // self.query_strides[0], W // self.query_strides[1])
            x = self.output(o)
            return x

        def to_ternary(self):
            self.attn_drop = nn.Identity()
            self.convert_to_ternary(self)
            return self
        
    class Attention2d(Module):
        def __init__(self, para):
            super().__init__(para)
            self.para: Conv2dModels.Attention2d = self.para

            self.num_heads = self.para.num_heads
            self.dim_head = self.para.dim_head
            self.head_first = self.para.head_first
            self.fused_attn = self.para.fused_attn

            self.qkv = self.para.qkv_layer.build()
            self.attn_drop = nn.Dropout(self.para.attn_drop)
            self.proj = self.para.proj_layer.build()
            self.proj_drop = nn.Dropout(self.para.proj_drop)

        def forward(self, x:torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
            B, C, H, W = x.shape
            res:torch.Tensor = self.qkv(x)

            if self.head_first:
                q, k, v = res.view(B, self.num_heads, self.dim_head * 3, -1).chunk(3, dim=2)
            else:
                q, k, v = res.reshape(B, 3, self.num_heads, self.dim_head, -1).unbind(1)

            if self.fused_attn:
                x = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(-1, -2).contiguous(),
                    k.transpose(-1, -2).contiguous(),
                    v.transpose(-1, -2).contiguous(),
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                ).transpose(-1, -2).reshape(B, -1, H, W)
            else:
                q = q.transpose(-1, -2)
                v = v.transpose(-1, -2)
                attn:torch.Tensor = q @ k * q.size(-1) ** -0.5
                if attn_mask is not None:
                    # NOTE: assumes mask is float and in correct shape
                    attn = attn + attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = (attn @ v).transpose(-1, -2).reshape(B, -1, H, W)

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        def to_ternary(self):
            self.attn_drop = nn.Identity()
            self.convert_to_ternary(self,mods=['qkv', 'proj'])
            return self

    class MobileAttention(Module):
        def __init__(self, para):
            super().__init__(para)
            self.para: Conv2dModels.MobileAttention = self.para

            self.conv_cpe = self.para.conv_cpe_layer.build() if self.para.conv_cpe_layer else None
            self.norm = self.para.norm_layer.build() if self.para.norm_layer is not None else nn.Identity()
            self.attn = self.para.attn_layer.build()
            self.layer_scale = (
                self.para.layer_scale_layer.build() if self.para.layer_scale_layer is not None else nn.Identity()
            )
            self.drop_path = DropPath(self.para.drop_path_rate) if self.para.drop_path_rate else nn.Identity()
            self.has_skip = (
                self.para.stride == 1 and self.para.in_channels == self.para.out_channels and not self.para.noskip
            )

        def forward(self, x):
            if self.conv_cpe is not None:
                x = x + self.conv_cpe(x)

            shortcut = x
            x = self.norm(x)
            x = self.attn(x)
            x = self.layer_scale(x)
            if self.has_skip:
                x = self.drop_path(x) + shortcut
            return x

        def to_ternary(self, mods=['conv_cpe', 'attn']):
            self.drop_path = nn.Identity()
            self.convert_to_ternary(self, mods)
            return self
























#...
