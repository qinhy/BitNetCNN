

import inspect
from typing import Optional, Sequence, Tuple, Union

from pydantic import Field, field_serializer, field_validator, model_validator
import torch

from .pool import Pools
from .helpers import make_divisible, to_2tuple, convert_padding
from .norms import Norms
from .acts import Acts
from . import nn


IntOrPair = Union[int, Tuple[int, int]]
KernelSizeArg = Union[int, Tuple[int, int], Sequence[int]]
PadArg = Union[str, int, Tuple[int, int]]

class Convs:

    class ConvTranspose2d(nn.Module, torch.nn.ConvTranspose2d):
        in_channels: int
        out_channels: int = -1 # just a place holder not valid
        kernel_size: IntOrPair = 3
        stride: IntOrPair = 1
        padding: PadArg = 0
        output_padding: IntOrPair = 0
        dilation: IntOrPair = 1
        groups: int = 1
        bias: bool = True
        padding_mode: str = 'zeros'

        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.ConvTranspose2d.__init__(self, **self.model_dump(exclude=["uuid"]))

    class Conv2dDepthwise(nn.Conv2d):
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
        
        def module_init(self):
            self.valid_model()
            return super().module_init()
        
    class Conv2dPointwise(nn.Conv2d):
        @model_validator(mode='after')
        def valid_model(self):
            self.kernel_size = 1
            self.padding = 0
            self.groups = 1
            return self
        
        def module_init(self):
            self.valid_model()
            return super().module_init()

    class Conv2dNorm(nn.Conv2d):
        norm: Union[Norms.type, Norms.cls]
        
        @field_validator("norm", mode="before")
        @classmethod
        def parse_norm(cls, v):
            if inspect.isclass(v):
                return v
            return Norms.parse(v)
        
        @field_serializer("norm")
        def serialize_norm(self, v):            
            if inspect.isclass(v):
                return {"class": v.__class__.__name__,}
            else:
                return v.model_dump()
        
        def module_init(self):
            super().module_init()
            if inspect.isclass(self.norm):
                self.norm = self.norm(num_features=self.out_channels,
                                      device=self.device)
                self.add_module("norm",self.norm)
        
        def forward(self, x):
            return self.norm(super().forward(x))

    class Conv2dAct(nn.Conv2d):
        act: Union[Acts.type, Acts.cls]
        @field_validator("act", mode="before")
        @classmethod
        def parse_act(cls, v):
            if inspect.isclass(v):
                return v(inplace=True)
            return Acts.parse(v)
        
        def forward(self, x):
            return self.act(super().forward(x))

    class Conv2dNormAct(nn.Conv2d):
        norm: Union[Norms.type, Norms.cls]
        act: Union[Acts.type, Acts.cls]
        
        @field_validator("act", mode="before")
        @classmethod
        def parse_act(cls, v):
            if inspect.isclass(v):
                return v(inplace=True)
            return Acts.parse(v)

        @field_validator("norm", mode="before")
        @classmethod
        def parse_norm(cls, v):
            if inspect.isclass(v):
                return v
            return Norms.parse(v)
        
        @field_serializer("norm")
        def serialize_norm(self, v):            
            if inspect.isclass(v):
                return {"class": v.__class__.__name__,}
            else:
                return v.model_dump()
        
        def module_init(self):
            super().module_init()
            if inspect.isclass(self.norm):
                self.norm = self.norm(num_features=self.out_channels,
                                      device=self.device)
                self.add_module("norm",self.norm)
                
        def forward(self, x):
            return self.act(self.norm(super().forward(x)))

    class Conv2dDepthwiseNorm(Conv2dDepthwise,Conv2dNorm):pass
    class Conv2dDepthwiseAct(Conv2dDepthwise,Conv2dAct):pass
    class Conv2dDepthwiseNormAct(Conv2dDepthwise,Conv2dNormAct):pass

    class Conv2dPointwiseNorm(Conv2dPointwise,Conv2dNorm):pass
    class Conv2dPointwiseAct(Conv2dPointwise,Conv2dAct):pass
    class Conv2dPointwiseNormAct(Conv2dPointwise,Conv2dNormAct):pass


    class SqueezeExcite(nn.Module):
        in_channels: int

        conv_reduce_layer: Optional['Convs.Conv2dAct'] = Field(
                default_factory=lambda:Convs.Conv2dAct(
                                            in_channels=-1, act=Acts.ReLU(),))
        
        conv_expand_layer: Optional['Convs.Conv2dAct'] = Field(
                default_factory=lambda:Convs.Conv2dAct(
                                            in_channels=-1, act=Acts.Sigmoid(),))
        rd_ratio: float = 0.25
        rd_channels: Optional[int] = None
        
        @model_validator(mode='after')
        def valid_model(self):
            if self.rd_channels is None or self.rd_channels<1:
                self.rd_channels = round(self.in_channels * self.rd_ratio)
            
            self.conv_reduce_layer.in_channels  = self.in_channels
            self.conv_reduce_layer.out_channels = self.rd_channels
            self.conv_reduce_layer.module_init()

            self.conv_expand_layer.in_channels  = self.rd_channels
            self.conv_expand_layer.out_channels = self.in_channels
            self.conv_expand_layer.module_init()
            return self
        
    class DepthwiseSeparableConv(nn.Module):
        in_channels: int
        out_channels: int

        kernel_size: Optional[IntOrPair] = None
        dw_kernel_size: int = 3
        pw_kernel_size: int = 1

        group_size: int = 1
        padding: PadArg = 'same'
        dilation: IntOrPair = 1

        stride: IntOrPair = 1
        bias: bool = False

        noskip: bool = False
        drop_path_rate: float = 0.0

        conv_s2d_layer: Optional['Convs.Conv2dNormAct'] = None

        conv_dw_layer: Optional['Convs.Conv2dDepthwiseNormAct'] = Field(
            default_factory=lambda: Convs.Conv2dDepthwiseNormAct(
                in_channels=-1,
                norm=Norms.BatchNorm2d(num_features=-1),
                act=Acts.ReLU(),
            )
        )

        se_layer: Optional['Convs.SqueezeExcite'] = None
        aa_layer: Optional[Pools.type] = None

        conv_pw_layer: Optional['Convs.Conv2dPointwiseNormAct'] = Field(
            default_factory=lambda: Convs.Conv2dPointwiseNormAct(
                in_channels=-1,
                norm=Norms.BatchNorm2d(num_features=-1),
                act=Acts.ReLU(),
            )
        )

        has_skip: bool = False
        drop_path: nn.Module = None

        @model_validator(mode='after')
        def valid_model(self, module_init=True):
            self.kernel_size = None
            self.padding = convert_padding(self.padding)
            dw_kernel_local = self.dw_kernel_size
            dw_pad_type = self.padding


            if self.conv_s2d_layer is not None:
                self.conv_s2d_layer.in_channels = self.in_channels
                self.conv_s2d_layer.out_channels = int(self.in_channels * 4)
                self.conv_s2d_layer.norm.num_features = self.conv_s2d_layer.out_channels
                self.conv_s2d_layer.kernel_size = 2
                self.conv_s2d_layer.stride = 2
                self.conv_s2d_layer.padding = 'same'
                if type(self.conv_s2d_layer.bias) is not torch.nn.Parameter:
                    self.conv_s2d_layer.bias = self.bias
                self.conv_s2d_layer.device = self.device
                self.conv_s2d_layer.dtype = self.dtype
                
                if dw_kernel_local in (3,4):
                    dw_pad_type = 'same'
                # we already downsampled
                self.aa_layer = None
            else:
                self.conv_s2d_layer = None

            use_aa = (self.aa_layer is None)
            in_channels = self.conv_s2d_layer.in_channels if hasattr(self.conv_s2d_layer, 'in_channels') else self.in_channels

            # Depthwise conv
            self.conv_dw_layer.in_channels = in_channels
            self.conv_dw_layer.out_channels = in_channels
            self.conv_dw_layer.norm.num_features = in_channels
            self.conv_dw_layer.kernel_size = dw_kernel_local
            self.conv_dw_layer.stride = 1 if use_aa else self.stride
            self.conv_dw_layer.padding = 1 if dw_kernel_local==3 else dw_pad_type
            self.conv_dw_layer.group_size = self.group_size
            if type(self.conv_dw_layer.bias) is not torch.nn.Parameter:
                self.conv_dw_layer.bias = self.bias
            self.conv_dw_layer.device = self.device
            self.conv_dw_layer.dtype = self.dtype

            # Pointwise conv
            self.conv_pw_layer.in_channels = in_channels
            self.conv_pw_layer.out_channels = self.out_channels
            self.conv_pw_layer.norm.num_features = self.out_channels
            self.conv_pw_layer.kernel_size = self.pw_kernel_size # always 1
            self.conv_pw_layer.stride = 1
            self.conv_pw_layer.padding = self.padding
            self.conv_pw_layer.groups = 1
            if type(self.conv_pw_layer.bias) is not torch.nn.Parameter:
                self.conv_pw_layer.bias = self.bias
            self.conv_pw_layer.device = self.device
            self.conv_pw_layer.dtype = self.dtype
             
            if self.se_layer:
                if self.aa_layer:
                    # aa_layer is a pooling layer in_channels is out_channels
                    se_in_channels = self.aa_layer.in_channels
                else:
                    se_in_channels = self.conv_dw_layer.out_channels
                self.se_layer:Convs.SqueezeExcite = self.se_layer.__class__(in_channels=se_in_channels,
                                                                  rd_ratio=self.se_layer.rd_ratio,
                                                                  rd_channels=self.se_layer.rd_channels,
                )
                self.se_layer.device = self.device
                self.se_layer.dtype = self.dtype

            # ---- DropPath / stochastic depth ----
            self.drop_path = nn.DropPath(drop_prob=self.drop_path_rate) if self.drop_path_rate > 0 else nn.Identity()
            self.has_skip = (self.in_channels == self.out_channels and self.stride == 1) and (
                not self.noskip
            )
            if module_init:
                self.module_init()
                
            return self
        
        def module_init(self):            
            if self.conv_s2d_layer:self.conv_s2d_layer.module_init()
            if self.conv_dw_layer:self.conv_dw_layer.module_init()
            if self.conv_pw_layer:self.conv_pw_layer.module_init()        

        def pre_conv(self,x):
            if self.conv_s2d_layer:x = self.conv_s2d_layer(x) # with norm and act OR Identity
            return x
        
        def mid_conv(self,x):
            if self.aa_layer:x = self.aa_layer(x)
            if self.se_layer:x = self.se_layer(x)
            return x

        def end_conv(self,x,shortcut):
            x = self.drop_path(x)           
            if self.has_skip:
                x = x + shortcut
            return x

        def forward(self, x:torch.Tensor):
            shortcut = x
            x = self.pre_conv(x)
            x = self.conv_dw_layer(x) # with norm and act
            x = self.mid_conv(x)           
            x = self.conv_pw_layer(x) # with norm and act
            x = self.end_conv(x,shortcut)
            return x
        
    class InvertedResidual(DepthwiseSeparableConv):
        """InvertedResidual (MBConv) block using string layer names."""
        exp_ratio: float = 1.0
        exp_kernel_size: int = 1

        conv_pw_exp_layer: Optional['Convs.Conv2dPointwiseNormAct'] = None
        # Field(
        #     default_factory=lambda: Convs.Conv2dPointwiseNormAct(
        #         in_channels=-1,
        #         norm=Norms.BatchNorm2d(num_features=-1),
        #         act=Acts.ReLU(),
        #     )
        # )

        conv_pw_layer: Optional['Convs.Conv2dPointwiseNorm'] = None
        # Field(
        #     default_factory=lambda: Convs.Conv2dPointwiseNorm(
        #         in_channels=-1,
        #         norm=Norms.BatchNorm2d(num_features=-1),
        #     )
        # )

        @model_validator(mode='after')
        def valid_model(self, module_init=True):
            super().valid_model(module_init=False)
            in_channels = self.in_channels
            mid_chs = make_divisible(in_channels * self.exp_ratio)

            # Point-wise expansion
            self.conv_pw_exp_layer.in_channels = in_channels
            self.conv_pw_exp_layer.out_channels = mid_chs
            self.conv_pw_exp_layer.kernel_size = self.exp_kernel_size
            self.conv_pw_exp_layer.stride = 1 if self.aa_layer else self.stride
            self.conv_pw_exp_layer.padding = self.padding
            self.conv_pw_exp_layer.dilation = self.dilation
            if type(self.conv_pw_exp_layer.bias) is not torch.nn.Parameter:
                self.conv_pw_exp_layer.bias = self.bias

            # Depth-wise convolution
            self.conv_dw_layer.in_channels = mid_chs
            self.conv_dw_layer.out_channels = mid_chs
            
            # Point-wise linear projection
            self.conv_pw_layer.in_channels = mid_chs
            self.conv_pw_layer.padding = self.padding
            if type(self.conv_pw_layer.bias) is not torch.nn.Parameter:
                self.conv_pw_layer.bias = self.bias

            if module_init:
                self.module_init()
            return self
        
        def module_init(self):
            if self.conv_s2d_layer:self.conv_s2d_layer.module_init()
            if self.conv_pw_exp_layer:self.conv_pw_exp_layer.module_init()
            if self.conv_dw_layer:self.conv_dw_layer.module_init()
            if self.conv_pw_layer:self.conv_pw_layer.module_init()
            
        def pre_conv(self,x):
            if self.conv_s2d_layer is not None:
                x = self.conv_s2d_layer(x) # with norm and act
            x = self.conv_pw_exp_layer(x) # with norm and act
            return x

    class CondConvResidual(InvertedResidual):
        num_experts:int
        routing_layer: nn.Linear =  Field(
            default_factory=lambda: nn.Linear(in_features=-1)
        )
        
        @model_validator(mode='after')
        def valid_model(self):
            self.routing_layer.in_features = self.in_channels
            self.routing_layer.out_features = self.num_experts
        
    class EdgeResidual(InvertedResidual):
        """Fused MBConv / EdgeResidual block configured via Pydantic model."""
        force_in_channels: int = 0
        bias: bool = False


        conv_pw_layer: Optional['Convs.Conv2dPointwiseNormAct'] = Field(
            default_factory=lambda: Convs.Conv2dPointwiseNormAct(
                in_channels=-1,
                norm=Norms.BatchNorm2d(num_features=-1),
                act=Acts.ReLU(),
            )
        )
        
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
    
    class ResNetBasicBlock(nn.Module):
        in_channels: int
        out_channels: int
        stride: int = 1
        dilation: int = 1
        padding: PadArg = 'same'
        drop_path_rate: float = 0.0
        noskip: bool = False
        act_layer: Acts.type = Field(default_factory=Acts.ReLU)

        conv1_layer: Optional['Convs.Conv2dNormAct'] = Field(
            default_factory=lambda: Convs.Conv2dNormAct(
                in_channels=-1,
                norm=Norms.BatchNorm2d(num_features=-1),
                act=Acts.ReLU(),
            )
        )
        conv2_layer: Optional['Convs.Conv2dNorm'] = Field(
            default_factory=lambda: Convs.Conv2dNorm(
                in_channels=-1,
                norm=Norms.BatchNorm2d(num_features=-1),
            )
        )
        shortcut_layer: Optional['Convs.Conv2dNorm'] = Field(default=None)

        @model_validator(mode='after')
        def valid_model(self):
            self.padding = convert_padding(self.padding)
            conv_pad = self.dilation if self.padding == 'same' else self.padding

            self.conv1_layer.in_channels = self.in_channels
            self.conv1_layer.out_channels = self.out_channels
            self.conv1_layer.kernel_size = 3
            self.conv1_layer.stride = self.stride
            self.conv1_layer.padding = conv_pad
            self.conv1_layer.dilation = self.dilation
            self.conv1_layer.bias = False

            self.conv2_layer.in_channels = self.out_channels
            self.conv2_layer.out_channels = self.out_channels
            self.conv2_layer.kernel_size = 3
            self.conv2_layer.stride = 1
            self.conv2_layer.padding = conv_pad
            self.conv2_layer.dilation = self.dilation
            self.conv2_layer.bias = False

            need_downsample = self.stride != 1 or self.in_channels != self.out_channels
            if self.shortcut_layer is None and need_downsample:
                self.shortcut_layer = Convs.Conv2dNorm(
                    in_channels=-1,
                    norm=Norms.BatchNorm2d(num_features=-1),
                )

            if self.shortcut_layer is not None:
                self.shortcut_layer.in_channels = self.in_channels
                self.shortcut_layer.out_channels = self.out_channels
                self.shortcut_layer.kernel_size = 1
                self.shortcut_layer.stride = self.stride
                self.shortcut_layer.padding = 0
                self.shortcut_layer.bias = False

            return self

    class ResNetBottleneck(nn.Module):
        in_channels: int
        out_channels: int
        stride: int = 1
        dilation: int = 1
        padding: PadArg = 'same'
        drop_path_rate: float = 0.0
        noskip: bool = False
        bottleneck_ratio: int = 4
        mid_channels: Optional[int] = None
        act_layer: Acts.type = Field(default_factory=Acts.ReLU)

        conv_reduce_layer: Optional['Convs.Conv2dNormAct'] = Field(
            default_factory=lambda: Convs.Conv2dNormAct(
                in_channels=-1,
                norm=Norms.BatchNorm2d(num_features=-1),
                act=Acts.ReLU(),
            )
        )
        conv_transform_layer: Optional['Convs.Conv2dNormAct'] = Field(
            default_factory=lambda: Convs.Conv2dNormAct(
                in_channels=-1,
                norm=Norms.BatchNorm2d(num_features=-1),
                act=Acts.ReLU(),
            )
        )
        conv_expand_layer: Optional['Convs.Conv2dNorm'] = Field(
            default_factory=lambda: Convs.Conv2dNorm(
                in_channels=-1,
                norm=Norms.BatchNorm2d(num_features=-1),
            )
        )
        shortcut_layer: Optional['Convs.Conv2dNorm'] = Field(default=None)

        @model_validator(mode='after')
        def valid_model(self):
            self.padding = convert_padding(self.padding)
            conv_pad = self.dilation if self.padding == 'same' else self.padding

            if self.mid_channels is None:
                if self.out_channels % self.bottleneck_ratio != 0:
                    raise ValueError("out_channels must be divisible by bottleneck_ratio when mid_channels not set.")
                self.mid_channels = self.out_channels // self.bottleneck_ratio

            self.conv_reduce_layer.in_channels = self.in_channels
            self.conv_reduce_layer.out_channels = self.mid_channels
            self.conv_reduce_layer.kernel_size = 1
            self.conv_reduce_layer.stride = 1
            self.conv_reduce_layer.padding = 0
            self.conv_reduce_layer.bias = False

            self.conv_transform_layer.in_channels = self.mid_channels
            self.conv_transform_layer.out_channels = self.mid_channels
            self.conv_transform_layer.kernel_size = 3
            self.conv_transform_layer.stride = self.stride
            self.conv_transform_layer.padding = conv_pad
            self.conv_transform_layer.dilation = self.dilation
            self.conv_transform_layer.bias = False

            self.conv_expand_layer.in_channels = self.mid_channels
            self.conv_expand_layer.out_channels = self.out_channels
            self.conv_expand_layer.kernel_size = 1
            self.conv_expand_layer.stride = 1
            self.conv_expand_layer.padding = 0
            self.conv_expand_layer.bias = False

            need_downsample = self.stride != 1 or self.in_channels != self.out_channels
            if self.shortcut_layer is None and need_downsample:
                self.shortcut_layer = Convs.Conv2dNorm(
                    in_channels=-1,
                    norm=Norms.BatchNorm2d(num_features=-1),
                )

            if self.shortcut_layer is not None:
                self.shortcut_layer.in_channels = self.in_channels
                self.shortcut_layer.out_channels = self.out_channels
                self.shortcut_layer.kernel_size = 1
                self.shortcut_layer.stride = self.stride
                self.shortcut_layer.padding = 0
                self.shortcut_layer.bias = False

            return self
    