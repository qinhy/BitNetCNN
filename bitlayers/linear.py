from __future__ import annotations

import json
from typing import Union

from pydantic import BaseModel
from torch import nn
import torch

from .acts import ActModels
from .bit import Bit
from .norms import NormModels


class LinearModels:
    class BasicModel(BaseModel):
        bit: bool = True
        scale_op: str = "median"

        def build(self):
            mod = LinearModules
            return mod.__dict__[f'{self.__class__.__name__}'](self)

    class Linear(BasicModel):
        in_features: int
        out_features: int
        bias: bool = True

        bit: bool = True
        scale_op: str = "median"

    class LinearBn(Linear):
        bn: NormModels.type

    class LinearAct(Linear):
        act: ActModels.type

    class LinearBnAct(Linear):
        bn: NormModels.type
        act: ActModels.type

    class LayerScale2d(BaseModel):
        dim: int
        init_values: float = 1e-5
        inplace: bool = False

        def build(self):
            mod = LinearModules
            return mod.__dict__[f'{self.__class__.__name__}'](self)

    type = Union[Linear,LinearBn,LinearAct,LinearBnAct,LayerScale2d]


class LinearModules:
    class Module(nn.Module):
        def __init__(self, para: BaseModel, para_cls):
            if isinstance(para, dict):
                para = para_cls(**para)
            self.para = json.loads(para.model_dump_json())
            super().__init__()

        @torch.no_grad()
        def to_ternary(self,mods=[]):
            for m in mods:
                if self.__dict__[m] and hasattr(self.__dict__[m],'to_ternary'):
                    setattr(self,m,self.__dict__[m].to_ternary())
            return self

    class Linear(Module):
        def __init__(self, para: LinearModels.Linear, para_cls=LinearModels.Linear):
            super().__init__(para, para_cls)
            self.para: LinearModels.Linear = self.para

            if para.bit:
                self.linear = Bit.Linear(
                    para.in_features, para.out_features,
                    bias=para.bias, scale_op=para.scale_op,
                )
            else:
                self.linear = nn.Linear(
                    para.in_features, para.out_features,
                    para.bias,
                )

        def forward(self,x):
            return self.linear(x)

        @torch.no_grad()
        def to_ternary(self):
            if hasattr(self.linear,'to_ternary'):return self.linear.to_ternary()
            print('to_ternary is no support!')

    class LinearBn(Linear):
        def __init__(self, para: LinearModels.LinearBn, para_cls=LinearModels.LinearBn):
            super().__init__(para, para_cls)
            self.bn = para.bn.build()

        def forward(self, x):
            return self.bn(super().forward(x))

        @torch.no_grad()
        def to_ternary(self):
            return nn.Sequential(super().to_ternary(),self.bn)

    class LinearAct(Linear):
        def __init__(self, para: LinearModels.LinearAct, para_cls=LinearModels.LinearAct):
            super().__init__(para, para_cls)
            self.act = para.act.build()

        def forward(self, x):
            return self.act(super().forward(x))

        @torch.no_grad()
        def to_ternary(self):
            return nn.Sequential(super().to_ternary(),self.act)

    class LinearBnAct(Linear):
        def __init__(self, para: LinearModels.LinearBnAct, para_cls=LinearModels.LinearBnAct):
            super().__init__(para, para_cls)
            self.bn = para.bn.build()
            self.act = para.act.build()

        def forward(self, x):
            return self.act(self.bn(super().forward(x)))

        @torch.no_grad()
        def to_ternary(self):
            return nn.Sequential(super().to_ternary(),self.bn,self.act)

    class LayerScale2d(Module):
        def __init__(self, para: LinearModels.LayerScale2d, para_cls=LinearModels.LayerScale2d):
            super().__init__(para, para_cls)
            self.para: LinearModels.LayerScale2d = self.para
            self.inplace = para.inplace
            self.gamma = nn.Parameter(para.init_values * torch.ones(para.dim))

        def forward(self, x):
            gamma = self.gamma.view(1, -1, 1, 1)
            return x.mul_(gamma) if self.inplace else x * gamma
