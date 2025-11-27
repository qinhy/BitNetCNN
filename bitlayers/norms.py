from __future__ import annotations

import json
from typing import Optional, Type, Union

from pydantic import BaseModel
from torch import nn
import torch
from torchvision.ops.misc import FrozenBatchNorm2d

class NormdModules:
    class Norm(nn.Module):
        def __init__(
            self,
            para: BaseModel | dict,
            para_cls: Type[BaseModel],
            layer_cls: Type[nn.Module],
        ):
            if isinstance(para, dict):
                para = para_cls(**para)
            self.para = json.loads(para.model_dump_json())

            super().__init__()
            self.norm = layer_cls(**para.model_dump())

        def forward(self, x):
            return self.norm(x)

class NormModels:
    class _BatchNormBase(BaseModel):
        num_features: int
        eps: float = 1e-5
        momentum: float = 0.1
        affine: bool = True
        track_running_stats: bool = True

    class BatchNorm1d(_BatchNormBase):
        def build(self) -> nn.Module:
            return NormdModules.Norm(self, type(self), nn.BatchNorm1d)

    class BatchNorm2d(_BatchNormBase):
        def build(self) -> nn.Module:
            return NormdModules.Norm(self, type(self), nn.BatchNorm2d)

    class BatchNorm3d(_BatchNormBase):
        def build(self) -> nn.Module:
            return NormdModules.Norm(self, type(self), nn.BatchNorm3d)

    class SyncBatchNorm(_BatchNormBase):
        process_group: Optional[object] = None

        def build(self) -> nn.Module:
            return NormdModules.Norm(self, type(self), nn.SyncBatchNorm)

    class _InstanceNormBase(BaseModel):
        num_features: int
        eps: float = 1e-5
        momentum: float = 0.1
        affine: bool = False
        track_running_stats: bool = False

    class InstanceNorm1d(_InstanceNormBase):
        def build(self) -> nn.Module:
            return NormdModules.Norm(self, type(self), nn.InstanceNorm1d)

    class InstanceNorm2d(_InstanceNormBase):
        def build(self) -> nn.Module:
            return NormdModules.Norm(self, type(self), nn.InstanceNorm2d)

    class InstanceNorm3d(_InstanceNormBase):
        def build(self) -> nn.Module:
            return NormdModules.Norm(self, type(self), nn.InstanceNorm3d)

    class GroupNorm(BaseModel):
        num_features: int
        num_groups: int = 32
        eps: float = 1e-5
        affine: bool = True

        def build(self) -> nn.Module:
            return NormdModules.Norm(self, type(self), nn.GroupNorm)

    # class GroupNorm1(BaseModel):
    #     num_features: int
    #     eps: float = 1e-5
    #     affine: bool = True

    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.GroupNorm1)

    class LayerNorm(BaseModel):
        num_features: int  # as same as normalized_shape
        eps: float = 1e-5
        elementwise_affine: bool = True
        bias: bool = True
        data_format="channels_last"

        def build(self) -> nn.Module:
            return NormdModules.Norm(self, type(self),
                                     NormModels._LayerNormModule)

    class _LayerNormModule(nn.Module):
        """ LayerNorm that supports two data formats: channels_last (default) or channels_first. """
        def __init__(self, num_features, eps=1e-6, data_format="channels_last", bias = True):
            super().__init__()
            normalized_shape = num_features
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
            self.eps = eps
            self.data_format = data_format
            if self.data_format not in ["channels_last", "channels_first"]:
                raise NotImplementedError
            self.normalized_shape = (normalized_shape, )

        def forward(self, x:torch.Tensor):
            if self.data_format == "channels_last":
                return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            elif self.data_format == "channels_first":
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x
                if self.bias is not None:
                    x = x + self.bias[:, None, None]
                return x

    class GlobalResponseNorm(BaseModel):
        num_features: int
        
        def build(self) -> nn.Module:
            return NormdModules.Norm(self, type(self),
                                     NormModels._GlobalResponseNormModule)

    class _GlobalResponseNormModule(nn.Module):
        """ GRN (Global Response Normalization) layer """
        def __init__(self, num_features):
            super().__init__()
            dim = num_features
            self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

        def forward(self, x):
            Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
            return self.gamma * (x * Nx) + self.beta + x
        
    # class LayerNormFp32(LayerNorm):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.LayerNormFp32)

    # class LayerNorm2d(LayerNorm):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.LayerNorm2d)

    # class LayerNorm2dFp32(LayerNorm):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.LayerNorm2dFp32)

    # class LayerNormExp2d(LayerNorm):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.LayerNormExp2d)

    class _RmsNormBase(BaseModel):
        num_features: int
        eps: float = 1e-6
        affine: bool = True

    class RmsNorm(_RmsNormBase):
        def build(self) -> nn.Module:
            return NormdModules.Norm(self, type(self), nn.RMSNorm)

    # class RmsNormFp32(_RmsNormBase):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.RMSNormFp32)

    # class RmsNorm2d(_RmsNormBase):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.RMSNorm2d)

    # class RmsNorm2dFp32(_RmsNormBase):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.RMSNorm2dFp32)

    # class _SimpleNormBase(BaseModel):
    #     num_features: int
    #     eps: float = 1e-6
    #     affine: bool = True

    # class SimpleNorm(_SimpleNormBase):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.SimpleNorm)

    # class SimpleNormFp32(_SimpleNormBase):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.SimpleNormFp32)

    # class SimpleNorm2d(_SimpleNormBase):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.SimpleNorm2d)

    # class SimpleNorm2dFp32(_SimpleNormBase):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.SimpleNorm2dFp32)

    class FrozenBatchNorm2d(BaseModel):
        num_features: int
        eps: float = 1e-5

        def build(self) -> nn.Module:
            return NormdModules.Norm(self, type(self), FrozenBatchNorm2d)

    class Identity(BaseModel):
        def build(self) -> nn.Module:
            return NormdModules.Norm(self, type(self), nn.Identity)

    type = Union[BatchNorm1d,BatchNorm2d,BatchNorm3d,SyncBatchNorm,InstanceNorm1d,InstanceNorm2d,InstanceNorm3d,GroupNorm,LayerNorm,RmsNorm,FrozenBatchNorm2d,Identity]
