from __future__ import annotations

import json
from typing import Optional, Type

from pydantic import BaseModel
from torch import nn

from timmlayers.norm import (
    GroupNorm as _GroupNorm,
    GroupNorm1 as _GroupNorm1,
    LayerNorm as _LayerNorm,
    LayerNorm2d as _LayerNorm2d,
    LayerNorm2dFp32 as _LayerNorm2dFp32,
    LayerNormExp2d as _LayerNormExp2d,
    LayerNormFp32 as _LayerNormFp32,
    RmsNorm as _RmsNorm,
    RmsNorm2d as _RmsNorm2d,
    RmsNorm2dFp32 as _RmsNorm2dFp32,
    RmsNormFp32 as _RmsNormFp32,
    SimpleNorm as _SimpleNorm,
    SimpleNorm2d as _SimpleNorm2d,
    SimpleNorm2dFp32 as _SimpleNorm2dFp32,
    SimpleNormFp32 as _SimpleNormFp32,
)
from torchvision.ops.misc import FrozenBatchNorm2d as _FrozenBatchNorm2d


class NormControllers:
    class NormController(nn.Module):
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


class _NormBase(BaseModel):
    def _build(self, layer_cls: Type[nn.Module]) -> nn.Module:
        return NormControllers.NormController(self, type(self), layer_cls)


class NormModels:
    class _BatchNormBase(_NormBase):
        num_features: int
        eps: float = 1e-5
        momentum: float = 0.1
        affine: bool = True
        track_running_stats: bool = True

    class BatchNorm1d(_BatchNormBase):
        def build(self) -> nn.Module:
            return self._build(nn.BatchNorm1d)

    class BatchNorm2d(_BatchNormBase):
        def build(self) -> nn.Module:
            return self._build(nn.BatchNorm2d)

    class BatchNorm3d(_BatchNormBase):
        def build(self) -> nn.Module:
            return self._build(nn.BatchNorm3d)

    class SyncBatchNorm(_BatchNormBase):
        process_group: Optional[object] = None

        def build(self) -> nn.Module:
            return self._build(nn.SyncBatchNorm)

    class _InstanceNormBase(_NormBase):
        num_features: int
        eps: float = 1e-5
        momentum: float = 0.1
        affine: bool = False
        track_running_stats: bool = False

    class InstanceNorm1d(_InstanceNormBase):
        def build(self) -> nn.Module:
            return self._build(nn.InstanceNorm1d)

    class InstanceNorm2d(_InstanceNormBase):
        def build(self) -> nn.Module:
            return self._build(nn.InstanceNorm2d)

    class InstanceNorm3d(_InstanceNormBase):
        def build(self) -> nn.Module:
            return self._build(nn.InstanceNorm3d)

    class GroupNorm(_NormBase):
        num_channels: int
        num_groups: int = 32
        eps: float = 1e-5
        affine: bool = True

        def build(self) -> nn.Module:
            return self._build(_GroupNorm)

    class GroupNorm1(_NormBase):
        num_channels: int
        eps: float = 1e-5
        affine: bool = True

        def build(self) -> nn.Module:
            return self._build(_GroupNorm1)

    class LayerNorm(_NormBase):
        num_channels: int
        eps: float = 1e-6
        affine: bool = True

        def build(self) -> nn.Module:
            return self._build(_LayerNorm)

    class LayerNormFp32(LayerNorm):
        def build(self) -> nn.Module:
            return self._build(_LayerNormFp32)

    class LayerNorm2d(LayerNorm):
        def build(self) -> nn.Module:
            return self._build(_LayerNorm2d)

    class LayerNorm2dFp32(LayerNorm):
        def build(self) -> nn.Module:
            return self._build(_LayerNorm2dFp32)

    class LayerNormExp2d(LayerNorm):
        def build(self) -> nn.Module:
            return self._build(_LayerNormExp2d)

    class _RmsNormBase(_NormBase):
        num_channels: int
        eps: float = 1e-6
        affine: bool = True

    class RmsNorm(_RmsNormBase):
        def build(self) -> nn.Module:
            return self._build(_RmsNorm)

    class RmsNormFp32(_RmsNormBase):
        def build(self) -> nn.Module:
            return self._build(_RmsNormFp32)

    class RmsNorm2d(_RmsNormBase):
        def build(self) -> nn.Module:
            return self._build(_RmsNorm2d)

    class RmsNorm2dFp32(_RmsNormBase):
        def build(self) -> nn.Module:
            return self._build(_RmsNorm2dFp32)

    class _SimpleNormBase(_NormBase):
        num_channels: int
        eps: float = 1e-6
        affine: bool = True

    class SimpleNorm(_SimpleNormBase):
        def build(self) -> nn.Module:
            return self._build(_SimpleNorm)

    class SimpleNormFp32(_SimpleNormBase):
        def build(self) -> nn.Module:
            return self._build(_SimpleNormFp32)

    class SimpleNorm2d(_SimpleNormBase):
        def build(self) -> nn.Module:
            return self._build(_SimpleNorm2d)

    class SimpleNorm2dFp32(_SimpleNormBase):
        def build(self) -> nn.Module:
            return self._build(_SimpleNorm2dFp32)

    class FrozenBatchNorm2d(_NormBase):
        num_features: int
        eps: float = 1e-5

        def build(self) -> nn.Module:
            return self._build(_FrozenBatchNorm2d)

    class Identity(_NormBase):
        def build(self) -> nn.Module:
            return self._build(nn.Identity)
