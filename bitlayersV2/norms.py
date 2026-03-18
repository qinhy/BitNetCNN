from __future__ import annotations

import json
import numbers
from typing import List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field
import torch
from torchvision.ops.misc import FrozenBatchNorm2d
from . import nn

class Norms:
    class _BatchNormBase(nn.Module):
        num_features: int
        eps: float = 1e-5
        momentum: float = 0.1
        affine: bool = True
        track_running_stats: bool = True
        weight: torch.Tensor = Field(default=None, exclude=True)
        bias: Optional[Union[torch.Tensor, torch.nn.Parameter, bool]] = Field(default=True, exclude=True)

    class BatchNorm1d(_BatchNormBase, torch.nn.BatchNorm1d):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.BatchNorm1d.__init__(self, **self.model_dump(exclude=["uuid"]))

    class BatchNorm2d(_BatchNormBase, torch.nn.BatchNorm2d):
        track_running_stats: bool = False
        num_batches_tracked: torch.Tensor = Field(default=None, exclude=True)

        def model_post_init(self, __context):
            super().model_post_init(__context)
            if self.num_features>0:
                self.module_init()

        def module_init(self):
            del self.num_batches_tracked
            torch.nn.BatchNorm2d.__init__(self, **self.model_dump(exclude=["uuid"]))
            torch.nn.BatchNorm2d.to(self, device=self.device)

    class BatchNorm3d(_BatchNormBase, torch.nn.BatchNorm3d):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.BatchNorm3d.__init__(self, **self.model_dump(exclude=["uuid"]))

    class SyncBatchNorm(_BatchNormBase, torch.nn.SyncBatchNorm):
        process_group: Optional[object] = None
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.SyncBatchNorm.__init__(self, **self.model_dump(exclude=["uuid"]))

    class _InstanceNormBase(BaseModel):
        num_features: int
        eps: float = 1e-5
        momentum: float = 0.1
        affine: bool = False
        track_running_stats: bool = False

    class InstanceNorm1d(_InstanceNormBase, torch.nn.InstanceNorm1d):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.InstanceNorm1d.__init__(self, **self.model_dump(exclude=["uuid"]))

    class InstanceNorm2d(_InstanceNormBase, torch.nn.InstanceNorm2d):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.InstanceNorm2d.__init__(self, **self.model_dump(exclude=["uuid"]))

    class InstanceNorm3d(_InstanceNormBase, torch.nn.InstanceNorm3d):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.InstanceNorm3d.__init__(self, **self.model_dump(exclude=["uuid"]))

    class GroupNorm(nn.Module, torch.nn.GroupNorm):
        num_features: int
        num_groups: int = 32
        eps: float = 1e-5
        affine: bool = True

        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.GroupNorm.__init__(self, **self.model_dump(exclude=["uuid"]))

    class LayerNorm(nn.Module):
        """ LayerNorm that supports two data formats: channels_last (default) or channels_first. """
        num_features: int  # as same as normalized_shape
        eps: float = 1e-5
        elementwise_affine: bool = True
        bias: bool = True
        data_format:str = "channels_last"
        
        def model_post_init(self, __context):
            super().model_post_init(__context)
            normalized_shape = self.num_features
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
            self.bias = torch.nn.Parameter(torch.zeros(normalized_shape)) if self.bias else None
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
            
    class GlobalResponseNorm(nn.Module):
        num_features: int

        """ GRN (Global Response Normalization) layer """
        def model_post_init(self, __context):
            super().model_post_init(__context)
            dim = self.num_features
            self.gamma = torch.nn.Parameter(torch.zeros(1, 1, 1, dim))
            self.beta = torch.nn.Parameter(torch.zeros(1, 1, 1, dim))

        def forward(self, x):
            Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
            return self.gamma * (x * Nx) + self.beta + x
        
    class LayerNorm2D(nn.Module):
        num_features: int
        norm_layer:Type[nn.LayerNorm] = nn.LayerNorm

        def model_post_init(self, __context):
            super().model_post_init(__context)
            self.ln = self.norm_layer(self.num_features) if self.norm_layer is not None else nn.Identity()

        def forward(self, x: torch.Tensor):
            """
            x: N C H W
            """
            x = x.permute(0, 2, 3, 1)
            x = self.ln(x)
            x = x.permute(0, 3, 1, 2)
            return x
        
    class RMSNorm(nn.Module, torch.nn.RMSNorm):
        num_features: int
        eps: float = 1e-6
        affine: bool = True

        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.RMSNorm.__init__(self, **self.model_dump(exclude=["uuid"]))

    class RMSNorm2d(nn.Module):
        """ RMSNorm2D for NCHW tensors, w/ fast apex or cast norm if available

        NOTE: It's currently (2025-05-10) faster to use an eager 2d kernel that does reduction
        on dim=1 than to permute and use internal PyTorch F.rms_norm, this may change if something
        like https://github.com/pytorch/pytorch/pull/150576 lands.
        """
        num_features: int
        eps: float = 1e-6
        affine: bool = True
        normalized_shape: Tuple[int, ...]
        eps: float = 1e-6
        elementwise_affine: bool = False
        _fast_norm: bool = False

        def model_post_init(self, __context):
            super().model_post_init(__context)
            normalized_shape = channels = self.num_features
            if isinstance(normalized_shape, numbers.Integral):
                # mypy error: incompatible types in assignment
                normalized_shape = (normalized_shape,)  # type: ignore[assignment]
            self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
            self.eps = self.eps
            self.elementwise_affine = self.affine
            self._fast_norm = False # is_fast_norm()  # can't script unless we have these flags here (no globals)

            if self.elementwise_affine:
                self.weight = torch.nn.Parameter(torch.empty(self.normalized_shape))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            if self.elementwise_affine:
                torch.nn.init.ones_(self.weight)

        @staticmethod
        def rms_norm2d(
            x: torch.Tensor,
            normalized_shape: List[int],
            weight: Optional[torch.Tensor] = None,
            eps: float = 1e-5,
        ):
            assert len(normalized_shape) == 1
            v = x.pow(2)
            v = torch.mean(v, dim=1, keepdim=True)
            x = x * torch.rsqrt(v + eps)
            if weight is not None:
                x = x * weight.reshape(1, -1, 1, 1)
            return x
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # NOTE fast norm fallback needs our rms norm impl, so both paths through here.
            # Since there is no built-in PyTorch impl, always use APEX RMSNorm if is installed.
            if self._fast_norm:
                pass
                # x = fast_rms_norm2d(x, self.normalized_shape, self.weight, self.eps)
            else:
                x = self.rms_norm2d(x, self.normalized_shape, self.weight, self.eps)
            return x

    class FrozenBatchNorm2d(BaseModel):
        num_features: int
        eps: float = 1e-5

        def build(self) -> nn.Module:
            return FrozenBatchNorm2d(**self.model_dump(exclude=["uuid"]))

    class Identity(BaseModel):
        def build(self) -> nn.Module:
            return nn.Identity(**self.model_dump(exclude=["uuid"]))

    type = Union[BatchNorm1d,BatchNorm2d,BatchNorm3d,SyncBatchNorm,
                 InstanceNorm1d,InstanceNorm2d,InstanceNorm3d,GroupNorm,LayerNorm,
                 RMSNorm,RMSNorm2d,FrozenBatchNorm2d,Identity]

    cls = Union[Type[BatchNorm1d],Type[BatchNorm2d],Type[BatchNorm3d],Type[SyncBatchNorm],
                Type[InstanceNorm1d],Type[InstanceNorm2d],Type[InstanceNorm3d],Type[GroupNorm],Type[LayerNorm],
                Type[RMSNorm],Type[RMSNorm2d],Type[FrozenBatchNorm2d],Type[Identity]]


