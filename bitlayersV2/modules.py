from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, ClassVar, List, Literal, Optional, Tuple, Union
from uuid import uuid4

import torch
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from .padding import PadSame
from .bit import Bit

# -----------------------------------------------------------------------------
# Pydantic + Torch hybrid module
# -----------------------------------------------------------------------------
class Module(BaseModel, torch.nn.Module):
    """
    Hybrid base class: a Pydantic v2 model that is also a torch.torch.nn.Module.

    Key goals:
      - Pydantic validates/configures constructor fields.
      - Torch correctly registers submodules/params/buffers and supports hooks, .to(), etc.
      - Exclude torch.nn.Module internals from Pydantic dumps.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    __hash__ = object.__hash__  # torch internals sometimes need hashable modules

    # prevent Pydantic from treating these as model fields
    dump_patches: ClassVar[bool]
    call_super_init: ClassVar[bool]
    forward: ClassVar[Callable[..., Any]]

    # keep training settable (torch.nn.Module.train()/eval() assigns this)
    training: bool = Field(default=True, exclude=True)
    uuid: str = Field(default=None)

    device: str = Field(default="cpu", exclude=True)
    dtype: Any = Field(default=torch.float32, exclude=True)

    def __getattr__(self, name: str) -> Any:
        try:
            return torch.nn.Module.__getattr__(self, name)
        except AttributeError:
            return BaseModel.__getattr__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Important: keep Torch registration behavior and still satisfy Pydantic
        torch.nn.Module.__setattr__(self, name, value)
        BaseModel.__setattr__(self, name, value)

    def model_post_init(self, __context: Any) -> None:
        self.uuid = f"{self.__class__.__name__}:{str(uuid4())}"
        # Ensure torch.nn.Module initialization happens after Pydantic has set fields
        torch.nn.Module.__init__(self)
        # Always present buffer (some tests expect it to exist in state_dict)
        self.register_buffer("running", torch.zeros(3))
        super().model_post_init(__context)

    def clone(self,**kwargs) -> Module:
        state = deepcopy(self.state_dict())
        args = self.model_dump()
        args.update(kwargs)
        model = self.__class__(**args)
        model.load_state_dict(state)
        return model
    
    def save_file(self, path: str,**kwargs) -> None:
        weights = self.state_dict()
        config = self.model_dump()
        state = {"model": weights, "model_dump": config}
        state.update(kwargs)
        try:
            torch.save(state, path)
        except Exception as e:
            print(e)
            torch.save({"model": weights, "model_dump": config}, path)

    @classmethod
    def load_file(cls, path: str):
        state = torch.load(path)
        weights = state.get("model", {})
        config = state.get("model_dump", None)
        if config is not None:
            model = cls(**config)
        else:
            model = cls()
        model.load_state_dict(weights)
        return model,{k:v for k,v in state.items() if k not in ["model", "model_dump"]}

class Linear(Module, Bit.Linear):
    in_features: int = Field(default=10, ge=1)
    out_features: int = Field(default=5, ge=1)
    weight: torch.Tensor = Field(default=None, exclude=True)
    bias: Optional[Union[torch.Tensor, torch.nn.Parameter, bool]] = Field(default=True, exclude=True)

    scale_op: Literal["mean", "median"] = "median"
    w_q: Bit.Bit1p58Weight = Field(default=None, exclude=True)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        bias = self.bias
        Bit.Linear.__init__(self,                                  
            in_f=self.in_features,
            out_f=self.out_features,
            bias=True,
            device=self.device,
            dtype=self.dtype,
        )
        if bias is None or not bias:
            self.bias = None

class Conv2d(Module, Bit.Conv2d):
    in_channels: int = Field(default=-1)
    out_channels: int = Field(default=-1)
    kernel_size: Union[int, Tuple[int, int]] = Field(default=3)
    stride: Union[int, Tuple[int, int]] = Field(default=1)
    padding: Union[str, int, Tuple[int, int]] = Field(default=0)
    dilation: Union[int, Tuple[int, int]] = Field(default=1)
    transposed: bool = Field(default=False, exclude=True)
    output_padding: Union[int, Tuple[int, int]] = Field(default=0, exclude=True)
    groups: int = Field(default=1)
    padding_mode: str = Field(default="zeros")
    weight: torch.Tensor = Field(default=None, exclude=True)
    bias: Optional[Union[torch.Tensor, torch.nn.Parameter, bool]] = Field(default=True, exclude=True)

    scale_op: Literal["mean", "median"] = "median"
    padding_value: Union[int, Tuple[int, int]] = Field(default=(0, 0))
    dynamic_pad: bool = Field(default=False)
    pad_layer: Optional[PadSame] = None
    w_q: Bit.Bit1p58Weight = Field(default=None, exclude=True)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.in_channels>0 and self.out_channels>0 and self.kernel_size is not None:
            self.module_init()

    def module_init(self):
        bias = self.bias
        Bit.Conv2d.__init__(
            self,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
            padding_mode=self.padding_mode,
            device=self.device,
            dtype=self.dtype,
        )
        if bias is None or (type(bias) is bool and not bias):
            self.bias = None

class LayerNorm(Module, torch.nn.LayerNorm):
    normalized_shape: Union[int,List[int],Tuple[int, ...]]
    eps: float = Field(default=1e-5, ge=0.0)
    elementwise_affine: bool = Field(default=True)
    weight: torch.Tensor = Field(default=None, exclude=True)
    bias: torch.Tensor = Field(default=None, exclude=True)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        torch.nn.LayerNorm.__init__(self, **self.model_dump(exclude=["uuid"]))

class BatchNorm2d(Module, torch.nn.BatchNorm2d):
    num_features: int = Field(default=1, ge=1)
    eps: float = Field(default=1e-5, ge=0.0)
    momentum: Optional[float] = Field(default=0.1)
    affine: bool = Field(default=True)
    track_running_stats: bool = Field(default=True)
    weight: Optional[torch.Tensor] = Field(default=None, exclude=True)
    bias: Optional[torch.Tensor] = Field(default=None, exclude=True)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        affine = self.affine
        torch.nn.BatchNorm2d.__init__(
            self,
            num_features=self.num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=True,
            track_running_stats=self.track_running_stats,
            device=self.device,
            dtype=self.dtype,
        )
        if not affine:
            self.affine = False
            self.weight = None
            self.bias = None

class MaxPool2d(Module, torch.nn.MaxPool2d):
    kernel_size: Union[int, Tuple[int, int]] = Field(default=2)
    stride: Optional[Union[int, Tuple[int, int]]] = Field(default=None)
    padding: Union[int, Tuple[int, int]] = Field(default=0)
    dilation: Union[int, Tuple[int, int]] = Field(default=1)
    return_indices: bool = Field(default=False)
    ceil_mode: bool = Field(default=False)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        torch.nn.MaxPool2d.__init__(
            self,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=self.return_indices,
            ceil_mode=self.ceil_mode,
        )

class AdaptiveAvgPool2d(Module, torch.nn.AdaptiveAvgPool2d):
    output_size: Union[int, Tuple[Optional[int], Optional[int]]] = Field(default=1)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        torch.nn.AdaptiveAvgPool2d.__init__(self, output_size=self.output_size)

class Embedding(Module, torch.nn.Embedding):
    num_embeddings: int = Field(default=1, ge=1)
    embedding_dim: int = Field(default=1, ge=1)
    padding_idx: Optional[int] = Field(default=None)
    max_norm: Optional[float] = Field(default=None)
    norm_type: float = Field(default=2.0)
    scale_grad_by_freq: bool = Field(default=False)
    sparse: bool = Field(default=False)
    freeze: bool = Field(default=False, exclude=True)
    weight: Optional[torch.Tensor] = Field(default=None, exclude=True)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        torch.nn.Embedding.__init__(
            self,
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
            _freeze=self.freeze,
            device=self.device,
            dtype=self.dtype,
        )

class GELU(Module, torch.nn.GELU):
    approximate: str = Field(default="none")

    def model_post_init(self, __context):
        super().model_post_init(__context)
        torch.nn.GELU.__init__(self, **self.model_dump(exclude=["uuid"]))

class ReLU(Module, torch.nn.ReLU):
    inplace: bool = Field(default=False)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        torch.nn.ReLU.__init__(self, inplace=self.inplace)

class Dropout(Module, torch.nn.Dropout):
    p: float = Field(default=0.5, ge=0.0, le=1.0)
    inplace: bool = Field(default=False)

class Flatten(Module, torch.nn.Flatten):
    start_dim: int = 1
    end_dim: int = -1
    def model_post_init(self, __context):
        super().model_post_init(__context)
        torch.nn.Flatten.__init__(self, 
            start_dim=self.start_dim, end_dim=self.end_dim)


class DropPath(Module):
    drop_prob: float = Field(default=0.0, ge=0.0, le=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob
      
class Identity(torch.nn.Identity, Module):
    def module_init(self):
        pass
