from __future__ import annotations

import json
from typing import Optional, Type, Union

from pydantic import BaseModel, Field
import torch
from . import nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Custom activations used by the factory
# -------------------------------------------------------------------------

def hard_mish(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Hard-Mish: 0.5 * x * clamp(x + 2, 0, 2)."""
    if inplace:
        x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
        return x
    return 0.5 * x * (x + 2).clamp(min=0, max=2)

class HardMish(nn.Module):
    inplace:bool = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hard_mish(x, self.inplace)

def gelu_tanh(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """GELU with tanh approximation."""
    return F.gelu(x, approximate="tanh")


class GELUTanh(nn.Module):
    inplace:bool = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


def quick_gelu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Quick GELU approximation."""
    return x * torch.sigmoid(1.702 * x)


class QuickGELU(nn.Module):
    inplace:bool = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quick_gelu(x)


def mish(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Mish activation: x * tanh(softplus(x))."""
    y = x * torch.tanh(F.softplus(x))
    if inplace:
        x.copy_(y)
        return x
    return y


# -------------------------------------------------------------------------
# Name aliases (normalize different spellings to the same key)
# -------------------------------------------------------------------------

_ACT_NAME_ALIASES = {
    "hardsigmoid": "hard_sigmoid",
    "hardswish": "hard_swish",
    # add more aliases here if needed
}


def _normalize_act_name(name: str) -> str:
    name = name.lower()
    return _ACT_NAME_ALIASES.get(name, name)


# -------------------------------------------------------------------------
# Function-level activations
# -------------------------------------------------------------------------

_ACT_FN_DEFAULT = dict(
    # standard
    relu=F.relu,
    relu6=F.relu6,
    leaky_relu=F.leaky_relu,
    elu=F.elu,
    celu=F.celu,
    selu=F.selu,
    # SiLU / Swish
    silu=F.silu,
    swish=F.silu,
    # Mish family
    mish=mish,
    hard_mish=hard_mish,
    # Hard-sigmoid / hard-swish
    hard_sigmoid=F.hardsigmoid,
    hard_swish=F.hardswish,
    # GELU family
    gelu=F.gelu,
    gelu_tanh=gelu_tanh,
    quick_gelu=quick_gelu,
    # basic
    sigmoid=torch.sigmoid,
    tanh=torch.tanh,
)

# kept only for potential external introspection
_ACT_FNS = (_ACT_FN_DEFAULT,)


# -------------------------------------------------------------------------
# Module / layer-level activations
# -------------------------------------------------------------------------

_ACT_LAYER_DEFAULT = dict(
    # standard
    relu=torch.nn.ReLU,
    relu6=torch.nn.ReLU6,
    leaky_relu=torch.nn.LeakyReLU,
    elu=torch.nn.ELU,
    prelu=torch.nn.PReLU,
    celu=torch.nn.CELU,
    selu=torch.nn.SELU,
    # SiLU / Swish
    silu=torch.nn.SiLU,
    swish=torch.nn.SiLU,
    # Mish family
    mish=torch.nn.Mish,
    hard_mish=HardMish,
    # Hard-sigmoid / hard-swish
    hard_sigmoid=torch.nn.Hardsigmoid,
    hard_swish=torch.nn.Hardswish,
    # GELU family
    gelu=torch.nn.GELU,
    gelu_tanh=GELUTanh,
    quick_gelu=QuickGELU,
    # basic
    sigmoid=torch.nn.Sigmoid,
    tanh=torch.nn.Tanh,
    identity=torch.nn.Identity,
)

_ACT_LAYERS = (_ACT_LAYER_DEFAULT,)


# -------------------------------------------------------------------------
# Shared lookup helper
# -------------------------------------------------------------------------

def _get_act(
    name: Optional[str],
    default_map: dict,
    me_map: Optional[dict] = None,  # kept for API compatibility, but unused
):
    """Shared activation lookup.

    - If name is None or empty, returns None.
    - If name is not a string (callable, nn.Module subclass, etc.), returns it directly.
    """
    if name is None:
        return None

    if not isinstance(name, str):
        # callable, module class, already-constructed layer, etc.
        return name

    if not name:
        return None

    act_name = _normalize_act_name(name)
    return default_map[act_name]


# -------------------------------------------------------------------------
# Public factory functions
# -------------------------------------------------------------------------

def get_act_fn(name: Optional[str] = "relu"):
    """Activation Function Factory.

    Returns a function (e.g. F.relu) or None.
    """
    return _get_act(name, _ACT_FN_DEFAULT)


def get_act_layer(name: Optional[str] = "relu"):
    """Activation Layer Factory.

    Returns an nn.Module class (e.g. nn.ReLU) or None.
    """
    return _get_act(name, _ACT_LAYER_DEFAULT)


def create_act_layer(
    name: Optional[str],
    inplace: Optional[bool] = None,
    **kwargs,
):
    """Instantiate an activation layer by name."""
    act_layer = get_act_layer(name)
    if act_layer is None:
        return None

    if inplace is None:
        return act_layer(**kwargs)

    try:
        return act_layer(inplace=inplace, **kwargs)
    except TypeError:
        # some layers don't accept `inplace`
        return act_layer(**kwargs)



class _InplaceActBase(BaseModel):
    inplace: bool = False


class Acts:
    class ReLU(nn.Module, _InplaceActBase, torch.nn.ReLU):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.ReLU.__init__(self, inplace=self.inplace)
    class ReLU6(nn.Module, _InplaceActBase, torch.nn.ReLU6):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.ReLU6.__init__(self, inplace=self.inplace)
    class LeakyReLU(nn.Module, _InplaceActBase, torch.nn.LeakyReLU):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.LeakyReLU.__init__(self, inplace=self.inplace)
    class ELU(nn.Module, _InplaceActBase, torch.nn.ELU):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.ELU.__init__(self, inplace=self.inplace)
    class CELU(nn.Module, _InplaceActBase, torch.nn.CELU):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.CELU.__init__(self, inplace=self.inplace)
    class SELU(nn.Module, _InplaceActBase, torch.nn.SELU):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.SELU.__init__(self, inplace=self.inplace)
    class GELU(nn.Module, _InplaceActBase, torch.nn.GELU):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.GELU.__init__(self, inplace=self.inplace)
    # class GELUTanh(nn.Module, _InplaceActBase, torch.nn.GELUTanh):
    #     def model_post_init(self, __context):
    #         super().model_post_init(__context)
    #         torch.nn.GELUTanh.__init__(self, inplace=self.inplace)
    # class QuickGELU(nn.Module, _InplaceActBase, torch.nn.QuickGELU):
    #     def model_post_init(self, __context):
    #         super().model_post_init(__context)
    #         torch.nn.QuickGELU.__init__(self, inplace=self.inplace)
    class PReLU(nn.Module, _InplaceActBase, torch.nn.PReLU):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.PReLU.__init__(self, inplace=self.inplace)
    class SiLU(nn.Module, _InplaceActBase, torch.nn.SiLU):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.SiLU.__init__(self, inplace=self.inplace)
    class Swish(nn.Module, _InplaceActBase, torch.nn.SiLU):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.SiLU.__init__(self, inplace=self.inplace)
    class Mish(nn.Module, _InplaceActBase, torch.nn.Mish):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.Mish.__init__(self, inplace=self.inplace)
    class HardMish(HardMish):
        pass
    # class HardSigmoid(nn.Module, _InplaceActBase, torch.nn.HardSigmoid):
    #     def model_post_init(self, __context):
    #         super().model_post_init(__context)
    #         torch.nn.HardSigmoid.__init__(self, inplace=self.inplace)
    # class HardSwish(nn.Module, _InplaceActBase, torch.nn.HardSwish):
    #     def model_post_init(self, __context):
    #         super().model_post_init(__context)
    #         torch.nn.HardSwish.__init__(self, inplace=self.inplace)
    class Sigmoid(nn.Module, _InplaceActBase, torch.nn.Sigmoid):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.Sigmoid.__init__(self)
    class Tanh(nn.Module, _InplaceActBase, torch.nn.Tanh):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.Tanh.__init__(self)
    class Identity(nn.Module, _InplaceActBase, torch.nn.Identity):
        def model_post_init(self, __context):
            super().model_post_init(__context)
            torch.nn.Identity.__init__(self)

    type = Union[ReLU,ReLU6,LeakyReLU,ELU,CELU,SELU,GELU,GELUTanh,QuickGELU,PReLU,SiLU,Swish,Mish,HardMish,
                #  HardSigmoid,HardSwish,
                 Sigmoid,Tanh,Identity,]
    cls = Union[Type[ReLU],Type[ReLU6],Type[LeakyReLU],Type[ELU],Type[CELU],Type[SELU],Type[GELU],Type[GELUTanh],Type[QuickGELU],
                Type[PReLU],Type[SiLU],Type[Swish],Type[Mish],Type[HardMish],
                # Type[HardSigmoid],Type[HardSwish],
                Type[Sigmoid],Type[Tanh],Type[Identity],]