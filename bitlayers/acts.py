from typing import Optional

import torch
from torch import nn
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
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hard_mish(x, self.inplace)


def gelu_tanh(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """GELU with tanh approximation."""
    return F.gelu(x, approximate="tanh")


class GELUTanh(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


def quick_gelu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Quick GELU approximation."""
    return x * torch.sigmoid(1.702 * x)


class QuickGELU(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()

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
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    elu=nn.ELU,
    prelu=nn.PReLU,
    celu=nn.CELU,
    selu=nn.SELU,
    # SiLU / Swish
    silu=nn.SiLU,
    swish=nn.SiLU,
    # Mish family
    mish=nn.Mish,
    hard_mish=HardMish,
    # Hard-sigmoid / hard-swish
    hard_sigmoid=nn.Hardsigmoid,
    hard_swish=nn.Hardswish,
    # GELU family
    gelu=nn.GELU,
    gelu_tanh=GELUTanh,
    quick_gelu=QuickGELU,
    # basic
    sigmoid=nn.Sigmoid,
    tanh=nn.Tanh,
    identity=nn.Identity,
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
