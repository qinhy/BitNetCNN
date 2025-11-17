"""Conv + Norm + Act wrapper for BitNet layers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

from pydantic import BaseModel
from torch import nn

from timmlayers.blur_pool import create_aa

ConvSpec = Union[nn.Module, BaseModel]
NormSpec = Union[nn.Module, BaseModel, Type[nn.Module], Callable[..., nn.Module], None]
ActSpec = Union[nn.Module, BaseModel, Type[nn.Module], Callable[..., nn.Module], None]


def _to_stride_tuple(stride: Union[int, Sequence[int]]) -> Tuple[int, ...]:
    if isinstance(stride, Sequence) and not isinstance(stride, (str, bytes)):
        return tuple(int(s) for s in stride)
    return (int(stride), int(stride))


def _stride_one_like(stride: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    if isinstance(stride, tuple):
        return tuple(1 for _ in stride)
    return 1


class ConvNormAct(nn.Module):
    """Composable Conv -> Norm -> Act block with optional AA and Drop layers."""

    def __init__(
        self,
        conv_layer: ConvSpec,
        norm_layer: NormSpec = None,
        act_layer: ActSpec = None,
        *,
        apply_norm: bool = True,
        apply_act: bool = True,
        aa_layer: Optional[Any] = None,
        drop_layer: Optional[Union[Type[nn.Module], nn.Module]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}

        conv_layer = self._maybe_normalize_conv(conv_layer)
        conv_module, aa_stride, aa_enabled = self._build_conv(conv_layer, aa_layer)
        self.conv = conv_module

        out_channels = getattr(self.conv, "out_channels", None)
        self.bn = self._build_norm(norm_layer, apply_norm, out_channels, norm_kwargs)
        self.drop = self._build_drop(drop_layer)
        self.act = self._build_act(act_layer, apply_act, act_kwargs)

        if aa_enabled and out_channels is not None:
            self.aa = create_aa(aa_layer, out_channels, stride=aa_stride, enable=True, noop=None)
        else:
            self.aa = None

    @staticmethod
    def _maybe_normalize_conv(conv_layer: ConvSpec) -> ConvSpec:
        if isinstance(conv_layer, dict):
            raise TypeError("dict specs are not supported; pass a Conv2dModels.* instance instead.")
        return conv_layer

    @staticmethod
    def _build_drop(drop_layer: Optional[Union[Type[nn.Module], nn.Module]]) -> Optional[nn.Module]:
        if drop_layer is None:
            return None
        if isinstance(drop_layer, nn.Module):
            return drop_layer
        return drop_layer()

    def _build_conv(self, conv_layer: ConvSpec, aa_layer: Optional[Any]):
        if isinstance(conv_layer, nn.Module):
            if aa_layer is not None:
                stride = getattr(conv_layer, "stride", (1, 1))
                stride = _to_stride_tuple(stride)
                if any(s > 1 for s in stride):
                    raise ValueError("Anti-aliasing requires a Conv2dModels spec to adjust strides safely.")
            return conv_layer, (1, 1), False

        if not isinstance(conv_layer, BaseModel):
            raise TypeError("conv_layer must be an nn.Module or Conv2dModels.* specification.")

        stride_val = getattr(conv_layer, "stride", 1)
        stride_tuple = _to_stride_tuple(stride_val)
        use_aa = aa_layer is not None and any(s > 1 for s in stride_tuple)
        conv_spec = conv_layer
        if use_aa:
            conv_spec = conv_spec.model_copy(update={"stride": _stride_one_like(stride_val)})
        conv_module = conv_spec.build()
        return conv_module, stride_tuple, use_aa

    def _build_norm(
        self,
        norm_layer: NormSpec,
        apply_norm: bool,
        num_features: Optional[int],
        norm_kwargs: Dict[str, Any],
    ) -> nn.Module:
        if not apply_norm:
            return nn.Identity()

        if isinstance(norm_layer, nn.Module):
            return norm_layer

        if isinstance(norm_layer, BaseModel):
            return norm_layer.build()

        if norm_layer is None:
            if num_features is None:
                return nn.Identity()
            return nn.BatchNorm2d(num_features, **norm_kwargs)

        if isinstance(norm_layer, type) and issubclass(norm_layer, nn.Module):
            if num_features is None:
                return norm_layer(**norm_kwargs)
            return norm_layer(num_features, **norm_kwargs)

        if callable(norm_layer):
            if num_features is None:
                return norm_layer(**norm_kwargs)
            return norm_layer(num_features, **norm_kwargs)

        raise TypeError("Unsupported norm_layer specification.")

    def _build_act(
        self,
        act_layer: ActSpec,
        apply_act: bool,
        act_kwargs: Dict[str, Any],
    ) -> nn.Module:
        if not apply_act:
            return nn.Identity()

        if isinstance(act_layer, nn.Module):
            return act_layer

        if isinstance(act_layer, BaseModel):
            return act_layer.build()

        if act_layer is None:
            return nn.ReLU(**act_kwargs)

        if isinstance(act_layer, type) and issubclass(act_layer, nn.Module):
            return act_layer(**act_kwargs)

        if callable(act_layer):
            return act_layer(**act_kwargs)

        raise TypeError("Unsupported act_layer specification.")

    @property
    def in_channels(self):
        return getattr(self.conv, "in_channels", None)

    @property
    def out_channels(self):
        return getattr(self.conv, "out_channels", None)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.act(x)
        if self.aa is not None:
            x = self.aa(x)
        return x


ConvBnAct = ConvNormAct
ConvNormActAa = ConvNormAct
