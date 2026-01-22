# -*- coding: utf-8 -*-
"""
BitNet model factory & checkpoint loader (refactored)

Public API (unchanged):
    - bitnet_mnist(...)
    - bitnet_resnet(...), bitnet_resnet18(...), bitnet_resnet50(...)
    - bitnet_mobilenetv2(...)
    - bitnet_convnextv2(...)

Highlights:
    • Backward-compatible torch.load with/without weights_only
    • Robust checkpoint selection (prefers model_ema > model > raw state_dict)
    • Proper dataset fullname and class-count helpers
    • Correct ConvNeXtV2 filename construction & dataset handling
    • Optional local checkpoint override, graceful URL fallbacks (.zip → .pt)
    • Gentle loads (strict=False) and optional ternary conversion reload
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import os
import zipfile
import tempfile
import urllib.request

import torch
import torch.nn as nn

from common_utils import convert_to_ternary

# ---------------------------------------------------------------------
# Constants & dataset registry
# ---------------------------------------------------------------------

BASE_URL = "https://github.com/qinhy/BitNetCNN/raw/refs/heads/main/models/"

# Filename templates for published checkpoints
RESNET_FILE_TMPL     = "bit_resnet_{model_size}_{dataset}_{ternary}.{ext}"
MBNV2_FILE_TMPL      = "bit_mobilenetv2_x{model_size}_{dataset}_{ternary}.{ext}"
CONVNEXT_FILE_TMPL   = "bit_convnextv2_{model_size}_{dataset}_{ternary}.{ext}"
RETINAFACE_FILE_TMPL = "bit_retface_rn{model_size}_{dataset}_{ternary}.{ext}"

# Canonical dataset specs
DATASETS: Dict[str, Dict[str, Any]] = {
    "c100": {
        "aliases": {"c100", "cifar100", "cifar-100"},
        "num_classes": 100,
        "url_tag": "c100",
        "full": "CIFAR-100",
    },
    "c10": {
        "aliases": {"c10", "cifar10", "cifar-10"},
        "num_classes": 10,
        "url_tag": "c10",
        "full": "CIFAR-10",
    },
    "timnet": {
        "aliases": {"timnet", "tiny-imagenet"},
        "num_classes": 200,
        "url_tag": "timnet",
        "full": "Tiny-ImageNet",
    },
    "widerface": {
        "aliases": {"widerface", "wider-face"},
        "num_classes": 2,
        "url_tag": "widerface",
        "full": "WiderFace",
    },
    # Add more datasets here when supported:
    # "imagenet": {"aliases": {"imagenet", "in1k", "ilsvrc2012"}, "num_classes": 1000, "url_tag": "in1k", "full": "ImageNet-1k"},
}

# ---------------------------------------------------------------------
# Small utility helpers
# ---------------------------------------------------------------------

def _canon_dataset(name: str) -> str:
    """Return canonical dataset key (e.g., 'c100'). Defaults to 'c100' with a note."""
    n = (name or "").lower().strip()
    for key, spec in DATASETS.items():
        if n == key or n in spec["aliases"]:
            return key
    print(f"[info] Unrecognized dataset='{name}', defaulting to CIFAR-100.")
    return "c100"


def _fullname_for(ds_key: str) -> str:
    """Human-readable dataset name (e.g., 'CIFAR-100')."""
    return str(DATASETS[ds_key]["full"])


def _num_classes_for(ds_key: str) -> int:
    """Number of classes for dataset (e.g., 100 for CIFAR-100)."""
    return int(DATASETS[ds_key]["num_classes"])


def _urljoin(filename: str) -> str:
    """Prefix BASE_URL unless filename is already an absolute http(s) URL."""
    if filename.startswith(("http://", "https://")):
        return filename
    return BASE_URL.rstrip("/") + "/" + filename.lstrip("/")


def _ensure_list(x: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(x, str):
        return [x]
    return list(x)


def _safe_torch_load(path: str, map_location: str = "cpu"):
    """
    torch.load with graceful fallback for older PyTorch that doesn't support weights_only.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)  # new kw
    except TypeError:
        return torch.load(path, map_location=map_location)  # older PyTorch


def _pick_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Accepts either:
      • raw state_dict (OrderedDict)
      • dict containing model weights under common keys (model_ema > model > state_dict)
    Returns a state_dict-like mapping.
    """
    if isinstance(ckpt, dict):
        # Prefer EMA if present
        for key in ("model_ema", "ema", "ema_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # Then standard keys
        for key in ("model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    # Might already be a raw state_dict (e.g., OrderedDict)
    if hasattr(ckpt, "keys") and hasattr(ckpt, "items"):
        return ckpt  # type: ignore
    raise ValueError("Unrecognized checkpoint format; no state_dict found.")


def _extract_accuracy(ckpt: Any) -> Optional[Union[float, str]]:
    """
    Opportunistically extract accuracy from common fields.
    """
    if isinstance(ckpt, dict):
        for k in ("acc_tern", "acc", "top1", "val_acc"):
            if k in ckpt:
                return ckpt[k]
    return None


# ---------------------------------------------------------------------
# File / URL checkpoint loaders
# ---------------------------------------------------------------------

def _load_checkpoint_from_file(path: str) -> Any:
    """
    Load checkpoint (dict or state_dict) from a local .pt or .zip path.
    """
    if path.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path, "r") as zf:
                pt_files = [f for f in zf.namelist() if f.endswith(".pt")]
                if not pt_files:
                    raise ValueError(f"No .pt file found in zip: {path}")
                pt_file = pt_files[0]
                zf.extract(pt_file, tmpdir)
                extracted = os.path.join(tmpdir, pt_file)
                return _safe_torch_load(extracted, map_location="cpu")
    else:
        return _safe_torch_load(path, map_location="cpu")


def _load_checkpoint_from_url(url: str) -> Any:
    """
    Load checkpoint (dict or state_dict) from a remote .pt or .zip URL.
    """
    if url.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "checkpoint.zip")
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                pt_files = [f for f in zf.namelist() if f.endswith(".pt")]
                if not pt_files:
                    raise ValueError(f"No .pt file found in zip from URL: {url}")
                pt_file = pt_files[0]
                zf.extract(pt_file, tmpdir)
                extracted = os.path.join(tmpdir, pt_file)
                return _safe_torch_load(extracted, map_location="cpu")
    else:
        # torch.hub helper supports caching & hashing; keep check_hash=False for flexible releases
        return torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=False)


def _try_load_urls(model: nn.Module, urls: Iterable[str], model_name: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Iterate URL list, trying to load. On first success, load into model (strict=False) and return state_dict.
    """
    for i, url in enumerate(_ensure_list(urls)):
        try:
            raw = _load_checkpoint_from_url(url)
            state_dict = _pick_state_dict(raw)
            model.load_state_dict(state_dict, strict=False)
            suffix = f" (fallback {i})" if i > 0 else ""
            acc = _extract_accuracy(raw)
            note = f" (acc: {acc})" if acc is not None else ""
            print(f"Loaded pretrained {model_name}{suffix} from {url}{note}")
            return state_dict
        except Exception as e:
            print(f"[warn] Failed to load URL {url}: {e}")
    return None


def _load_pretrained_weights(
    model: nn.Module,
    checkpoint_urls: Union[str, Iterable[str]],
    checkpoint_path: Optional[str] = None,
    model_name: str = "model",
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Generic loader with local override and URL fallbacks. Returns the used state_dict (or None).
    """
    # 1) Local file overrides everything if provided
    if checkpoint_path:
        try:
            raw = _load_checkpoint_from_file(checkpoint_path)
            state_dict = _pick_state_dict(raw)
            model.load_state_dict(state_dict, strict=True)
            acc = _extract_accuracy(raw)
            note = f" (acc: {acc})" if acc is not None else ""
            print(f"Loaded checkpoint from {checkpoint_path}{note}")
            return state_dict
        except Exception as e:
            print(f"[warn] Could not load local checkpoint '{checkpoint_path}': {e}")

    # 2) URLs fallback in order
    state_dict = _try_load_urls(model, checkpoint_urls, model_name)
    if state_dict is None:
        print(f"[warn] Could not load pretrained weights for {model_name}. Using random init.")
    return state_dict


def _convert_to_ternary_if_needed(model: nn.Module, ternary: bool, state_dict: Optional[Dict[str, torch.Tensor]]) -> nn.Module:
    """
    Optionally convert to ternary inference model and non-strictly reload the same state_dict.
    """
    model = convert_to_ternary(model)
    if ternary and state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
        print("Converted to ternary inference model")
    return model


# ---------------------------------------------------------------------
# Public model factories
# ---------------------------------------------------------------------

def bitnet_mnist(
    pretrained: bool = False,
    scale_op: str = "median",
    ternary: bool = True,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """
    BitNetCNN model for MNIST (1 channel, 10 classes).
    """
    from BitNetCNN import NetCNN
    model = NetCNN(in_channels=1, num_classes=10, expand_ratio=5, scale_op=scale_op)

    state_dict: Optional[Dict[str, torch.Tensor]] = None
    _convert_to_ternary_if_needed(model, ternary, state_dict)
    if pretrained or checkpoint_path:
        checkpoint_url = "https://github.com/qinhy/BitNetCNN/raw/refs/heads/main/models/bit_netcnn_small_mnist_ternary.zip"
        state_dict = _load_pretrained_weights(model, checkpoint_url, checkpoint_path, model_name="BitNetCNN MNIST")
    if state_dict:model.load_state_dict(state_dict, strict=True)
    return model


def bitnet_resnet(
    pretrained: bool = False,
    model_size: Union[str, int] = "18",
    dataset: str = "c100",
    scale_op: str = "median",
    ternary: bool = True,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """
    Bit-ResNet builder for CIFAR-style inputs.

    model_size: "18" or "50"
    dataset: "c100"|"c10"|aliases
    """
    ds = _canon_dataset(dataset)
    model_size = str(model_size)

    from BitResNet import BitResNet,BasicBlockBit,BottleneckBit
    if model_size == "18":

        model = BitResNet(BasicBlockBit, [2, 2, 2, 2], _num_classes_for(ds), 1, scale_op, 3, True)
    elif model_size == "50":

        model = BitResNet(BottleneckBit, [3, 4, 6, 3], _num_classes_for(ds), 4, scale_op, 3, True)
    else:
        raise ValueError(f"Unsupported model_size='{model_size}'. Use '18' or '50'.")

    state_dict: Optional[Dict[str, torch.Tensor]] = None
    _convert_to_ternary_if_needed(model,ternary,state_dict)
    if checkpoint_path or pretrained:
        filenames = [
            RESNET_FILE_TMPL.format(
                model_size=model_size,
                dataset=ds,
                ternary=("ternary" if ternary else "best_fp"),
                ext=ext,
            )
            for ext in ("zip", "pt")
        ]
        checkpoint_urls = [_urljoin(f) for f in filenames]
        state_dict = _load_pretrained_weights(
            model,
            checkpoint_urls,
            checkpoint_path,
            model_name=f"BitResNet{model_size} {_fullname_for(ds)}{' (ternary)' if ternary else ''}",
        )

    if state_dict:model.load_state_dict(state_dict, strict=True)
    return model


def bitnet_resnet18(
    pretrained: bool = False,
    dataset: str = "c100",
    scale_op: str = "median",
    ternary: bool = True,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    return bitnet_resnet(
        pretrained=pretrained,
        model_size="18",
        dataset=dataset,
        scale_op=scale_op,
        ternary=ternary,
        checkpoint_path=checkpoint_path,
    )


def bitnet_resnet50(
    pretrained: bool = False,
    dataset: str = "c100",
    scale_op: str = "median",
    ternary: bool = True,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    return bitnet_resnet(
        pretrained=pretrained,
        model_size="50",
        dataset=dataset,
        scale_op=scale_op,
        ternary=ternary,
        checkpoint_path=checkpoint_path,
    )


def bitnet_retinaface(
    pretrained: bool = False,
    model_size="50",
    dataset: str = "widerface",
    scale_op: str = "median",
    ternary: bool = True,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    from BitRetinaFace import BitRetinaFace, RetinaFaceAnchors

    # Anchors
    anchor_gen = RetinaFaceAnchors(
        min_sizes=[[16, 32], [64, 128], [256, 512]],
        steps=[8, 16, 32], clip=False
    )
    model = BitRetinaFace(
        backbone_size=model_size, fpn_channels=256,
        scale_op=scale_op, small_stem=False
    )

    state_dict: Optional[Dict[str, torch.Tensor]] = None
    _convert_to_ternary_if_needed(model,ternary,state_dict)
    if checkpoint_path or pretrained:
        filenames = [
            RETINAFACE_FILE_TMPL.format(
                model_size=model_size,
                dataset=dataset,
                ternary=("ternary" if ternary else "best_fp"),
                ext=ext,
            )
            for ext in ("zip", "pt")
        ]
        checkpoint_urls = [_urljoin(f) for f in filenames]
        state_dict = _load_pretrained_weights(
            model,
            checkpoint_urls,
            checkpoint_path,
            model_name=f"BitRetinaFace {model_size} {_fullname_for(dataset)}{' (ternary)' if ternary else ''}",
        )

    if state_dict:model.load_state_dict(state_dict, strict=True)
    
    model.register_buffer("anchors", anchor_gen((640, 640)))
    return model

def _format_mbnv2_width_tag(width_mult: float) -> int:
    """
    Convert width multiplier (e.g., 1.0 → 100, 1.4 → 140) for filename tags.
    Rounds to nearest integer percentage.
    """
    return int(round(width_mult * 100))


def bitnet_mobilenetv2(
    pretrained: bool = False,
    width_mult: float = 1.0,
    dataset: str = "c100",
    scale_op: str = "median",
    ternary: bool = True,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """
    Bit-MobileNetV2 for CIFAR-style inputs.

    width_mult: e.g., 0.75, 1.0, 1.4 (published checkpoints typically: 75, 100, 140…)
    """
    from BitMobileNetV2 import BitMobileNetV2

    ds = _canon_dataset(dataset)
    model = BitMobileNetV2(num_classes=_num_classes_for(ds), width_mult=width_mult, scale_op=scale_op)

    state_dict: Optional[Dict[str, torch.Tensor]] = None
    _convert_to_ternary_if_needed(model, ternary, state_dict)
    if checkpoint_path or pretrained:
        size_tag = _format_mbnv2_width_tag(width_mult)
        filenames = [
            MBNV2_FILE_TMPL.format(
                model_size=size_tag,
                dataset=ds,
                ternary=("ternary" if ternary else "best_fp"),
                ext=ext,
            )
            for ext in ("zip", "pt")
        ]
        checkpoint_urls = [_urljoin(f) for f in filenames]
        state_dict = _load_pretrained_weights(
            model,
            checkpoint_urls,
            checkpoint_path,
            model_name=f"BitMobileNetV2 x{width_mult} {_fullname_for(ds)}{' (ternary)' if ternary else ''}",
        )

    if state_dict:model.load_state_dict(state_dict, strict=True)
    return model


def bitnet_convnextv2(
    pretrained: bool = False,
    model_size: str = "nano",
    dataset: str = "c100",
    scale_op: str = "median",
    ternary: bool = True,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """
    Bit-ConvNeXtV2 family for CIFAR-style inputs.

    model_size: e.g., "femto" | "pico" | "nano" | "tiny" | "base" ... (match your releases)
    """
    from BitConvNeXtv2 import ConvNeXtV2

    ds = _canon_dataset(dataset)
    model = ConvNeXtV2.convnextv2(size=model_size, num_classes=_num_classes_for(ds), scale_op=scale_op)

    state_dict: Optional[Dict[str, torch.Tensor]] = None
    if checkpoint_path or pretrained:
        filenames = [
            CONVNEXT_FILE_TMPL.format(
                model_size=model_size,
                dataset=ds,
                ternary=("ternary" if ternary else "best_fp"),
                ext=ext,
            )
            for ext in ("zip", "pt")
        ]
        checkpoint_urls = [_urljoin(f) for f in filenames]
        state_dict = _load_pretrained_weights(
            model,
            checkpoint_urls,
            checkpoint_path,
            model_name=f"BitConvNeXtv2 {model_size} {_fullname_for(ds)}{' (ternary)' if ternary else ''}",
        )

    if state_dict:model.load_state_dict(state_dict, strict=True)
    return model
