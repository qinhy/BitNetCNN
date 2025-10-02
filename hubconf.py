"""
PyTorch Hub configuration for BitNetCNN models.

Usage:
    import torch

    # Load BitNetCNN for MNIST
    model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_mnist', pretrained=True)

    # Load BitResNet18 for CIFAR-100
    model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18', pretrained=True)

    # Load BitResNet50 for CIFAR-100
    model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet50', pretrained=True)

    # Load BitMobileNetV2 for CIFAR-100
    model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_mobilenetv2', pretrained=True)

    # Load BitConvNeXtv2 for CIFAR-100
    model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_convnextv2', pretrained=True)
"""

dependencies = ['torch', 'torchvision']

import torch
import torch.nn as nn
import zipfile
import os
import tempfile
from BitNetCNN import NetCNN
from common_utils import Bit, convert_to_ternary


def _load_checkpoint_from_file(path):
    """
    Load checkpoint from file, supporting both .pt and .zip formats.

    Args:
        path (str): Path to checkpoint file (.pt or .zip)

    Returns:
        dict: Loaded checkpoint
    """
    if path.endswith('.zip'):
        # Extract .pt file from zip
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path, 'r') as zip_ref:
                # Find .pt file in zip
                pt_files = [f for f in zip_ref.namelist() if f.endswith('.pt')]
                if not pt_files:
                    raise ValueError(f"No .pt file found in {path}")

                # Extract first .pt file
                pt_file = pt_files[0]
                zip_ref.extract(pt_file, tmpdir)
                extracted_path = os.path.join(tmpdir, pt_file)

                # Load checkpoint
                checkpoint = torch.load(extracted_path, map_location='cpu', weights_only=False)
                return checkpoint
    else:
        # Load .pt file directly
        return torch.load(path, map_location='cpu', weights_only=False)


def _load_checkpoint_from_url(url):
    """
    Load checkpoint from URL, supporting both .pt and .zip formats.

    Args:
        url (str): URL to checkpoint file

    Returns:
        dict or OrderedDict: Loaded checkpoint or state_dict
    """
    if url.endswith('.zip'):
        # Download and extract
        import urllib.request
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, 'checkpoint.zip')

            # Download zip
            urllib.request.urlretrieve(url, zip_path)

            # Extract and load
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                pt_files = [f for f in zip_ref.namelist() if f.endswith('.pt')]
                if not pt_files:
                    raise ValueError(f"No .pt file found in zip from {url}")

                pt_file = pt_files[0]
                zip_ref.extract(pt_file, tmpdir)
                extracted_path = os.path.join(tmpdir, pt_file)

                return torch.load(extracted_path, map_location='cpu', weights_only=False)
    else:
        # Use torch hub's built-in download
        return torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=False)


def bitnet_mnist(pretrained=False, scale_op="median", ternary=False):
    """
    BitNetCNN model for MNIST (1 channel, 10 classes).

    Args:
        pretrained (bool): If True, loads pre-trained weights
        scale_op (str): Scale operation for quantization ('mean' or 'median')
        ternary (bool): If True, returns ternary inference model (int8 weights)

    Returns:
        nn.Module: BitNetCNN model
    """
    model = NetCNN(in_channels=1, num_classes=10, expand_ratio=5, scale_op=scale_op)

    if pretrained:
        # Try to load from GitHub releases or a hosted URL
        try:
            checkpoint_url = 'https://github.com/qinhy/BitNetCNN/raw/refs/heads/main/models/bit_netcnn_small_mnist_best_fp.pt'
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu', check_hash=False)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict)
            print(f"Loaded pretrained BitNetCNN MNIST model")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Returning model with random initialization")

    if ternary:
        model = convert_to_ternary(model)
        print("Converted to ternary inference model")

    return model


def bitnet_resnet18(pretrained=False, scale_op="median", ternary=False, checkpoint_path=None):
    """
    BitResNet-18 model for CIFAR-100.

    Args:
        pretrained (bool): If True, loads pre-trained weights from GitHub releases
        scale_op (str): Scale operation for quantization ('mean' or 'median')
        ternary (bool): If True, returns ternary inference model (int8 weights)
        checkpoint_path (str): Path to local checkpoint file (overrides pretrained)

    Returns:
        nn.Module: BitResNet18 model
    """
    from BitResNet18 import BitResNet18CIFAR, BottleneckBit

    model = BitResNet18CIFAR(BottleneckBit, [2,2,2,2], num_classes=100, scale_op=scale_op)

    # Load from local checkpoint if provided
    if checkpoint_path is not None:
        try:
            checkpoint = _load_checkpoint_from_file(checkpoint_path)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
            acc = checkpoint.get('acc_tern', checkpoint.get('acc', 'unknown'))
            print(f"Loaded checkpoint from {checkpoint_path} (acc: {acc})")
        except Exception as e:
            print(f"Warning: Could not load checkpoint from {checkpoint_path}: {e}")
    # Load from GitHub releases
    elif pretrained:
        try:
            # Try ternary checkpoint first (prefer .zip for smaller size)
            checkpoint_url = 'https://github.com/qinhy/BitNetCNN/raw/refs/heads/main/models/bit_resnet_18_c100_ternary.pt.zip'
            checkpoint = _load_checkpoint_from_url(checkpoint_url)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained BitResNet18 CIFAR-100 model (ternary)")
        except Exception:
            # Fallback to .pt file
            try:
                checkpoint_url = 'https://github.com/qinhy/BitNetCNN/raw/refs/heads/main/models/bit_resnet_18_c100_ternary.pt'
                checkpoint = _load_checkpoint_from_url(checkpoint_url)
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained BitResNet18 CIFAR-100 model (ternary, .pt)")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")
                print("Returning model with random initialization")

    if ternary:
        model = convert_to_ternary(model)
        print("Converted to ternary inference model")

    return model


def bitnet_resnet50(pretrained=False, scale_op="median", ternary=False):
    """
    BitResNet-50 model for CIFAR-100.

    Args:
        pretrained (bool): If True, loads pre-trained weights
        scale_op (str): Scale operation for quantization ('mean' or 'median')
        ternary (bool): If True, returns ternary inference model (int8 weights)

    Returns:
        nn.Module: BitResNet50 model
    """
    from BitResNet50 import BitResNet50CIFAR

    model = BitResNet50CIFAR(num_classes=100, scale_op=scale_op)

    if pretrained:
        try:
            checkpoint_url = 'https://github.com/qinhy/BitNetCNN/raw/refs/heads/main/models/bit_resnet_50_c100_best_fp.pt'
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu', check_hash=False)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict)
            print(f"Loaded pretrained BitResNet50 CIFAR-100 model")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Returning model with random initialization")

    if ternary:
        model = convert_to_ternary(model)
        print("Converted to ternary inference model")

    return model


def bitnet_mobilenetv2(pretrained=False, scale_op="median", width_mult=1.0, ternary=False):
    """
    BitMobileNetV2 model for CIFAR-100.

    Args:
        pretrained (bool): If True, loads pre-trained weights
        scale_op (str): Scale operation for quantization ('mean' or 'median')
        width_mult (float): Width multiplier for model channels
        ternary (bool): If True, returns ternary inference model (int8 weights)

    Returns:
        nn.Module: BitMobileNetV2 model
    """
    from BitMobileNetV2 import BitMobileNetV2

    model = BitMobileNetV2(num_classes=100, width_mult=width_mult, scale_op=scale_op)

    if pretrained:
        try:
            checkpoint_url = f'https://github.com/qinhy/BitNetCNN/raw/refs/heads/main/models/bit_mobilenetv2_{width_mult}_c100_best_fp.pt'
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu', check_hash=False)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict)
            print(f"Loaded pretrained BitMobileNetV2 (width={width_mult}) CIFAR-100 model")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Returning model with random initialization")

    if ternary:
        model = convert_to_ternary(model)
        print("Converted to ternary inference model")

    return model


def bitnet_convnextv2(pretrained=False, scale_op="median", ternary=False):
    """
    BitConvNeXtv2 model for CIFAR-100.

    Args:
        pretrained (bool): If True, loads pre-trained weights
        scale_op (str): Scale operation for quantization ('mean' or 'median') - Note: not used in ConvNeXtV2
        ternary (bool): If True, returns ternary inference model (int8 weights)

    Returns:
        nn.Module: BitConvNeXtv2 model
    """
    from BitConvNeXtv2 import ConvNeXtV2

    # ConvNeXtV2 uses smaller dims for CIFAR-100
    model = ConvNeXtV2(in_chans=3, num_classes=100,
                       depths=[3, 3, 9, 3], dims=[48, 96, 192, 384])

    if pretrained:
        try:
            checkpoint_url = 'https://github.com/qinhy/BitNetCNN/raw/refs/heads/main/models/bit_convnextv2_c100_best_fp.pt'
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu', check_hash=False)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict)
            print(f"Loaded pretrained BitConvNeXtv2 CIFAR-100 model")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Returning model with random initialization")

    if ternary:
        model = convert_to_ternary(model)
        print("Converted to ternary inference model")

    return model
