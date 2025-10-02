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
from BitNetCNN import NetCNN
from common_utils import Bit, convert_to_ternary


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
            checkpoint_url = 'https://github.com/qinhy/BitNetCNN/releases/download/v1.0/bit_netcnn_small_mnist_best_fp.pt'
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


def bitnet_resnet18(pretrained=False, scale_op="median", ternary=False):
    """
    BitResNet-18 model for CIFAR-100.

    Args:
        pretrained (bool): If True, loads pre-trained weights
        scale_op (str): Scale operation for quantization ('mean' or 'median')
        ternary (bool): If True, returns ternary inference model (int8 weights)

    Returns:
        nn.Module: BitResNet18 model
    """
    from BitResNet18 import BitResNet18CIFAR, BottleneckBit

    model = BitResNet18CIFAR(BottleneckBit, [2,2,2,2], num_classes=100, scale_op=scale_op)

    if pretrained:
        try:
            checkpoint_url = 'https://github.com/qinhy/BitNetCNN/releases/download/v1.0/bit_resnet_18_c100_best_fp.pt'
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu', check_hash=False)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict)
            print(f"Loaded pretrained BitResNet18 CIFAR-100 model")
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
    from BitResNet50 import BitResNet50CIFAR, BottleneckBit

    model = BitResNet50CIFAR(BottleneckBit, [3,4,6,3], num_classes=100, scale_op=scale_op)

    if pretrained:
        try:
            checkpoint_url = 'https://github.com/qinhy/BitNetCNN/releases/download/v1.0/bit_resnet_50_c100_best_fp.pt'
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
    from BitMobileNetV2 import BitMobileNetV2CIFAR

    model = BitMobileNetV2CIFAR(num_classes=100, width_mult=width_mult, scale_op=scale_op)

    if pretrained:
        try:
            checkpoint_url = f'https://github.com/qinhy/BitNetCNN/releases/download/v1.0/bit_mobilenetv2_{width_mult}_c100_best_fp.pt'
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
        scale_op (str): Scale operation for quantization ('mean' or 'median')
        ternary (bool): If True, returns ternary inference model (int8 weights)

    Returns:
        nn.Module: BitConvNeXtv2 model
    """
    from BitConvNeXtv2 import BitConvNeXtV2CIFAR

    model = BitConvNeXtV2CIFAR(num_classes=100, scale_op=scale_op)

    if pretrained:
        try:
            checkpoint_url = 'https://github.com/qinhy/BitNetCNN/releases/download/v1.0/bit_convnextv2_c100_best_fp.pt'
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
