# PyTorch Hub Usage Guide

This repository supports loading models via PyTorch Hub for easy access to pre-trained BitNet models.

## Quick Start

```python
import torch

# Load BitNetCNN for MNIST
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_mnist', pretrained=True)

# Load BitResNet18 for CIFAR-100
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18', pretrained=True)
```

## Available Models

### 1. BitNetCNN (MNIST)

```python
# Load model without pretrained weights
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_mnist')

# Load with pretrained weights
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_mnist', pretrained=True)

# Load ternary inference model (int8 weights)
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_mnist', pretrained=True, ternary=True)
```

**Input:** 1x28x28 (MNIST grayscale images)
**Output:** 10 classes
**Parameters:** ~140K

### 2. BitResNet18 (CIFAR-100)

```python
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18', pretrained=True)

# Ternary version
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18', pretrained=True, ternary=True)
```

**Input:** 3x32x32 (CIFAR-100 RGB images)
**Output:** 100 classes
**Parameters:** ~11M

### 3. BitResNet50 (CIFAR-100)

```python
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet50', pretrained=True)
```

**Input:** 3x32x32
**Output:** 100 classes
**Parameters:** ~23M

### 4. BitMobileNetV2 (CIFAR-100)

```python
# Default width multiplier (1.0)
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_mobilenetv2', pretrained=True)

# Custom width multiplier
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_mobilenetv2',
                       pretrained=True, width_mult=0.5)
```

**Input:** 3x32x32
**Output:** 100 classes
**Parameters:** Varies with width_mult

### 5. BitConvNeXtv2 (CIFAR-100)

```python
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_convnextv2', pretrained=True)
```

**Input:** 3x32x32
**Output:** 100 classes

## Complete Example

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load model
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_mnist', pretrained=True)
model.eval()

# Prepare input
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

img = Image.open('digit.png')
img_tensor = transform(img).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(img_tensor)
    predicted = output.argmax(dim=1)
    print(f"Predicted digit: {predicted.item()}")
```

## Ternary Inference Models

Ternary models use int8 weights (-1, 0, +1) for efficient inference:

```python
# Load ternary model
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18',
                       pretrained=True, ternary=True)

# Model now uses:
# - Int8 weights in {-1, 0, +1}
# - Per-channel scaling factors
# - Much smaller memory footprint
# - Faster inference on compatible hardware
```

## Local Development

For local testing before pushing to GitHub:

```python
import torch

# Load from local directory
model = torch.hub.load('.', 'bitnet_mnist', source='local', pretrained=False)
```

## Parameters

All models support these parameters:

- **`pretrained`** (bool): Load pre-trained weights (default: False)
- **`scale_op`** (str): Quantization scale operation - 'mean' or 'median' (default: 'median')
- **`ternary`** (bool): Return ternary inference model with int8 weights (default: False)

## Uploading Pre-trained Weights

To make `pretrained=True` work, upload your trained model checkpoints to GitHub Releases:

1. Train your model and save checkpoints
2. Go to your GitHub repo → Releases → Create a new release
3. Tag it as `v1.0` (or appropriate version)
4. Upload checkpoint files with these names:
   - `bit_netcnn_small_mnist_best_fp.pt`
   - `bit_resnet_18_c100_best_fp.pt`
   - `bit_resnet_50_c100_best_fp.pt`
   - `bit_mobilenetv2_1.0_c100_best_fp.pt`
   - `bit_convnextv2_c100_best_fp.pt`

The URLs in `hubconf.py` will automatically point to these files.

## List Available Models

```python
# List all available models
models = torch.hub.list('qinhy/BitNetCNN')
print(models)
```

## Force Reload

Force reloading the hub configuration:

```python
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_mnist',
                       force_reload=True, pretrained=True)
```

## Notes

- Models are loaded in evaluation mode by default
- Use `.train()` if you want to fine-tune
- Ternary models are inference-only (no training)
- MNIST models expect 1-channel input, CIFAR models expect 3-channel input
