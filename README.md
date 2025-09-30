# BitNetCNN

A PyTorch implementation of CNN architectures using BitNet principles for efficient neural networks with ternary weights quantization.

## Overview

BitNetCNN is inspired by Microsoft's BitNet research, focusing on extreme quantization for neural networks. This implementation provides multiple CNN architectures with 1.58-bit (ternary) weight quantization and configurable activation quantization, designed for efficient inference with minimal accuracy loss.

## Features

- **Ternary Weight Quantization**: Uses 1.58-bit weights (values limited to {-1, 0, 1}) with per-output-channel scaling
- **Power-of-Two Scaling**: Optional power-of-two scaling for even more efficient inference
- **Multiple Model Architectures**:
  - BitNetCNN: Basic CNN with inverted residual blocks
  - BitResNet18/50: ResNet architectures with ternary weights
  - BitMobileNetV2: MobileNetV2 architecture with ternary weights
  - BitConvNeXtv2: ConvNeXt v2 architecture with ternary weights
- **Training and Inference Modes**: Supports both training with fake quantization and inference with true quantization
- **Knowledge Distillation**: Support for training with knowledge distillation from full-precision teachers
- **PyTorch Lightning Integration**: Complete training pipelines for MNIST and CIFAR datasets

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BitNetCNN.git
cd BitNetCNN

# Install dependencies
pip install -e .
```

### Requirements

- Python ≥ 3.11
- PyTorch 2.4.1
- torchvision 0.19.1
- pytorch-lightning
- huggingface_hub (for BitResNet18/50)

## Available Models

### BitNetCNN
Basic CNN architecture with inverted residual blocks for MNIST classification.

### BitResNet18/50
ResNet architectures with ternary weights for CIFAR classification.

### BitMobileNetV2
MobileNetV2 architecture with ternary weights for CIFAR classification.

### BitConvNeXtv2
ConvNeXt v2 architecture with ternary weights.

## Usage

### Training BitNetCNN on MNIST

```python
python BitNetCNN.py --epochs 10 --batch-size 128 --scale-op median
```

### Training BitResNet18 on CIFAR-100

```python
python BitResNet18.py --epochs 100 --batch-size 128 --scale-op median --kd
```

### Training BitMobileNetV2 on CIFAR-100

```python
python BitMobileNetV2.py --epochs 100 --batch-size 128 --scale-op median --kd
```

### Key Parameters

- `--scale-op`: Scaling operation for weight quantization ("mean" or "median")
- `--eval-ternary`: Evaluate using frozen ternary model during training
- `--amp`: Enable mixed precision training on CUDA devices
- `--kd`: Enable knowledge distillation (for ResNet and MobileNetV2)
- `--lr`: Learning rate
- `--wd`: Weight decay

### Using BitNet Models in Your Code

```python
# Basic BitNetCNN
from BitNetCNN import NetCNN, Bit, convert_to_ternary

# Create a model for training
model = NetCNN(in_channels=1, num_classes=10, mod=Bit)

# Train the model
# ...

# Convert to ternary for efficient inference
ternary_model = convert_to_ternary(model)

# For power-of-two scaling (more efficient)
from BitNetCNN import convert_to_ternary_p2
ternary_p2_model = convert_to_ternary_p2(model)
```

## Model Architecture

### BitNetCNN
Uses an architecture inspired by MobileNetV2 with inverted residual blocks:

1. **Stem**: Initial convolution to process input images
2. **Stages**: Three inverted residual blocks with increasing channels (32→64→128→256)
3. **Head**: Global average pooling followed by a classifier

### BitResNet18/50
Standard ResNet architecture with ternary weights:

1. **Stem**: Initial convolution layer
2. **Layers**: Four layer groups with BasicBlock (ResNet18) or Bottleneck (ResNet50)
3. **Head**: Global average pooling followed by a classifier

### BitMobileNetV2
MobileNetV2 architecture with ternary weights:

1. **Stem**: Initial convolution layer
2. **Inverted Residual Blocks**: Series of inverted residual blocks with expansion factor
3. **Head**: Global average pooling followed by a classifier

All convolution and linear layers use ternary weights for extreme parameter efficiency.

## Microsoft BitNet Inspiration

This implementation is inspired by Microsoft's BitNet research, which explores extreme quantization for neural networks. The BitNet papers demonstrate that models with 1-2 bit weights can achieve competitive performance while significantly reducing computational requirements.

Key concepts from Microsoft's BitNet research implemented here:
- 1.58-bit ternary weight quantization ({-1, 0, 1})
- Per-output-channel scaling factors
- Optional power-of-two scaling for more efficient inference
- Support for various model architectures (CNN, ResNet, MobileNetV2, ConvNeXtv2)

## Performance

The ternary models achieve competitive accuracy compared to their full-precision counterparts while significantly reducing memory footprint and computational requirements:

- BitNetCNN: >99% accuracy on MNIST
- BitResNet18: Competitive accuracy on CIFAR-100 with knowledge distillation
- BitMobileNetV2: Efficient alternative to full-precision MobileNetV2

## License

See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
