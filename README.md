# BitNetCNN

A PyTorch implementation of a CNN architecture using BitNet principles for efficient neural networks with ternary weights quantization.

## Overview

BitNetCNN is inspired by Microsoft's BitNet research, focusing on extreme quantization for neural networks. This implementation provides a CNN architecture with 1.58-bit (ternary) weight quantization and configurable activation quantization, designed for efficient inference with minimal accuracy loss.

## Features

- **Ternary Weight Quantization**: Uses 1.58-bit weights (values limited to {-1, 0, 1}) with per-output-channel scaling
- **Configurable Activation Quantization**: Supports 4-bit or 8-bit activation quantization
- **Efficient Architecture**: Based on inverted residual blocks similar to MobileNet design
- **Training and Inference Modes**: Supports both training with fake quantization and inference with true quantization
- **MNIST Example**: Includes complete training pipeline for MNIST classification

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

## Usage

### Training a BitNetCNN model on MNIST

```python
python BitNetCNN.py --epochs 10 --batch-size 128 --act-bits 8
```

### Key Parameters

- `--act-bits`: Activation quantization bits (4 or 8)
- `--scale-op`: Scaling operation for weight quantization ("mean" or "median")
- `--eval-ternary`: Evaluate using frozen ternary model during training
- `--amp`: Enable mixed precision training on CUDA devices

### Using the BitNetCNN Model in Your Code

```python
from BitNetCNN import BitNetCNN, convert_to_ternary

# Create a model for training
model = BitNetCNN(in_channels=1, num_classes=10, act_bits=8)

# Train the model
# ...

# Convert to ternary for efficient inference
ternary_model = convert_to_ternary(model)
```

## Model Architecture

BitNetCNN uses an architecture inspired by MobileNetV2 with inverted residual blocks:

1. **Stem**: Initial convolution to process input images
2. **Stages**: Three inverted residual blocks with increasing channels (32→64→128→256)
3. **Head**: Global average pooling followed by a classifier

All convolution and linear layers use ternary weights for extreme parameter efficiency.

## Microsoft BitNet Inspiration

This implementation is inspired by Microsoft's BitNet research, which explores extreme quantization for neural networks. BitNet demonstrates that models with 1-2 bit weights can achieve competitive performance while significantly reducing computational requirements.

## License

See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
