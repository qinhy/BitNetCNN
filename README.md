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
- **PyTorch Hub Integration**: Load pre-trained models with a single line of code
- **Multi-GPU Training**: Distributed training support with PyTorch Lightning DDP
- **Training and Inference Modes**: Supports both training with fake quantization and inference with true quantization
- **Knowledge Distillation**: Support for training with knowledge distillation from full-precision teachers
- **Compressed Checkpoints**: Pre-trained models in .zip format (8-9x smaller than .pt files)

## Quick Start with PyTorch Hub

Load pre-trained models instantly:

```python
import torch

# Load BitResNet18 for CIFAR-100 (71.69% accuracy)
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18', pretrained=True)

# Load BitNetCNN for MNIST
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_mnist', pretrained=True)

# Available models:
# - bitnet_mnist (MNIST, 10 classes)
# - bitnet_resnet18 (CIFAR-100, 71.69% acc)
# - bitnet_resnet50 (CIFAR-100)
# - bitnet_mobilenetv2 (CIFAR-100)
# - bitnet_convnextv2 (CIFAR-100)
```

See [docs/TORCH_HUB_USAGE.md](docs/TORCH_HUB_USAGE.md) for complete documentation.

## Installation

```bash
# Clone the repository
git clone https://github.com/qinhy/BitNetCNN.git
cd BitNetCNN

# Install dependencies (using uv for fast installation)
pip install uv
uv pip install -e .
```

### Requirements

- Python >= 3.11
- PyTorch 2.4.1
- torchvision 0.19.1
- pytorch-lightning
- huggingface_hub (for BitResNet18/50)
- ultralytics (for YOLOv8 distillation)

## Repository Structure

```
BitNetCNN/
├── BitNetCNN.py          # BitNetCNN model and training script
├── BitResNet.py          # BitResNet18/50 models and training script
├── BitMobileNetV2.py     # BitMobileNetV2 model and training script
├── BitYOLOv8Distill.py   # Tiny-ImageNet KD from YOLOv8 into Bit-based YOLOv8 student
├── BitConvNeXtv2.py      # BitConvNeXtv2 model and training script
├── common_utils.py       # Shared utilities (Bit layers, LitBit, data modules)
├── hubconf.py            # PyTorch Hub configuration
├── models/               # Pre-trained model checkpoints (.zip format, ) 
│   └── bit_resnet_18_c100_ternary.pt.zip
├── docs/                 # Documentation
│   ├── TORCH_HUB_USAGE.md    # Complete PyTorch Hub guide
│   ├── RELEASE_GUIDE.md      # How to upload checkpoints
│   ├── QUICK_START.md        # Quick reference
│   └── ZIP_SUPPORT.md        # Technical details on .zip support
└── tests/                # Test scripts
    ├── test_hub.py           # Test all hub models
    └── test_checkpoint.py    # Test checkpoint loading
```

## Pre-trained Models

Pre-trained models are available via PyTorch Hub and can be loaded automatically:
MNIST, c100(CIFAR-100), timnet(tiny-imagenet)

| Model | Dataset | Accuracy | Format | Size |
|-------|---------|----------|--------|------|
| BitResNet18 (Ternary) | CIFAR-100 | 71.69% | .zip | ~5 MB |
| BitNetCNN | MNIST | >99% | .pt | ~1 MB |

Load models with:

```python
import torch

# Automatic download from GitHub
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18', pretrained=True)

# Or load from local checkpoint (.zip supported!)
model = torch.hub.load('.', 'bitnet_resnet18', source='local',
                       checkpoint_path='models/bit_resnet_18_c100_ternary.pt.zip')
```

## Training

### Single GPU Training

#### BitNetCNN on MNIST

```bash
python BitNetCNN.py --epochs 10 --batch-size 128 --scale-op median
```

#### BitResNet18 on CIFAR-100 with Knowledge Distillation

```bash
python BitResNet.py --model-size 18 --epochs 100 --batch-size 128 --scale-op median --kd
```

#### BitMobileNetV2 on CIFAR-100

```bash
python BitMobileNetV2.py --epochs 100 --batch-size 128 --scale-op median --kd
```

### Multi-GPU Training

Train on multiple GPUs for faster training:

```bash
# Use all available GPUs
python BitResNet.py --model-size 18 --epochs 100 --batch-size 128 --gpus -1 --kd

# Use specific number of GPUs
python BitResNet.py --model-size 18 --epochs 100 --batch-size 128 --gpus 2 --kd

# Custom DDP strategy (auto-detects Windows/Linux)
python BitResNet.py --model-size 18 --epochs 100 --batch-size 128 --gpus 2 --strategy ddp --kd
```

**Multi-GPU Features:**
- Automatic backend selection (gloo on Windows, NCCL on Linux)
- Distributed Data Parallel (DDP) with PyTorch Lightning
- Scales training speed linearly with number of GPUs

### Training Parameters

Common parameters for all models:

- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--lr`: Learning rate (default: 1e-3)
- `--wd`: Weight decay (default: 1e-4)
- `--scale-op`: Scaling operation for quantization ("mean" or "median")
- `--eval-ternary`: Evaluate using frozen ternary model during training
- `--amp`: Enable mixed precision training on CUDA devices
- `--kd`: Enable knowledge distillation (BitResNet, BitMobileNetV2)
- `--gpus`: Number of GPUs to use (-1 for all, default: 1)
- `--strategy`: Training strategy ("auto" or "ddp")

## Using BitNet Models in Your Code

```python
from BitNetCNN import NetCNN
from common_utils import Bit, convert_to_ternary

# Create a model for training
model = NetCNN(in_channels=1, num_classes=10, scale_op="median")

# Train the model
# ...

# Convert to ternary for efficient inference
ternary_model = convert_to_ternary(model)

# For power-of-two scaling (more efficient)
from common_utils import convert_to_ternary_p2
ternary_p2_model = convert_to_ternary_p2(model)
```

## Model Architectures

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

### BitConvNeXtv2
ConvNeXt v2 architecture with ternary weights:

1. **Stem**: Patchify layer (4×4 patches)
2. **Stages**: Four stages with ConvNeXt blocks
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

- **BitNetCNN**: >99% accuracy on MNIST
- **BitResNet18**: 71.69% accuracy on CIFAR-100 with knowledge distillation (ternary)
- **BitMobileNetV2**: Efficient alternative to full-precision MobileNetV2
- **BitConvNeXtv2**: State-of-the-art architecture with ternary weights

## Testing

Run tests to verify the implementation:

```bash
# Test all PyTorch Hub models
uv run tests/test_hub.py

# Test checkpoint loading
uv run tests/test_checkpoint.py
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [TORCH_HUB_USAGE.md](docs/TORCH_HUB_USAGE.md) - Complete PyTorch Hub guide
- [RELEASE_GUIDE.md](docs/RELEASE_GUIDE.md) - How to upload checkpoints to GitHub
- [QUICK_START.md](docs/QUICK_START.md) - Quick reference for using checkpoints
- [ZIP_SUPPORT.md](docs/ZIP_SUPPORT.md) - Technical details on .zip checkpoint support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{bitnetcnn2024,
  author = {qinhy},
  title = {BitNetCNN: PyTorch Implementation of CNN Architectures with BitNet Principles},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/qinhy/BitNetCNN}
}
```

## Acknowledgments

This implementation is inspired by Microsoft's BitNet research on extreme quantization for neural networks.
# YOLOv8 ➜ BitNet (Tiny-ImageNet)

```bash
python BitYOLOv8Distill.py --epochs 30 --batch-size 256 --teacher-variant yolov8n-cls.pt \
    --width-mult 0.25 --depth-mult 0.34
```

> Install `ultralytics` (`pip install ultralytics`) and provide a YOLOv8 classification checkpoint (e.g. `yolov8n-cls.pt`). The script builds a Bit-quantised YOLOv8-like student, and `--width-mult` / `--depth-mult` let you scale Tiny-ImageNet capacity.
