# GitHub Release Guide for BitNetCNN Checkpoints

This guide explains how to upload your trained model checkpoints to GitHub Releases so users can load them with PyTorch Hub.

## Your Checkpoint

You have: `bit_resnet_18_c100_ternary_val_acc@0.71688.pt.zip`
- **Model**: BitResNet18 for CIFAR-100
- **Accuracy**: 71.69%
- **Type**: Ternary (int8 weights)
- **Format**: ZIP (compressed, recommended!)

## Why Use .zip?

✅ **Recommended**: Upload `.zip` files to GitHub Releases
- Much smaller file size (10-20x smaller)
- Faster downloads for users
- GitHub has file size limits (2GB for free accounts)
- Our hub code automatically handles `.zip` files!

## Steps to Upload

### 1. Prepare Checkpoint Files

**Option A: Keep as .zip (Recommended)**

Just rename for consistency:

```bash
# Rename to standard format
mv bit_resnet_18_c100_ternary_val_acc@0.71688.pt.zip bit_resnet_18_c100_ternary.pt.zip
```

**Option B: Extract to .pt (Not recommended for large files)**

```bash
# Extract if you prefer .pt
unzip bit_resnet_18_c100_ternary_val_acc@0.71688.pt.zip
mv bit_resnet_18_c100_ternary_val_acc@0.71688.pt bit_resnet_18_c100_ternary.pt
```

### 2. Create a GitHub Release

1. Go to your GitHub repository: `https://github.com/qinhy/BitNetCNN`

2. Click on "Releases" (right sidebar or `/releases` in URL)

3. Click "Create a new release"

4. Fill in the release details:
   - **Tag**: `v1.0` (or `v1.0.0`)
   - **Release title**: `BitNetCNN v1.0 - Pre-trained Models`
   - **Description**:
   ```markdown
   ## Pre-trained BitNet Models for CIFAR-100 and MNIST

   This release includes pre-trained models with the following accuracies:

   ### CIFAR-100 Models
   - **BitResNet18 (Ternary)**: 71.69% accuracy
   - File: `bit_resnet_18_c100_ternary.pt`

   ### Usage

   Load models with PyTorch Hub:
   ```python
   import torch
   model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18', pretrained=True)
   ```

   Or load from local checkpoint:
   ```python
   model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18',
                          checkpoint_path='path/to/checkpoint.pt')
   ```

   ### Model Details
   - All models are trained on CIFAR-100
   - Ternary models use 1.58-bit quantization (-1, 0, +1)
   - Compatible with PyTorch 2.0+
   ```

5. Upload your checkpoint file(s):
   - Click "Attach binaries by dropping them here or selecting them"
   - Upload: `bit_resnet_18_c100_ternary.pt.zip` (recommended - compressed)
   - OR: `bit_resnet_18_c100_ternary.pt` (uncompressed)

6. Click "Publish release"

### 3. Using the Released Checkpoint

Once uploaded, users can load your model:

**From GitHub (automatic download):**
```python
import torch

# Load from GitHub releases
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18',
                       pretrained=True)
```

**From local file (supports both .pt and .zip):**
```python
# Load from local .zip checkpoint (recommended)
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18',
                       checkpoint_path='bit_resnet_18_c100_ternary.pt.zip')

# Or from .pt file
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18',
                       checkpoint_path='bit_resnet_18_c100_ternary.pt')
```

## Recommended File Names

For consistency, use these naming conventions when uploading:

### Ternary Checkpoints - .zip (Recommended!)
- `bit_netcnn_small_mnist_ternary.pt.zip`
- `bit_resnet_18_c100_ternary.pt.zip` ⭐ **Your file**
- `bit_resnet_50_c100_ternary.pt.zip`
- `bit_mobilenetv2_1.0_c100_ternary.pt.zip`
- `bit_convnextv2_c100_ternary.pt.zip`

### Alternative: Uncompressed .pt (larger files)
- `bit_netcnn_small_mnist_ternary.pt`
- `bit_resnet_18_c100_ternary.pt`
- `bit_resnet_50_c100_ternary.pt`
- `bit_mobilenetv2_1.0_c100_ternary.pt`
- `bit_convnextv2_c100_ternary.pt`

**Note**: Hub code tries `.zip` first, then falls back to `.pt`

## Checkpoint Format

Your checkpoints should be saved as:

```python
torch.save({
    'model': model.state_dict(),
    'acc_tern': ternary_accuracy,  # or 'acc'
    'epoch': epoch_num,  # optional
}, 'checkpoint.pt')
```

## Testing Locally

Before uploading, test loading your checkpoint:

```bash
uv run test_checkpoint.py
```

## After Release

Update your README.md to include:

```markdown
## Pre-trained Models

Pre-trained models are available via PyTorch Hub:

```python
import torch

# Load BitResNet18 for CIFAR-100 (71.69% accuracy)
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18', pretrained=True)

# Available models:
# - bitnet_mnist (MNIST, 10 classes)
# - bitnet_resnet18 (CIFAR-100, 71.69% acc)
# - bitnet_resnet50 (CIFAR-100)
# - bitnet_mobilenetv2 (CIFAR-100)
# - bitnet_convnextv2 (CIFAR-100)
```

See [TORCH_HUB_USAGE.md](TORCH_HUB_USAGE.md) for complete documentation.
```

## Updating Checkpoints

To update a checkpoint:
1. Edit the existing release
2. Delete the old file
3. Upload the new file with the same name
4. Users will get the new version on next download

## Direct Download URLs

After uploading to release `v1.0`, your checkpoint will be at:

**If you upload .zip (recommended):**
```
https://github.com/qinhy/BitNetCNN/releases/download/v1.0/bit_resnet_18_c100_ternary.pt.zip
```

**If you upload .pt:**
```
https://github.com/qinhy/BitNetCNN/releases/download/v1.0/bit_resnet_18_c100_ternary.pt
```

Both URLs are already configured in `hubconf.py` with automatic fallback!

## File Size Comparison

Example for ResNet18:
- **Uncompressed .pt**: ~43 MB
- **Compressed .pt.zip**: ~5 MB (8-9x smaller!) ✅

Upload the `.zip` to save bandwidth and storage!
