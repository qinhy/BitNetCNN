# Quick Start - Using Your Checkpoint

## Current Status

✅ Your checkpoint is ready: `bit_resnet_18_c100_ternary_val_acc@0.71688.pt.zip`
- **Model**: BitResNet18
- **Dataset**: CIFAR-100
- **Accuracy**: 71.69%
- **Type**: Ternary (int8 quantized weights)
- **Format**: ZIP compressed ✅ (recommended for uploading!)

## Option 1: Use Locally (Right Now)

```python
import torch

# Load model with your local checkpoint (.zip works directly!)
model = torch.hub.load('.', 'bitnet_resnet18', source='local',
                       checkpoint_path='bit_resnet_18_c100_ternary_val_acc@0.71688.pt.zip')

# Test it
import torchvision.transforms as transforms
from torchvision import datasets

# CIFAR-100 test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# Evaluate
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f'Accuracy: {100.*correct/total:.2f}%')
```

## Option 2: Upload to GitHub (For Public Use)

### Step 1: Rename (Keep .zip format!)

```bash
# Rename to standard format (keep as .zip for smaller size)
mv bit_resnet_18_c100_ternary_val_acc@0.71688.pt.zip bit_resnet_18_c100_ternary.pt.zip

# ✅ Upload the .zip file - it's 8-9x smaller!
# The hub code automatically extracts it
```

### Step 2: Push Code to GitHub

```bash
git add hubconf.py TORCH_HUB_USAGE.md RELEASE_GUIDE.md test_checkpoint.py
git commit -m "Add PyTorch Hub support with checkpoint loading"
git push
```

### Step 3: Create GitHub Release

1. Go to: `https://github.com/qinhy/BitNetCNN/releases/new`
2. Tag: `v1.0`
3. Title: `BitNetCNN v1.0 - Pre-trained Models`
4. Upload file: `bit_resnet_18_c100_ternary.pt.zip` ⭐ (upload the .zip!)
5. Click "Publish release"

**Why .zip?**
- ~5 MB instead of ~43 MB
- Faster downloads
- GitHub's file size limits

### Step 4: Users Can Load Your Model

```python
import torch

# Automatic download from GitHub releases
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18', pretrained=True)

# Model is ready to use!
# This will download the checkpoint first time, then cache it
```

## What Users Get

When someone runs:
```python
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18', pretrained=True)
```

They get:
- ✅ Your trained model with 71.69% accuracy
- ✅ Automatic download and caching
- ✅ Ready for inference on CIFAR-100
- ✅ Can convert to ternary format with `ternary=True`

## Test Script

Run this to verify everything works:

```bash
# Test local loading
uv run test_checkpoint.py

# Test full hub integration
uv run test_hub.py
```

## Files Created

- ✅ `hubconf.py` - PyTorch Hub configuration
- ✅ `TORCH_HUB_USAGE.md` - Complete usage guide
- ✅ `RELEASE_GUIDE.md` - How to upload checkpoints
- ✅ `test_checkpoint.py` - Test your checkpoint
- ✅ `test_hub.py` - Test all models

## Need Help?

See detailed guides:
- [TORCH_HUB_USAGE.md](TORCH_HUB_USAGE.md) - How users load models
- [RELEASE_GUIDE.md](RELEASE_GUIDE.md) - How to upload checkpoints to GitHub

## Example: Complete Workflow

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

# 1. Load model
model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_resnet18',
                       checkpoint_path='bit_resnet_18_c100_ternary.pt')
model.eval()

# 2. Prepare image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

img = Image.open('test_image.jpg')
img_tensor = transform(img).unsqueeze(0)

# 3. Predict
with torch.no_grad():
    output = model(img_tensor)
    pred = output.argmax(dim=1)
    print(f"Predicted class: {pred.item()}")
```

Your model is ready to share with the world! 🚀
