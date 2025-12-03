"""
Test script for PyTorch Hub integration.
Tests loading models locally before pushing to GitHub.
"""

import torch

print("Testing PyTorch Hub integration locally...\n")

# Test 1: Load BitNetCNN for MNIST
print("=" * 60)
print("Test 1: Loading BitNetCNN (MNIST)")
print("=" * 60)
try:
    model = torch.hub.load('.', 'bitnet_mnist', source='local', pretrained=False)
    print(f"[OK] Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == (2, 10), f"Expected output shape (2, 10), got {out.shape}"
    print("  [OK] Forward pass successful\n")
except Exception as e:
    print(f"[FAIL] Failed: {e}\n")

# Test 2: Load BitNetCNN ternary version
print("=" * 60)
print("Test 2: Loading BitNetCNN (ternary)")
print("=" * 60)
try:
    model = torch.hub.load('.', 'bitnet_mnist', source='local', pretrained=True, ternary=True)
    print(f"[OK] Ternary model loaded successfully")

    # Test forward pass
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    print(f"  Output shape: {out.shape}")
    print("  [OK] Ternary forward pass successful\n")
except Exception as e:
    print(f"[FAIL] Failed: {e}\n")

# Test 3: Load BitResNet18
print("=" * 60)
print("Test 3: Loading BitResNet18 (CIFAR-100)")
print("=" * 60)
try:
    model = torch.hub.load('.', 'bitnet_resnet18', source='local', pretrained=False)
    print(f"[OK] Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == (2, 100), f"Expected output shape (2, 100), got {out.shape}"
    print("  [OK] Forward pass successful\n")
except Exception as e:
    print(f"[FAIL] Failed: {e}\n")

# Test 4: Load BitResNet50
print("=" * 60)
print("Test 4: Loading BitResNet50 (CIFAR-100)")
print("=" * 60)
try:
    model = torch.hub.load('.', 'bitnet_resnet50', source='local', pretrained=False)
    print(f"[OK] Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"  Output shape: {out.shape}")
    print("  [OK] Forward pass successful\n")
except Exception as e:
    print(f"[FAIL] Failed: {e}\n")

# Test 5: Load BitMobileNetV2
print("=" * 60)
print("Test 5: Loading BitMobileNetV2 (CIFAR-100)")
print("=" * 60)
try:
    model = torch.hub.load('.', 'bitnet_mobilenetv2', source='local', pretrained=False, width_mult=1.0)
    print(f"[OK] Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"  Output shape: {out.shape}")
    print("  [OK] Forward pass successful\n")
except Exception as e:
    print(f"[FAIL] Failed: {e}\n")

# Test 6: Load BitConvNeXtv2
print("=" * 60)
print("Test 6: Loading BitConvNeXtv2 (CIFAR-100)")
print("=" * 60)
try:
    model = torch.hub.load('.', 'bitnet_convnextv2', source='local', pretrained=False)
    print(f"[OK] Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"  Output shape: {out.shape}")
    print("  [OK] Forward pass successful\n")
except Exception as e:
    print(f"[FAIL] Failed: {e}\n")

exit()
# Test 7: List available models
print("=" * 60)
print("Test 7: Listing available models")
print("=" * 60)
try:
    models = torch.hub.list('qinhy/BitNetCNN', force_reload=True)
    print(f"[OK] Available models from GitHub:")
    for m in models:
        print(f"  - {m}")
    print()
except Exception as e:
    print(f"[Note] Cannot list from GitHub (need to push first): {e}")
    print(f"  Available models defined in hubconf.py:")
    print(f"  - bitnet_mnist")
    print(f"  - bitnet_resnet18")
    print(f"  - bitnet_resnet50")
    print(f"  - bitnet_mobilenetv2")
    print(f"  - bitnet_convnextv2")
    print()

print("=" * 60)
print("All tests completed!")
print("=" * 60)
print("\nNext steps:")
print("1. Push hubconf.py to your GitHub repo")
print("2. Upload trained model checkpoints to GitHub Releases (tag: v1.0)")
print("3. Users can load models with:")
print("   model = torch.hub.load('qinhy/BitNetCNN', 'bitnet_mnist', pretrained=True)")
