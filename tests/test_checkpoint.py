"""
Test loading your checkpoint file.
"""
import torch
import zipfile
import os

# Path to your checkpoint
checkpoint_zip = "./models/bit_resnet_18_c100_ternary.zip"
checkpoint_pt = "./models/bit_resnet_18_c100_ternary_val_acc@0.71688.pt"

# Extract if needed
if os.path.exists(checkpoint_zip) and not os.path.exists(checkpoint_pt):
    print(f"Extracting {checkpoint_zip}...")
    with zipfile.ZipFile(checkpoint_zip, 'r') as zip_ref:
        zip_ref.extractall('./models')
    print(f"Extracted to {checkpoint_pt}")

# Load checkpoint to inspect
if os.path.exists(checkpoint_pt):
    print(f"\nLoading checkpoint: {checkpoint_pt}")
    checkpoint = torch.load(checkpoint_pt, map_location='cpu', weights_only=False)

    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print(f"State dict is under 'model' key")
    else:
        state_dict = checkpoint
        print(f"Checkpoint is the state dict directly")

    print(f"\nNumber of parameters: {len(state_dict)}")
    print(f"Sample keys:")
    for i, key in enumerate(list(state_dict.keys())[:5]):
        print(f"  {key}: {state_dict[key].shape}")

    # Check for metadata
    if 'acc_tern' in checkpoint:
        print(f"\nTernary accuracy: {checkpoint['acc_tern']}")
    if 'acc' in checkpoint:
        print(f"Accuracy: {checkpoint['acc']}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")

    # Try loading with hub
    print("\n" + "="*60)
    print("Testing with PyTorch Hub...")
    print("="*60)

    try:
        model = torch.hub.load('.', 'bitnet_resnet18', source='local',
                               checkpoint_path=checkpoint_pt)
        print(f"\n[OK] Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test inference
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        print(f"[OK] Forward pass successful!")

    except Exception as e:
        print(f"[FAIL] Error loading model: {e}")
        import traceback
        traceback.print_exc()

else:
    print(f"Checkpoint file not found: {checkpoint_pt}")
    print(f"Please make sure the file exists in the current directory")
