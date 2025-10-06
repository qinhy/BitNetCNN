import argparse
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models import resnet50, ResNet50_Weights
from huggingface_hub import hf_hub_download
from common_utils import *  # Bit, LitBit, add_common_args, setup_trainer

# -------------------------------------------------
# BitResNet-50: CIFAR or ImageNet stem (toggle)
# -------------------------------------------------
class BottleneckBit(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, scale_op="median"):
        super().__init__()
        width = planes
        self.conv1 = Bit.Conv2d(inplanes, width, 1, stride=1, padding=0, bias=True, scale_op=scale_op)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = Bit.Conv2d(width, width, 3, stride=stride, padding=1, bias=True, scale_op=scale_op)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = Bit.Conv2d(width, planes * self.expansion, 1, stride=1, padding=0, bias=True, scale_op=scale_op)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act = nn.SiLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)

class BitResNet50(nn.Module):
    def __init__(self, num_classes, scale_op="median", in_ch=3, small_stem=True):
        super().__init__()
        self.in_ch = in_ch
        self.scale_op = scale_op
        self.num_classes = num_classes
        self.inplanes = 64

        if small_stem:
            # CIFAR stem: 3x3 stride 1, no maxpool
            self.stem = nn.Sequential(
                Bit.Conv2d(in_ch, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True, scale_op=scale_op),
                nn.BatchNorm2d(self.inplanes),
                nn.SiLU(inplace=True),
            )
        else:
            # ImageNet stem: 7x7 stride 2 + maxpool
            self.stem = nn.Sequential(
                Bit.Conv2d(in_ch, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True, scale_op=scale_op),
                nn.BatchNorm2d(self.inplanes),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.layer1 = self._make_layer(BottleneckBit, 64,  3, stride=1, scale_op=scale_op)  # 64  -> 256
        self.layer2 = self._make_layer(BottleneckBit, 128, 4, stride=2, scale_op=scale_op)  # 128 -> 512
        self.layer3 = self._make_layer(BottleneckBit, 256, 6, stride=2, scale_op=scale_op)  # 256 -> 1024
        self.layer4 = self._make_layer(BottleneckBit, 512, 3, stride=2, scale_op=scale_op)  # 512 -> 2048
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            Bit.Linear(512 * BottleneckBit.expansion, num_classes, bias=True, scale_op=scale_op),
        )

    def _make_layer(self, block, planes, blocks, stride, scale_op):
        downsample = None
        out_ch = planes * block.expansion
        if stride != 1 or self.inplanes != out_ch:
            downsample = nn.Sequential(
                Bit.Conv2d(self.inplanes, out_ch, kernel_size=1, stride=stride, bias=True, scale_op=scale_op),
                nn.BatchNorm2d(out_ch),
            )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample, scale_op=scale_op)]
        self.inplanes = out_ch
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None, scale_op=scale_op))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.head(x)

    def clone(self):
        # Preserve stem choice by checking for MaxPool presence
        small_stem = not any(isinstance(m, nn.MaxPool2d) for m in self.stem.modules())
        return BitResNet50(self.num_classes, self.scale_op, self.in_ch, small_stem=small_stem)

# -------------------------------------------------
# Teachers
# -------------------------------------------------
# Teacher: ResNet-50 (CIFAR stem) from HF (100 classes)
class ResNet50CIFAR(ResNet):
    def __init__(self, num_classes=100):
        super().__init__(block=Bottleneck, layers=[3,4,6,3], num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.Identity()

def make_resnet50_cifar_teacher_from_hf(device="cuda"):
    model = ResNet50CIFAR(num_classes=100)
    ckpt_path = hf_hub_download(repo_id="edadaltocg/resnet50_cifar100", filename="pytorch_model.bin")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"[teacher][c100] Missing keys: {missing}")
    if unexpected: print(f"[teacher][c100] Unexpected keys: {unexpected}")
    return model.eval().to(device)

# Teacher: ResNet-50 (ImageNet) from torchvision (1000 classes)
def make_resnet50_imagenet_teacher(device="cuda"):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    return model.to(device)

# ------------- NEW: Tiny-ImageNet teacher (from zeyuanyin/tiny-imagenet) -------------
def _strip_module_prefix(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def make_resnet50_tiny_teacher_from_hf(epochs: int = 200, device: str = "cuda"):
    """
    Load Tiny-ImageNet ResNet-50 weights from `zeyuanyin/tiny-imagenet`.
    epochs: one of {50, 100, 200} to pick rn50_{epochs}ep/checkpoint_best.pth
    """
    assert epochs in (50, 100, 200), "epochs must be 50/100/200"
    # Tiny-ImageNet uses 200 classes and a CIFAR-like stem (3x3 s1, no maxpool)
    model = resnet50(num_classes=200)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # download checkpoint from the model zoo repo
    fname = f"rn50_{epochs}ep/checkpoint_best.pth"
    ckpt_path = hf_hub_download(repo_id="zeyuanyin/tiny-imagenet", filename=fname)

    ckpt = torch.load(ckpt_path, map_location="cpu",weights_only=False)
    # try a few common keys
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    sd = _strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:    print(f"[teacher][tiny][rn50_{epochs}ep] Missing keys: {missing}")
    if unexpected: print(f"[teacher][tiny][rn50_{epochs}ep] Unexpected keys: {unexpected}")
    return model.eval().to(device)

# -------------------------------------------------
# LightningModule wrapper: KD + hints (dataset-aware)
# -------------------------------------------------
class LitBitResNet50KD(LitBit):
    def __init__(self, lr, wd, epochs, label_smoothing=0.1,
                 alpha_kd=0.7, alpha_hint=0.05, T=4.0, scale_op="median",
                 width_mult=1.0, amp=True,
                 export_dir="./checkpoints_kd_rn50",
                 dataset_name='c100',
                 timnet_teacher_epochs: int = 200):  # NEW

        if dataset_name in ['c100','cifar100']:
            num_classes = 100
            small_stem = True
        elif dataset_name in ['imnet','imagenet']:
            num_classes = 1000
            small_stem = False
        elif dataset_name == 'timnet':
            num_classes = 200
            small_stem = True   # Tiny uses the CIFAR-like 3x3 stem (no maxpool)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        student = BitResNet50(num_classes=num_classes, scale_op=scale_op, small_stem=small_stem)

        # Teacher per dataset
        if dataset_name in ['c100','cifar100']:
            teacher = make_resnet50_cifar_teacher_from_hf(device="cpu")
        elif dataset_name in ['imnet','imagenet']:
            teacher = make_resnet50_imagenet_teacher(device="cpu")
        elif dataset_name == 'timnet':
            teacher = make_resnet50_tiny_teacher_from_hf(epochs=timnet_teacher_epochs, device="cpu")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        super().__init__(lr, wd, epochs, label_smoothing,
                         alpha_kd, alpha_hint, T, scale_op,
                         width_mult, amp,
                         export_dir,
                         dataset_name=dataset_name,
                         model_name='resnet',
                         model_size='50',
                         hint_points=["layer1", "layer2", "layer3", "layer4"],
                         student=student,
                         teacher=teacher,
                         num_classes=num_classes)

# -------------------------------------------------
# CLI / main
# -------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p = add_common_args(p)
    p.add_argument("--model_size", type=str, default="50")
    p.add_argument("--dataset", type=str, default="timnet",
                   choices=["c100","imnet","timnet"],
                   help="Dataset to use (affects stems, classes, transforms)")
    p.add_argument("--timnet_teacher_epochs", type=int, default=200,
                   choices=[50, 100, 200],
                   help="Which Tiny-ImageNet ResNet-50 teacher to load from zeyuanyin/tiny-imagenet")
    p.set_defaults(out=None)
    return p.parse_args()

def main():
    args = parse_args()

    if args.out is None:
        args.out = f"./checkpoints_{args.dataset}_rn{args.model_size}"

    export_dir = f"{args.out}_{args.dataset}"

    lit = LitBitResNet50KD(
        lr=args.lr, wd=args.wd, epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        alpha_kd=args.alpha_kd, alpha_hint=args.alpha_hint, T=args.T,
        scale_op=args.scale_op, amp=args.amp, export_dir=export_dir,
        dataset_name=args.dataset, timnet_teacher_epochs=args.timnet_teacher_epochs
    )

    dmargs = dict(
        data_dir=args.data,
        batch_size=args.batch_size,
        num_workers=4,
        aug_mixup=args.mixup,
        aug_cutmix=args.cutmix,
        alpha=args.mix_alpha
    )

    if args.dataset=="c100":
        dm = CIFAR100DataModule(**dmargs)
    elif args.dataset=="imnet":
        dm = ImageNetDataModule(**dmargs)
    elif args.dataset=="timnet":
        dm = TinyImageNetDataModule(**dmargs)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    trainer, dm = setup_trainer(args, lit, dm)
    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)

if __name__ == "__main__":
    main()
