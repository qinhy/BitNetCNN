import argparse
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models import resnet18, ResNet18_Weights
from huggingface_hub import hf_hub_download
from common_utils import *  # provides Bit, LitBit, add_common_args, setup_trainer

# -------------------------------------------------
# BitResNet-18: CIFAR or ImageNet stem (toggle)
# -------------------------------------------------
class BottleneckBit(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, scale_op="median"):
        super().__init__()
        self.conv1 = Bit.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=True, scale_op=scale_op)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = Bit.Conv2d(planes, planes, 3, stride=1, padding=1, bias=True, scale_op=scale_op)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)

class BitResNet18(nn.Module):
    def __init__(self, block, layers, num_classes, scale_op="median", in_ch=3, cifar_stem=True):
        super().__init__()
        self.in_ch = in_ch
        self.scale_op = scale_op
        self.num_classes = num_classes
        self.layers = layers
        self.inplanes = 64

        if cifar_stem:
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

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, scale_op=scale_op)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, scale_op=scale_op)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, scale_op=scale_op)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, scale_op=scale_op)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            Bit.Linear(512, num_classes, bias=True, scale_op=scale_op)
        )

    def _make_layer(self, block, planes, blocks, stride, scale_op):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Bit.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=True, scale_op=scale_op),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, scale_op=scale_op)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, scale_op=scale_op))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.head(x)

    def clone(self):
        # Preserve stem choice by checking for MaxPool presence
        cifar_stem = not any(isinstance(m, nn.MaxPool2d) for m in self.stem.modules())
        return BitResNet18(BottleneckBit, self.layers, self.num_classes, self.scale_op, self.in_ch, cifar_stem=cifar_stem)

# -------------------------------------------------
# Teachers
# -------------------------------------------------
# Teacher: ResNet-18 (CIFAR stem) from HF (100 classes)
class ResNet18CIFAR(ResNet):
    def __init__(self, num_classes=100):
        super().__init__(block=BasicBlock, layers=[2,2,2,2], num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.Identity()

def make_resnet18_cifar_teacher_from_hf(device="cuda"):
    model = ResNet18CIFAR(num_classes=100)
    ckpt_path = hf_hub_download(repo_id="edadaltocg/resnet18_cifar100", filename="pytorch_model.bin")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"[teacher][cifar100] Missing keys: {missing}")
    if unexpected: print(f"[teacher][cifar100] Unexpected keys: {unexpected}")
    return model.eval().to(device)

# Teacher: ResNet-18 (ImageNet) from torchvision (1000 classes)
def make_resnet18_imagenet_teacher(device="cuda"):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model.to(device)

# -------------------------------------------------
# LightningModule wrapper: KD + hints (dataset-aware)
# -------------------------------------------------
class LitBitResNetKD(LitBit):
    def __init__(self, lr, wd, epochs, label_smoothing=0.1,
                 alpha_kd=0.7, alpha_hint=0.05, T=4.0, scale_op="median",
                 width_mult=1.0, amp=True,
                 export_dir="./checkpoints_kd_rn18",
                 dataset_name='c100'):
        """
        dataset_name: 'c100'/'cifar100' or 'imagenet'
        """
        num_classes = 100 if dataset_name in ['c100', 'cifar100'] else 1000
        cifar_stem = dataset_name in ['c100', 'cifar100']

        student = BitResNet18(BottleneckBit, [2,2,2,2], num_classes, scale_op, cifar_stem=cifar_stem)

        # Choose teacher
        if dataset_name in ['c100', 'cifar100']:
            teacher = make_resnet18_cifar_teacher_from_hf(device="cpu")
        elif dataset_name == ['imnet', 'imagenet']:
            teacher = make_resnet18_imagenet_teacher(device="cpu")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        super().__init__(lr, wd, epochs, label_smoothing,
                         alpha_kd, alpha_hint, T, scale_op,
                         width_mult, amp,
                         export_dir,
                         dataset_name=dataset_name,
                         model_name='resnet',
                         model_size='18',
                         hint_points=["layer1", "layer2", "layer3", "layer4"],
                         student=student,
                         teacher=teacher)

# -------------------------------------------------
# CLI / main
# -------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p = add_common_args(p)
    p.add_argument("--model_size", type=str, default="18")
    p.add_argument("--dataset", type=str, default="c100", choices=["c100", "imnet"],
                   help="Dataset to use (affects stems, classes, transforms)")
    p.set_defaults(out=None)
    return p.parse_args()

def main():
    args = parse_args()

    if args.out is None:
        args.out = f"./checkpoints_{args.dataset}_rn{args.model_size}"

    # Normalize dataset_name for LitBit
    export_dir = f"{args.out}_{args.dataset}"

    lit = LitBitResNetKD(
        lr=args.lr, wd=args.wd, epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        alpha_kd=args.alpha_kd, alpha_hint=args.alpha_hint, T=args.T,
        scale_op=args.scale_op, amp=args.amp, export_dir=export_dir,
        dataset_name=args.dataset
    )
    dmargs = dict(
                data_dir=args.data, batch_size=args.batch_size, num_workers=4,
                aug_mixup=args.mixup, aug_cutmix=args.cutmix, alpha=args.mix_alpha
            )
    if args.dataset=="c100":
        dm = CIFAR100DataModule(**dmargs)
    elif args.dataset=="imnet":
        dm = ImageNetDataModule(**dmargs)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    trainer, dm = setup_trainer(args, lit, dm)
    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)

if __name__ == "__main__":
    main()
