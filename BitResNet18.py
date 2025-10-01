import argparse
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
from huggingface_hub import hf_hub_download
from common_utils import *

# ----------------------------
# BitResNet-18 (CIFAR stem)
# ----------------------------
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

class BitResNet18CIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=100, scale_op="median", in_ch=3):
        super().__init__()
        self.in_ch = in_ch
        self.scale_op = scale_op
        self.num_classes = num_classes
        self.layers = layers
        self.inplanes = 64
        self.stem = nn.Sequential(
            Bit.Conv2d(in_ch, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True, scale_op=scale_op),
            nn.BatchNorm2d(self.inplanes),
            nn.SiLU(inplace=True),
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
        return BitResNet18CIFAR(BottleneckBit, self.layers, self.num_classes, self.scale_op, self.in_ch)

# ----------------------------
# Teacher: ResNet-18 (CIFAR stem) from HF
# ----------------------------
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
    if missing:   print(f"[teacher] Missing keys: {missing}")
    if unexpected: print(f"[teacher] Unexpected keys: {unexpected}")
    return model.eval().to(device)

# ----------------------------
# LightningModule: KD + hints + ternary eval/export
# ----------------------------
class LitBitResNetKD(LitBit):
    def __init__(self, lr, wd, epochs, label_smoothing=0.1,
                 alpha_kd=0.7, alpha_hint=0.05, T=4.0, scale_op="median",
                 width_mult=1.0, amp=True,
                 export_dir="./checkpoints_c100_rn18"):
        super().__init__(lr, wd, epochs, label_smoothing,
                 alpha_kd, alpha_hint, T, scale_op,
                 width_mult, amp,
                 export_dir,
                 dataset_name='c100',
                 model_name='resnet',
                 model_size='18',
                 hint_points=["layer1", "layer2", "layer3", "layer4"],
                 student=BitResNet18CIFAR(BottleneckBit, [2,2,2,2], 100, scale_op),
                 teacher=make_resnet18_cifar_teacher_from_hf(device="cpu"),
                 )
# ----------------------------
# CLI / main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p = add_common_args(p)
    p.set_defaults(out="./ckpt_c100_kd_rn18")
    return p.parse_args()

def main():
    args = parse_args()

    lit = LitBitResNetKD(
        lr=args.lr, wd=args.wd, epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        alpha_kd=args.alpha_kd, alpha_hint=args.alpha_hint, T=args.T,
        scale_op=args.scale_op, amp=args.amp, export_dir=args.out
    )

    trainer, dm = setup_trainer(args, lit)
    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)

if __name__ == "__main__":
    main()
