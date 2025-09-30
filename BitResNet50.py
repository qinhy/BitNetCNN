import os, argparse
from common_utils import *
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
from huggingface_hub import hf_hub_download

# --- Lightning ---
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# ----------------------------
# BitResNet-50 (CIFAR stem, Bottleneck)
# ----------------------------
class BottleneckBit(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, scale_op="median"):
        super().__init__()
        width = planes
        self.conv1 = Bit.Conv2d(inplanes, width, kernel_size=1, stride=1, padding=0, bias=True, scale_op=scale_op)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = Bit.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=True, scale_op=scale_op)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = Bit.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=True, scale_op=scale_op)
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

class BitResNet50CIFAR(nn.Module):
    """ResNet-50 for CIFAR-sized inputs: 3x3 conv stem (no pool), then layers [3,4,6,3]."""
    def __init__(self, num_classes=100, scale_op="median", in_ch=3):
        super().__init__()
        self.in_ch = in_ch
        self.num_classes = num_classes
        self.scale_op = scale_op
        self.inplanes = 64
        self.stem = nn.Sequential(
            Bit.Conv2d(in_ch, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True, scale_op=scale_op),
            nn.BatchNorm2d(self.inplanes),
            nn.SiLU(inplace=True),
        )
        self.layer1 = self._make_layer(BottleneckBit, 64,  3, stride=1, scale_op=scale_op)  # 64 -> 256
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
        out_channels = planes * block.expansion
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                Bit.Conv2d(self.inplanes, out_channels, kernel_size=1, stride=stride, bias=True, scale_op=scale_op),
                nn.BatchNorm2d(out_channels),
            )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample, scale_op=scale_op)]
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None, scale_op=scale_op))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.head(x)

    def clone(self):
        return BitResNet50CIFAR(self.num_classes, self.scale_op, self.in_ch)
# ----------------------------
# Teacher: ResNet-50 (CIFAR stem) from HF
# ----------------------------
from torchvision.models.resnet import ResNet, Bottleneck  # ensure Bottleneck is imported

class ResNet50CIFAR(ResNet):
    def __init__(self, num_classes=100):
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)
        # CIFAR stem: 3x3, stride=1, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.Identity()

def make_resnet50_cifar_teacher_from_hf(device="cuda"):
    model = ResNet50CIFAR(num_classes=100)
    ckpt_path = hf_hub_download(repo_id="edadaltocg/resnet50_cifar100", filename="pytorch_model.bin")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"[teacher] Missing keys: {missing}")
    if unexpected: print(f"[teacher] Unexpected keys: {unexpected}")
    return model.eval().to(device)

# ----------------------------
# LightningModule: KD + hints + ternary eval/export
# ----------------------------
class LitBitResNet50KD(LitBit):
    def __init__(self, lr, wd, epochs, label_smoothing=0.1,
                 alpha_kd=0.7, alpha_hint=0.05, T=4.0, scale_op="median",
                 width_mult=1.0, amp=True,
                 export_dir="./checkpoints_c100_rn50"):
        super().__init__(lr, wd, epochs, label_smoothing,
                 alpha_kd, alpha_hint, T, scale_op,
                 width_mult, amp,
                 export_dir,
                 dataset_name='c100',
                 model_name='resnet',
                 model_size='50',
                 hint_points=["layer1", "layer2", "layer3", "layer4"],
                 student=BitResNet50CIFAR(100, scale_op),
                 teacher=make_resnet50_cifar_teacher_from_hf(device="cpu"),
                 )
# ----------------------------
# CLI / main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--out",  type=str, default="./ckpt_c100_kd_rn50")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--alpha-kd", type=float, default=0.3)  # keep default off; teacher optional
    p.add_argument("--alpha-hint", type=float, default=0.05)
    p.add_argument("--T", type=float, default=4.0)
    p.add_argument("--scale-op", type=str, default="median", choices=["mean","median"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--mixup", action="store_true")
    p.add_argument("--cutmix", action="store_true")
    p.add_argument("--mix-alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    dm = CIFAR100DataModule(
        data_dir=args.data, batch_size=args.batch_size, num_workers=4,
        aug_mixup=args.mixup, aug_cutmix=args.cutmix, alpha=args.mix_alpha
    )

    lit = LitBitResNet50KD(
        lr=args.lr, wd=args.wd, epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        alpha_kd=args.alpha_kd, alpha_hint=args.alpha_hint, T=args.T,
        scale_op=args.scale_op, amp=args.amp, export_dir=args.out
    )

    os.makedirs(args.out, exist_ok=True)
    logger = CSVLogger(save_dir=args.out, name="logs")
    ckpt_cb = ModelCheckpoint(monitor="val/acc_tern", mode="max", save_top_k=1, save_last=True)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    export_cb = ExportBestTernary(args.out, monitor="val/acc_tern", mode="max")
    callbacks = [ckpt_cb, lr_cb, export_cb]

    accelerator = "cpu" if args.cpu else "auto"
    precision = "16-mixed" if args.amp else "32-true"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        deterministic=False,
    )

    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)

if __name__ == "__main__":
    main()
