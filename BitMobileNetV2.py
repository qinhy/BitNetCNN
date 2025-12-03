import argparse
from pydanticV2_argparse import ArgumentParser
import torch
import torch.nn as nn
from bitlayers import convs
from bitlayers.acts import ActModels
from bitlayers.norms import NormModels
from common_utils import *

# ----------------------------
# MobileNetV2 (Bit) blocks
# ----------------------------
def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

ConvBNActBit = convs.Conv2dModels.Conv2dNormAct
# class ConvBNActBit(nn.Module):
#     def __init__(self, in_ch, out_ch, k, s, p, groups=1, scale_op="median", act="silu"):
#         super().__init__()
#         self.conv = Bit.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False,
#                               groups=groups, scale_op=scale_op)
#         self.norm   = nn.BatchNorm2d(out_ch)
#         if act == "relu6":
#             self.act = nn.ReLU6(inplace=True)
#         elif act == "silu":
#             self.act = nn.SiLU(inplace=True)
#         else:
#             self.act = nn.Identity()
#     def forward(self, x):
#         return self.act(self.norm(self.conv(x)))

InvertedResidualBit = convs.Conv2dModels.InvertedResidual
# class InvertedResidualBit(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio, scale_op="median", act="silu"):
#         super().__init__()
#         assert stride in [1, 2]
#         hidden_dim = int(round(inp * expand_ratio))
#         self.use_res_connect = (stride == 1 and inp == oup)
#         layers = []
#         if expand_ratio != 1:
#             layers.append(ConvBNActBit(inp, hidden_dim, 1, 1, 0, scale_op=scale_op, act=act))
#         layers.append(ConvBNActBit(hidden_dim, hidden_dim, 3, stride, 1,
#                                    groups=hidden_dim, scale_op=scale_op, act=act))
#         layers.append(Bit.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False, scale_op=scale_op))
#         layers.append(nn.BatchNorm2d(oup))
#         self.block = nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.block(x)
#         if self.use_res_connect:
#             out = x + out
#         return out

class BitMobileNetV2(nn.Module):
    """
    CIFAR-friendly BitMobileNetV2:
      - 3x3 s=1 stem
      - MobileNetV2 setting (CIFAR-sized)
      - Collects stage-end names in self.hint_names for hooks
    """
    def __init__(self, num_classes=100, width_mult=1.0, round_nearest=8,
                 scale_op="median", in_ch=3, act=ActModels.SiLU(), last_channel_override=None):
        super().__init__()
        self.num_classes, self.width_mult, self.round_nearest, self.scale_op, self.in_ch, self.act, self.last_channel_override = \
            num_classes, width_mult, round_nearest, scale_op, in_ch, act, last_channel_override
        setting = [
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = _make_divisible(32 * width_mult, round_nearest)
        if last_channel_override is not None:
            last_channel = _make_divisible(last_channel_override * max(1.0, width_mult), round_nearest)
        else:
            last_channel = _make_divisible(1280 * max(1.0, width_mult), round_nearest)

        self.stem = ConvBNActBit(in_channels=in_ch,out_channels=input_channel,
                                 kernel_size=3,stride=1,padding=1,scale_op=scale_op,
                                 norm=NormModels.BatchNorm2d(num_features=-1),
                                 act=act).build()

        features = []
        self.hint_names = []
        for t, c, n, s in setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidualBit(in_channels=input_channel, out_channels=output_channel, stride=stride,
                                        expand_ratio=t, scale_op=scale_op,act=act).build()
                )
                input_channel = output_channel
            self.hint_names.append(f"features.{len(features)-1}")
        self.features = nn.Sequential(*features)

        self.head_conv = ConvBNActBit(in_channels=input_channel,out_channels=last_channel,
                                 kernel_size=1,stride=1,padding=0,scale_op=scale_op,
                                 norm=NormModels.BatchNorm2d(num_features=-1),
                                 act=act).build()
        
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = Bit.Linear(last_channel, num_classes, bias=True, scale_op=scale_op)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Bit.Conv2d):
                if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, Bit.Linear):
                if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
                    nn.init.normal_(m.weight, 0.0, 0.01)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head_conv(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
    
    def clone(self):
        return BitMobileNetV2(self.num_classes, self.width_mult, self.round_nearest,
                              self.scale_op, self.in_ch, self.act, self.last_channel_override)

# ----------------------------
# Teacher: MobileNetV2 from torch.hub
# ----------------------------
def make_mobilenetv2_teacher_from_hub(variant="cifar100_mobilenetv2_x1_4", device="cuda"):
    teacher = torch.hub.load("chenyaofo/pytorch-cifar-models", variant, pretrained=True)
    return teacher.to(device).eval()

def make_resnet18_tiny_teacher_from_self(device: str = "cuda"):
    model = torch.hub.load('.', 'bitnet_resnet18', source='local',
                        pretrained = True,dataset= "timnet",ternary = True)
    return model.to(device)

# ----------------------------
# LightningModule: KD + hints + ternary eval/export
# ----------------------------
class LitBitMBv2KD(LitBit):
    def __init__(self, lr, wd, epochs, label_smoothing=0.1,
                 alpha_kd=0.7, alpha_hint=0.05, T=4.0, scale_op="median",
                 width_mult=1.0, amp=True, teacher_variant="cifar100_mobilenetv2_x1_4",
                 export_dir="./ckpt_c100_mbv2",
                 dataset_name='c100',
                 timnet_teacher_epochs: int = 200):  # NEW

        if dataset_name in ['c100','cifar100']:
            num_classes = 100
        elif dataset_name == 'timnet':
            num_classes = 200
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        student = BitMobileNetV2(num_classes=num_classes, width_mult=width_mult, scale_op=scale_op)

        # Teacher per dataset
        if dataset_name in ['c100','cifar100']:
            teacher = make_mobilenetv2_teacher_from_hub(teacher_variant, device="cpu")
        elif dataset_name == 'timnet':
            teacher = make_resnet18_tiny_teacher_from_self()
            alpha_hint = 0.0
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        super().__init__(lr, wd, epochs, label_smoothing,
        alpha_kd, alpha_hint, T, scale_op,
        width_mult, amp,
        export_dir,
        dataset_name=dataset_name,
        model_name='mobilenetv2',
        model_size='x140',
        hint_points=student.hint_names,
        student=student,
        teacher=teacher,
        num_classes=num_classes)
        
# ----------------------------
# CLI / main
# ----------------------------
class Config(CommonTrainConfig):
    width_mult: float = Field(
        default=1.4,
        description="Width multiplier for MobileNetV2.",
    )

    teacher_variant: str = Field(
        default="cifar100_mobilenetv2_x1_4",
        description="Teacher checkpoint/variant name.",
    )

    dataset: Literal["c100", "timnet"] = Field(
        default="timnet",
        description="Dataset to use (affects classes, transforms).",
    )

    timnet_teacher_epochs: Literal[50, 100, 200] = Field(
        default=200,
        description=(
            "Which Tiny-ImageNet MobileNetV2 teacher to load from "
            "zeyuanyin/tiny-imagenet."
        ),
    )
    
    out:Optional[str]=None
    batch_size:int=16
    
def main():
    parser = ArgumentParser(model=Config)
    args = parser.parse_typed_args()

    if args.out is None:
        args.out = f"./ckpt_{args.dataset}_mbv2"

    lit = LitBitMBv2KD(
        lr=args.lr, wd=args.wd, epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        alpha_kd=args.alpha_kd, alpha_hint=args.alpha_hint, T=args.T,
        scale_op=args.scale_op, width_mult=args.width_mult,
        amp=args.amp, teacher_variant=args.teacher_variant,
        export_dir=args.out,
        dataset_name=args.dataset,
        timnet_teacher_epochs=args.timnet_teacher_epochs
    )

    dmargs = dict(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        mixup=args.mixup,
        cutmix=args.cutmix,
        mix_alpha=args.mix_alpha
    )

    if args.dataset == "c100":
        dm = CIFAR100DataModule(**dmargs)
    elif args.dataset == "timnet":
        dm = TinyImageNetDataModule(**dmargs)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    trainer, dm = setup_trainer(args, lit, dm)
    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)

if __name__ == "__main__":
    main()
