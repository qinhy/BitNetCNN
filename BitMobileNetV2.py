import argparse
import torch
import torch.nn as nn
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

class ConvBNActBit(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, groups=1, scale_op="median", act="silu"):
        super().__init__()
        self.conv = Bit.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False,
                              groups=groups, scale_op=scale_op)
        self.bn   = nn.BatchNorm2d(out_ch)
        if act == "relu6":
            self.act = nn.ReLU6(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class InvertedResidualBit(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, scale_op="median", act="silu"):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNActBit(inp, hidden_dim, 1, 1, 0, scale_op=scale_op, act=act))
        layers.append(ConvBNActBit(hidden_dim, hidden_dim, 3, stride, 1,
                                   groups=hidden_dim, scale_op=scale_op, act=act))
        layers.append(Bit.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False, scale_op=scale_op))
        layers.append(nn.BatchNorm2d(oup))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connect:
            out = x + out
        return out

class BitMobileNetV2(nn.Module):
    """
    CIFAR-friendly BitMobileNetV2:
      - 3x3 s=1 stem
      - MobileNetV2 setting (CIFAR-sized)
      - Collects stage-end names in self.hint_names for hooks
    """
    def __init__(self, num_classes=100, width_mult=1.0, round_nearest=8,
                 scale_op="median", in_ch=3, act="silu", last_channel_override=None):
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

        self.stem = ConvBNActBit(in_ch, input_channel, k=3, s=1, p=1, scale_op=scale_op, act=act)

        features = []
        self.hint_names = []
        for t, c, n, s in setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidualBit(input_channel, output_channel, stride,
                                        expand_ratio=t, scale_op=scale_op, act=act)
                )
                input_channel = output_channel
            self.hint_names.append(f"features.{len(features)-1}")
        self.features = nn.Sequential(*features)

        self.head_conv  = ConvBNActBit(input_channel, last_channel, k=1, s=1, p=0,
                                       scale_op=scale_op, act=act)
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

# ----------------------------
# LightningModule: KD + hints + ternary eval/export
# ----------------------------
class LitBitMBv2KD(LitBit):
    def __init__(self, lr, wd, epochs, label_smoothing=0.1,
                 alpha_kd=0.7, alpha_hint=0.05, T=4.0, scale_op="median",
                 width_mult=1.0, amp=True,teacher_variant="cifar100_mobilenetv2_x1_4",
                 export_dir="./checkpoints_c100_mbv2"):
        super().__init__(lr, wd, epochs, label_smoothing,
        alpha_kd, alpha_hint, T, scale_op,
        width_mult, amp,
        export_dir,
        dataset_name='c100',
        model_name='mobilenetv2',
        model_size='x140',
        hint_points=BitMobileNetV2(num_classes=100, width_mult=width_mult, scale_op=scale_op).hint_names,
        student=BitMobileNetV2(num_classes=100, width_mult=width_mult, scale_op=scale_op),
        teacher=make_mobilenetv2_teacher_from_hub(teacher_variant, device="cpu"),)
        
# ----------------------------
# CLI / main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p = add_common_args(p)
    p.set_defaults(out="./ckpt_c100_kd_mbv2", batch_size=512)
    p.add_argument("--width-mult", type=float, default=1.4)
    p.add_argument("--teacher-variant", type=str, default="cifar100_mobilenetv2_x1_4")
    return p.parse_args()

def main():
    args = parse_args()

    lit = LitBitMBv2KD(
        lr=args.lr, wd=args.wd, epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        alpha_kd=args.alpha_kd, alpha_hint=args.alpha_hint, T=args.T,
        scale_op=args.scale_op, width_mult=args.width_mult,
        amp=args.amp, teacher_variant=args.teacher_variant,
        export_dir=args.out
    )

    trainer, dm = setup_trainer(args, lit)
    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)

if __name__ == "__main__":
    main()
