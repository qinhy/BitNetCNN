from typing import Literal, Optional

from pydantic import Field
from pydanticV2_argparse import ArgumentParser
import torch
import torch.nn as nn

from bitlayers import convs
from bitlayers.bit import Bit
from bitlayers.acts import ActModels
from bitlayers.norms import NormModels
from common_utils import summ
from dataset import DataModuleConfig
from trainer import AccelTrainer, CommonTrainConfig, LitBit, LitBitConfig

torch.set_float32_matmul_precision("high")

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
      - Collects stage-end names in self.hint_points for hooks
    """
    def __init__(self, num_classes=100, width_mult=1.0, round_nearest=8,
                 scale_op="median", in_ch=3, act=ActModels.SiLU(inplace=True), last_channel_override=None):
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

        def norm():
            return NormModels.BatchNorm2d(num_features=-1)

        self.stem = ConvBNActBit(in_channels=in_ch, out_channels=input_channel,
                                 kernel_size=3,stride=1,padding=1,scale_op=scale_op,
                                 norm=norm(),
                                 act=act).build()

        features = []
        self.hint_points = []
        for t, c, n, s in setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidualBit(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        stride=stride,
                        exp_ratio=t,
                        scale_op=scale_op,
                        conv_pw_exp_layer=convs.Conv2dModels.Conv2dPointwiseNormAct(
                            in_channels=-1, norm=norm(), act=act
                        ),
                        conv_dw_layer=convs.Conv2dModels.Conv2dDepthwiseNormAct(
                            in_channels=-1, norm=norm(), act=act
                        ),
                        conv_pw_layer=convs.Conv2dModels.Conv2dPointwiseNorm(
                            in_channels=-1, norm=norm()
                        ),
                    ).build()
                )
                input_channel = output_channel
            self.hint_points.append((f"features.{len(features)-1}.conv_pw.conv",f"features.{len(features)}"))
        self.features = nn.Sequential(*features)

        self.head_conv = ConvBNActBit(in_channels=input_channel, out_channels=last_channel,
                                 kernel_size=1,stride=1,padding=0,scale_op=scale_op,
                                 norm=norm(),
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
    return teacher.eval().to(device)


def make_mobilenetv2_teacher_imagenet(device: str = "cuda"):
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    return model.eval().to(device)

# ----------------------------
# LightningModule: KD + hints + ternary eval/export
# ----------------------------
class LitBitMBv2KD(LitBit):
    def __init__(self, config:LitBitConfig,
                 width_mult,
                 teacher_variant: str = "cifar100_mobilenetv2_x1_4",
                 teacher_device: str = "cpu"):

        config = LitBitConfig.model_validate(config)
        if config.dataset is None:
            raise ValueError("LitBitMBv2KD requires config.dataset to be set (DataModuleConfig).")

        config.student = student = BitMobileNetV2(
            num_classes=config.dataset.num_classes,
            width_mult=width_mult,
            scale_op=config.scale_op,
        )
        config.hint_points = student.hint_points

        # Teacher per dataset
        ds = config.dataset.dataset_name
        if ds == "c100":
            config.teacher = make_mobilenetv2_teacher_from_hub(teacher_variant, device=teacher_device)
        elif ds == "imnet":
            config.teacher = make_mobilenetv2_teacher_imagenet(device=teacher_device)
        elif ds == "timnet":
            config.teacher = None
        else:
            raise ValueError(f"Unsupported dataset: {ds}")
        
        config.model_name = config.model_name or "mbv2"
        config.model_size = config.model_size or f"x{int(width_mult*100)}"

        super().__init__(config)
        
# ----------------------------
# CLI / main
# ----------------------------
class Config(CommonTrainConfig):
    data: Optional[str] = Field(default=None, description="Alias for --data_dir (back-compat).")
    width_mult: float = Field(
        default=1.4,
        description="Width multiplier for MobileNetV2.",
    )
    model_name: str = "mbv2"
    model_size: str = ""

    teacher_variant: str = Field(
        default="cifar100_mobilenetv2_x1_4",
        description="Teacher checkpoint/variant name.",
    )

    num_workers: int = 1

    batch_size:int = 512
    epochs:int = 10

    mixup: bool = True
    cutmix: bool = True

    lr: float = 0.2
    alpha_kd: float = 0.3
    alpha_hint: float = 0.05

    scale_op:str="median"
    export_dir:str="./ckpt_c100_mbv2"
    dataset_name:str='c100'

def main():
    parser = ArgumentParser(model=Config)
    args = parser.parse_typed_args()

    if args.data is not None:
        args.data_dir = args.data

    dm = DataModuleConfig.model_validate(args.model_dump())
    config = LitBitConfig.model_validate(args.model_dump())
    config.dataset = dm.model_copy()

    args.export_dir = config.export_dir = f"./ckpt_{config.dataset.dataset_name}_mbv2"
    config.model_name = "mbv2"
    config.model_size = f"x{int(args.width_mult*100)}"

    lit = LitBitMBv2KD(
        config=config,
        width_mult=args.width_mult,
        teacher_variant=args.teacher_variant,
        teacher_device="cpu",
    )

    trainer = AccelTrainer(
        max_epochs=args.epochs,
        mixed_precision="fp16" if args.amp else "no",
        gradient_accumulation_steps=1,
        log_every_n_steps=10,
    )
    trainer.fit(lit, datamodule=dm.build())


if __name__ == "__main__":
    main()
