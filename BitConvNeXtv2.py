import argparse
from typing import Literal
from pydantic import Field
from pydanticV2_argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from bitlayers.drop import DropPath
from bitlayers.weight_init import trunc_normal_
from common_utils import *
from dataset.config import DataModuleConfig
from trainer import AccelTrainer, CommonTrainConfig, LitBit, LitBitConfig  # provides Bit, LitBit, add_common_args, setup_trainer, *DataModule classes if available

EPS = 1e-12

# ----------------------------
# Utilities
# ----------------------------
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

# ----------------------------
# ConvNeXt V2 blocks
# ----------------------------
class ConvNeXtV2Block(nn.Module):
    """ ConvNeXtV2 Block. """
    def __init__(self, dim, kernel_size=7, drop_path=0., scale_op="median"):
        super().__init__()
        # depthwise conv
        self.dwconv = Bit.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, scale_op=scale_op)
        self.norm = LayerNorm(dim, eps=1e-6)
        # MLP (pointwise)
        self.pwconv1 = Bit.Linear(dim, 4 * dim, scale_op=scale_op)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = Bit.Linear(4 * dim, dim, scale_op=scale_op)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x:torch.Tensor):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # back to (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2 """
    def __init__(
        self, in_chans=3, num_classes=1000,
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
        drop_path_rate=0., head_init_scale=1., scale_op="median"
    ):
        super().__init__()
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        self.head_init_scale = head_init_scale

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            Bit.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, scale_op=scale_op),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                Bit.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2, scale_op=scale_op),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dim=dims[i], drop_path=dp_rates[cur + j], scale_op=scale_op) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = Bit.Linear(dims[-1], num_classes, scale_op=scale_op)

        self.apply(self._init_weights)
        with torch.no_grad():
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (Bit.Conv2d, Bit.Linear)):
            trunc_normal_(m.weight, std=.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # GAP
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def clone(self):
        return ConvNeXtV2(self.in_chans, self.num_classes, self.depths, self.dims, self.drop_path_rate, self.head_init_scale)

    @staticmethod
    def convnextv2(size='atto', num_classes=100, drop_path_rate=0.3, scale_op="median"):
        params = {
            "atto":  dict(depths=[2, 2, 6, 2],  dims=[40, 80, 160, 320]),
            "femto": dict(depths=[2, 2, 6, 2],  dims=[48, 96, 192, 384]),
            "pico":  dict(depths=[2, 2, 6, 2],  dims=[64, 128, 256, 512]),
            "nano":  dict(depths=[2, 2, 8, 2],  dims=[80, 160, 320, 640]),
            "tiny":  dict(depths=[3, 3, 9, 3],  dims=[96, 192, 384, 768]),
            "base":  dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
            "large": dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),
            "huge":  dict(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816]),
        }
        if size not in params:
            raise ValueError(f"Unknown ConvNeXtV2 size: {size}")
        return ConvNeXtV2(**params[size], num_classes=num_classes, drop_path_rate=drop_path_rate, scale_op=scale_op)

# ----------------------------
# Teacher: convnextv2 via timm
# ----------------------------
def make_convnextv2_from_timm(size='pico', device="cuda", pretrained=True):
    import timm
    # FCMAE fine-tuned IN1K variants exist for convnextv2_<size>.fcmae_ft_in1k
    model_name = f'convnextv2_{size}.fcmae_ft_in1k' if pretrained else f'convnextv2_{size}'
    m = timm.create_model(model_name, pretrained=pretrained)
    return m.eval().to(device)

def make_teacher_for_dataset(size: str, dataset: str, num_classes: int, device: str = "cpu", pretrained: bool = True):
    """
    Builds a ConvNeXtV2 teacher for the target dataset.
    - For ImageNet-1k ('imnet'): use timm IN1K-pretrained with correct head (1000).
    - For others: optionally load IN1K-pretrained backbone, then REPLACE the head to match num_classes.
      (New head is randomly initialized; KD + hints still work. If you rely on KD logits, consider a dataset-specific teacher.)
    """
    t = make_convnextv2_from_timm(size=size, device=device, pretrained=pretrained)
    # If teacher head classes != dataset classes, swap head
    head_out = getattr(t, 'num_classes', None)
    if head_out is None:
        # timm models typically set .num_classes; fallback by inspecting last linear
        last = getattr(t, 'head', None)
        head_out = last.out_features if isinstance(last, nn.Linear) else 1000

    if head_out != num_classes:
        # Replace classifier to match dataset
        in_f = t.get_classifier().in_features if hasattr(t, "get_classifier") else (t.head.in_features if hasattr(t, "head") else None)
        if in_f is None:
            # fall back: try named attribute
            in_f = t.head.in_features
        new_head = nn.Linear(in_f, num_classes)
        if hasattr(t, "reset_classifier"):
            # timm utility (will handle .head/.fc as appropriate)
            t.reset_classifier(num_classes=num_classes)
        else:
            t.head = new_head
        # Expose num_classes attr if not present
        t.num_classes = num_classes
    return t.eval().to(device)

# ----------------------------
# CLI / main
# ----------------------------
class Config(CommonTrainConfig):
    dataset_name: str = "timnet"
    model_size: Literal[
        "atto", "femto", "pico", "nano",
        "tiny", "base", "large", "huge",
    ] = Field(
        default="nano",
        description="Model size preset.",
    )

    drop_path_rate: float = Field(
        default=0.1,
        description="Stochastic depth drop-path rate.",
    )

    teacher_pretrained: bool = Field(
        default=True,
        description=(
            "Use ImageNet-pretrained teacher backbone when classes != 1000 "
            "(head is replaced)."
        ),
    )

    out: Optional[str] =None
    batch_size: int =512
    num_workers: int = 0

    lr: float =0.2
    alpha_kd: float =0.0
    alpha_hint: float =0.1

def main() -> None:
    parser = ArgumentParser(model=Config)
    args = parser.parse_typed_args()
    dm = DataModuleConfig.model_validate(args.model_dump())
    config = LitBitConfig.model_validate(args.model_dump())
    config.dataset = dm.model_copy()
    dataset_name = config.dataset.dataset_name
    num_classes = config.dataset.num_classes
    config.export_dir = args.export_dir = f"./ckpt_{config.dataset.dataset_name}_conxt_{config.model_size}"

    config.model_size = model_size = str(config.model_size)

    config.student = ConvNeXtV2.convnextv2(model_size, num_classes=num_classes,
                                           drop_path_rate=args.drop_path_rate)
    config.teacher = make_teacher_for_dataset(
            size=model_size,
            dataset=dataset_name,
            num_classes=num_classes,
            device="cpu",
            pretrained=True
    )
    config.model_name="convnxtv2"
    config.hint_points=['stages.0','stages.1','stages.2','stages.3']
    
    lit = LitBit(config)
    dm = dm.build()
    
    trainer = AccelTrainer(
        max_epochs=args.epochs,
        mixed_precision="bf16" if args.amp else "no",
        gradient_accumulation_steps=1,
        log_every_n_steps=10,
    )
    trainer.fit(lit, datamodule=dm)

if __name__ == "__main__":
    main()
    