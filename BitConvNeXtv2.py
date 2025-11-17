import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import trunc_normal_, DropPath
from common_utils import *  # provides Bit, LitBit, add_common_args, setup_trainer, *DataModule classes if available

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

    def forward(self, x):
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
# LightningModule: KD + hints
# ----------------------------
class LitConvNeXtV2KD(LitBit):
    def __init__(
        self,
        lr, wd, epochs,
        dataset_name='c100',
        model_size="pico",
        label_smoothing=0.1, alpha_kd=0.0, alpha_hint=0.0, T=4.0,
        amp=True, export_dir="./ckpt_convnextv2",
        drop_path_rate=0.1, scale_op="median",
        teacher_pretrained=True
    ):
        # dataset -> classes
        dataset_name = dataset_name.lower()
        if dataset_name in ['c10', 'cifar10']:
            num_classes = 10
        elif dataset_name in ['c100', 'cifar100']:
            num_classes = 100
        elif dataset_name in ['timnet', 'tiny', 'tinyimagenet', 'tiny-imagenet']:
            num_classes = 200
        elif dataset_name in ['imnet', 'imagenet', 'in1k', 'imagenet1k']:
            num_classes = 1000
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        student = ConvNeXtV2.convnextv2(model_size, num_classes=num_classes, drop_path_rate=drop_path_rate, scale_op=scale_op)
        teacher = make_teacher_for_dataset(
            size=model_size,
            dataset=dataset_name,
            num_classes=num_classes,
            device="cpu",
            pretrained=teacher_pretrained
        )

        super().__init__(
            lr, wd, epochs, label_smoothing,
            alpha_kd, alpha_hint, T,
            amp,
            export_dir,
            dataset_name=dataset_name,
            model_name='convnextv2',
            model_size=model_size,
            hint_points=["stages.0", "stages.1", "stages.2", "stages.3"],
            student=student,
            teacher=teacher,
            num_classes=num_classes
        )

# ----------------------------
# CLI / main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p = add_common_args(p)

    # Defaults tuned for small images; adjust as needed from CLI
    p.add_argument("--dataset", type=str, default="timnet",
                   choices=["c10", "cifar10", "c100", "cifar100", "timnet", "tiny",
                            "tinyimagenet", "tiny-imagenet", "imnet", "imagenet", "in1k", "imagenet1k"],
                   help="Target dataset (affects datamodule, num_classes, transforms).")
    p.add_argument("--model-size", type=str, default="nano",
                   choices=["atto", "femto", "pico", "nano", "tiny", "base", "large", "huge"])
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--teacher-pretrained", type=lambda x: str(x).lower() in ["1","true","yes","y"], default=True,
                   help="Use ImageNet-pretrained teacher backbone when classes != 1000 (head is replaced).")

    p.set_defaults(out=None,batch_size=512,lr=0.2,alpha_kd=0.0,alpha_hint=0.1)
    
    args = p.parse_args()
    
    if args.out is None:
        args.out = f"./ckpt_{args.dataset}_convnextv2_{args.model_size}"

    return args

def _pick_datamodule(dataset_name: str, dmargs: dict):
    ds = dataset_name.lower()
    # Prefer datamodules provided by your common_utils; fallback if missing
    if ds in ['c100', 'cifar100']:
        if 'CIFAR100DataModule' in globals():
            return CIFAR100DataModule(**dmargs)
        else:
            raise RuntimeError("CIFAR100DataModule not found in common_utils.")
    elif ds in ['timnet', 'tiny', 'tinyimagenet', 'tiny-imagenet']:
        if 'TinyImageNetDataModule' in globals():
            return TinyImageNetDataModule(**dmargs)
        else:
            raise RuntimeError("TinyImageNetDataModule not found in common_utils.")
    elif ds in ['imnet', 'imagenet', 'in1k', 'imagenet1k']:
        if 'ImageNetDataModule' in globals():
            return ImageNetDataModule(**dmargs)
        else:
            raise RuntimeError("ImageNetDataModule not found in common_utils.")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def main():
    args = parse_args()

    # Derive num_classes to bake into export dir for clarity
    ds = args.dataset.lower()
    if ds in ['c10', 'cifar10']:
        ncls = 10
    elif ds in ['c100', 'cifar100']:
        ncls = 100
    elif ds in ['timnet', 'tiny', 'tinyimagenet', 'tiny-imagenet']:
        ncls = 200
    elif ds in ['imnet', 'imagenet', 'in1k', 'imagenet1k']:
        ncls = 1000
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    out_dir = f"{args.out}_{ds}_{args.model_size}_{ncls}c"

    lit = LitConvNeXtV2KD(
        lr=args.lr, wd=args.wd, epochs=args.epochs,
        dataset_name=args.dataset,
        model_size=args.model_size,
        label_smoothing=args.label_smoothing,
        alpha_kd=args.alpha_kd, alpha_hint=args.alpha_hint, T=args.T,
        amp=args.amp, export_dir=out_dir, drop_path_rate=args.drop_path,
        teacher_pretrained=args.teacher_pretrained
    )

    dmargs = dict(
        data_dir=args.data,
        batch_size=args.batch_size,
        num_workers=4,
        aug_mixup=args.mixup,
        aug_cutmix=args.cutmix,
        alpha=args.mix_alpha
    )
    dm = _pick_datamodule(args.dataset, dmargs)

    trainer, dm = setup_trainer(args, lit, dm)
    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)

if __name__ == "__main__":
    main()
