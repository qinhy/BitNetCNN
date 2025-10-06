import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath
from common_utils import *

EPS = 1e-12

# ----------------------------
# Utilities
# ----------------------------
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
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
    """ GRN (Global Response Normalization) layer
    """
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
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.,scale_op="median"):
        super().__init__()
        self.dwconv = Bit.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise co, scale_op=scale_opnv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = Bit.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear laye, scale_op=scale_oprs
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = Bit.Linear(4 * dim, dim, scale_op=scale_op)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.,scale_op="median"
                 ):
        super().__init__()
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        self.head_init_scale = head_init_scale

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
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

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dim=dims[i], drop_path=dp_rates[cur + j], scale_op=scale_op) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = Bit.Linear(dims[-1], num_classes, scale_op=scale_op)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    def clone(self):
        return ConvNeXtV2(self.in_chans, self.num_classes, self.depths, self.dims, self.drop_path_rate, self.head_init_scale)
    @staticmethod
    def convnextv2(size='atto',num_classes=100, drop_path_rate=0.3, scale_op="median"):
        params = {            
            "atto":dict(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320]),
            "femt":dict(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384]),
            "pico":dict(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512]),
            "nano":dict(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640]),
            "tiny":dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]),
            "base":dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
            "larg":dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),
            "huge":dict(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816])
        }
        return ConvNeXtV2(**params[size],num_classes=num_classes,drop_path_rate=drop_path_rate,scale_op=scale_op)

# ----------------------------
# Teacher: convnextv2 from HF
# ----------------------------
def make_convnextv2_from_timm(size='pico',device="cuda", pretrained=True):
    import timm
    m = timm.create_model(f'convnextv2_{size}.fcmae_ft_in1k', pretrained=pretrained)
    return m.eval().to(device)

# ----------------------------
# LightningModule: KD + hints
# ----------------------------
class LitConvNeXtV2KD(LitBit):
    def __init__(self, lr, wd, epochs, model_size="convnextv2_pico",
                 label_smoothing=0.1, alpha_kd=0.0, alpha_hint=0.0, T=4.0,
                 amp=True, export_dir="./checkpoints_c100_convnextv2", drop_path_rate=0.1):
        super().__init__(lr, wd, epochs, label_smoothing,
        alpha_kd, alpha_hint, T,
        amp,
        export_dir,
        dataset_name='c100',
        model_name='convnextv2',
        model_size=model_size,
        hint_points=["stages.0", "stages.1", "stages.2", "stages.3"],
        student=ConvNeXtV2.convnextv2(model_size,num_classes=100, drop_path_rate=drop_path_rate),
        teacher=make_convnextv2_from_timm(model_size,device="cpu"),)

# ----------------------------
# CLI / main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p = add_common_args(p)
    p.set_defaults(out="./ckpt_c100_convnextv2", batch_size=1024, lr=0.2, alpha_kd=0.0, alpha_hint=0.1)
    p.add_argument("--model-size", type=str, default="nano", choices=["atto","femto","pico","nano","tiny","base","large","huge"])
    p.add_argument("--drop-path", type=float, default=0.1)
    return p.parse_args()

def main():
    args = parse_args()

    lit = LitConvNeXtV2KD(
        lr=args.lr, wd=args.wd, epochs=args.epochs,
        model_size=args.model_size,
        label_smoothing=args.label_smoothing,
        alpha_kd=args.alpha_kd, alpha_hint=args.alpha_hint, T=args.T,
        amp=args.amp, export_dir=args.out, drop_path_rate=args.drop_path
    )

    trainer, dm = setup_trainer(args, lit)
    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)

if __name__ == "__main__":
    main()
