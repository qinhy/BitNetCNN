import copy
from typing import Literal, Optional
import warnings

from pydantic import BaseModel, Field
from pydanticV2_argparse import ArgumentParser
import torch
import torch.nn as nn
from bitlayers.attn2d import Attention2dModels
from bitlayers.bit import Bit
from bitlayers.convs import Conv2dModels
from bitlayers.norms import NormModels
from bitlayers.uir import UniversalInvertedResidual
from common_utils import IN1kToTiny200Adapter, load_tiny200_to_in1k_map, summ
from dataset import DataModuleConfig
from trainer import AccelTrainer, CommonTrainConfig, LitBit, LitBitConfig

torch.set_float32_matmul_precision("high")

Conv2dNorm = Conv2dModels.Conv2dNorm
Conv2dNormAct = Conv2dModels.Conv2dNormAct
InvertedResidual = Conv2dModels.InvertedResidual
UniversalInvertedResidual = UniversalInvertedResidual

def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
    ) -> int:
    """
    This function is copied from here 
    "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"
    
    This is to ensure that all layers have channels that are divisible by 8.

    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.

    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)
    
class MobileNetV4Head(nn.Module):
    """
    Flexible head for MobileNet-V4.
    - pool: 'avg' (default), 'max', or 'avgmax' (concat avg & max)
    - use_bn: add BatchNorm1d before linear
    - act: 'relu' | 'gelu' | 'hswish' | None
    """
    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        pool: Literal['avg','max','avgmax'] = 'avg',
        dropout: float = 0.0,
        use_bn: bool = False,
        act: Optional[str] = None
    ):
        super().__init__()
        self.pool_kind = pool
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
            out_ch = in_ch
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
            out_ch = in_ch
        elif pool == 'avgmax':
            self.pool_avg = nn.AdaptiveAvgPool2d(1)
            self.pool_max = nn.AdaptiveMaxPool2d(1)
            self.pool = None
            out_ch = in_ch * 2
        else:
            raise ValueError(f"Unknown pool: {pool}")

        self.flatten = nn.Flatten(1)

        self.bn = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()

        if act is None:
            self.act = nn.Identity()
        elif act.lower() == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        elif act.lower() == 'hswish':
            self.act = nn.Hardswish()
        else:
            raise ValueError(f"Unsupported act: {act}")

        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = Bit.Linear(out_ch, num_classes)

    @torch.no_grad()
    def feature_dim(self) -> int:
        return self.fc.in_features

    def forward(self, x, return_features: bool = False):
        # x: (N, C, H, W) features from backbone
        if self.pool_kind == 'avgmax':
            xa = self.pool_avg(x)
            xm = self.pool_max(x)
            x = torch.cat([xa, xm], dim=1)
        else:
            x = self.pool(x)
        x = self.flatten(x)      # (N, F)
        f = self.bn(x)
        f = self.act(f)
        f = self.drop(f)
        logits = self.fc(f)
        return (logits, f) if return_features else logits


def norm():return NormModels.BatchNorm2d(num_features=-1)
    
POOL = lambda:nn.AdaptiveAvgPool2d(1)#AdaptiveAvgPool2d(output_size=1)

def convbn(in_c, out_c, k, s):
    return Conv2dNorm(in_channels=in_c,out_channels=out_c,kernel_size=k,stride=s,norm=norm()).build()

def fused_ib(in_c, out_c, stride, expand, act=False, se=None):
    # act always false
    # 5-field (…act) or 6-field (…act, se)
    # 'inp', 'oup', 'stride', 'expand_ratio', 'act'
    # in_c, out_c, stride, expand, act=False, se=None
    # return [in_c, out_c, stride, expand, act] if se is None \
    #     else [in_c, out_c, stride, expand, act, se]
    return InvertedResidual(
        in_channels=in_c,
        out_channels=out_c,
        stride=stride,
        exp_ratio=expand,
        se_layer=Conv2dModels.SqueezeExcite(in_channels=-1) if se else None
    ).build()

def MHSA(num_heads, key_dim, value_dim, px):
    kv_strides = 2 if px == 24 else 1 if px == 12 else 1
    # [heads, kdim, vdim, q_h_s, q_w_s, kv_s, use_layer_scale, use_multi_query, use_residual]
    # return [num_heads, key_dim, value_dim, 1, 1, kv_strides, True, True, True]
    return Attention2dModels.MobileAttention(
                                            in_channels=-1,
                                            out_channels=-1,
                                            stride=1,
                                            num_heads=num_heads,
                                            key_dim=key_dim,
                                            value_dim=value_dim,
                                            use_multi_query=True,
                                            query_strides=(1, 1),
                                            kv_stride=kv_strides,
                                            attn_drop=0.0,
                                            proj_drop=0.0,
                                            layer_scale_init_value=1e-5,
                                            noskip=False,
                                            use_cpe=False,
                                            fused_attn=False,       
                                        )

def uib(in_c, out_c, k1, k2, se, stride, e, shortcut=False, mhsa:Attention2dModels.MobileAttention=None):
    # ['inp', 'oup', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride', 'expand_ratio', 'use_layer_scale', 'mhsa']
    base = [in_c, out_c, k1, k2, se, stride, e, shortcut]
    base = UniversalInvertedResidual(in_channels=in_c,
                                        out_channels=out_c,
                                        dw_kernel_size_start=k1,
                                        dw_kernel_size_mid=k2,                                         
                                    #  se_layer=Conv2dModels.SqueezeExcite(),
                                        stride=stride,
                                        exp_ratio=e,
                                        shortcut=shortcut,
                                    ).build()
    if mhsa is not None:
        mhsa.in_channels = out_c
        mhsa.out_channels = out_c
    return base if mhsa is None else nn.Sequential(base,mhsa.build())

def stage(block_name, *blocks):
    flat = []
    for b in blocks:
        if isinstance(b, list) and b and isinstance(b[0], list):
            flat.extend(b)     # already a list of specs
        else:
            flat.append(b)     # single spec
    # return {"block_name": block_name, "num_blocks": len(flat), "block_specs": flat}
    return nn.Sequential(*flat)

def repeat(n, spec:nn.Module):
    # Deep-ish copy not required for immutable atoms; list() to avoid alias surprises.
    # return [list(spec) for _ in range(n)]
    return nn.Sequential(*[copy.deepcopy(spec) for _ in range(n)])

# ---- Specs (DRY) ------------------------------------------------------------

MNV4ConvSmall = lambda:{
    "conv0":  stage("convbn", convbn(3, 32, 3, 2)),
    "layer1": stage("convbn",
                    convbn(32, 32, 3, 2),
                    convbn(32, 32, 1, 1)),
    "layer2": stage("convbn",
                    convbn(32, 96, 3, 2),
                    convbn(96,  64, 1, 1)),
    "layer3": stage("uib",
                    uib(64, 96, 5, 5, True, 2, 3, False),
                    repeat(4, uib(96, 96, 0, 3, True, 1, 2, False)),
                    uib(96, 96, 3, 0, True, 1, 4, False)),
    "layer4": stage("uib",
                    uib(96,  128, 3, 3, True, 2, 6, False),
                    uib(128, 128, 5, 5, True, 1, 4, False),
                    uib(128, 128, 0, 5, True, 1, 4, False),
                    uib(128, 128, 0, 5, True, 1, 3, False),
                    repeat(2, uib(128, 128, 0, 3, True, 1, 4, False))),
    "layer5": stage("convbn",
                    convbn(128, 960, 1, 1),
                    POOL(),
                    convbn(960, 1280, 1, 1)),
}

MNV4ConvMedium = lambda:{
    "conv0":  stage("convbn", convbn(3, 32, 3, 2)),
    "layer1": stage("fused_ib", fused_ib(32, 48, 2, 4.0, False)),
    "layer2": stage("uib",
                    uib(48, 80, 3, 5, True, 2, 4, False),
                    uib(80, 80, 3, 3, True, 1, 2, False)),
    "layer3": stage("uib",
                    uib(80, 160, 3, 5, True, 2, 6, False),
                    repeat(2, uib(160, 160, 3, 3, True, 1, 4, False)),
                    uib(160, 160, 3, 5, True, 1, 4, False),
                    uib(160, 160, 3, 3, True, 1, 4, False),
                    uib(160, 160, 3, 0, True, 1, 4, False),
                    uib(160, 160, 0, 0, True, 1, 2, False),
                    uib(160, 160, 3, 0, True, 1, 4, False)),
    "layer4": stage("uib",
                    uib(160, 256, 5, 5, True, 2, 6, False),
                    uib(256, 256, 5, 5, True, 1, 4, False),
                    repeat(2, uib(256, 256, 3, 5, True, 1, 4, False)),
                    uib(256, 256, 0, 0, True, 1, 4, False),
                    uib(256, 256, 3, 0, True, 1, 4, False),
                    uib(256, 256, 3, 5, True, 1, 2, False),
                    uib(256, 256, 5, 5, True, 1, 4, False),
                    repeat(2, uib(256, 256, 0, 0, True, 1, 4, False)),
                    uib(256, 256, 5, 0, True, 1, 2, False)),
    "layer5": stage("convbn",
                    convbn(256, 960, 1, 1),
                    POOL(),
                    convbn(960, 1280, 1, 1)),
}

MNV4ConvLarge = lambda:{
    "conv0":  stage("convbn", convbn(3, 24, 3, 2)),
    "layer1": stage("fused_ib", fused_ib(24, 48, 2, 4.0, False)),
    "layer2": stage("uib",   # FIX: ensure 8 fields (…shortcut=False)
                    uib(48, 96, 3, 5, True, 2, 4, False),
                    uib(96, 96, 3, 3, True, 1, 4, False)),
    "layer3": stage("uib",
                    uib(96, 192, 3, 5, True, 2, 4, False),
                    repeat(3, uib(192, 192, 3, 3, True, 1, 4, False)),
                    uib(192, 192, 3, 5, True, 1, 4, False),
                    repeat(5, uib(192, 192, 5, 3, True, 1, 4, False)),
                    uib(192, 192, 3, 0, True, 1, 4, False)),
    "layer4": stage("uib",
                    uib(192, 512, 5, 5, True, 2, 4, False),
                    repeat(3, uib(512, 512, 5, 5, True, 1, 4, False)),
                    uib(512, 512, 5, 0, True, 1, 4, False),
                    uib(512, 512, 5, 3, True, 1, 4, False),
                    repeat(2, uib(512, 512, 5, 0, True, 1, 4, False)),
                    uib(512, 512, 5, 3, True, 1, 4, False),
                    uib(512, 512, 5, 5, True, 1, 4, False),
                    repeat(3, uib(512, 512, 5, 0, True, 1, 4, False))),
    "layer5": stage("convbn",
                    convbn(512, 960, 1, 1),
                    POOL(),
                    convbn(960, 1280, 1, 1)),
}

MNV4HybridConvMedium = lambda:{
    "conv0":  stage("convbn", convbn(3, 32, 3, 2)),
    "layer1": stage("fused_ib", fused_ib(32, 48, 2, 4.0, False)),
    "layer2": stage("uib",
                    uib(48, 80, 3, 5, True, 2, 4, True),
                    uib(80, 80, 3, 3, True, 1, 2, True)),
    "layer3": stage("uib",
                    uib(80,  160, 3, 5, True, 2, 6, True),
                    uib(160, 160, 0, 0, True, 1, 2, True),
                    uib(160, 160, 3, 3, True, 1, 4, True),
                    uib(160, 160, 3, 5, True, 1, 4, True, MHSA(4, 64, 64, 24)),
                    uib(160, 160, 3, 3, True, 1, 4, True, MHSA(4, 64, 64, 24)),
                    uib(160, 160, 3, 0, True, 1, 4, True, MHSA(4, 64, 64, 24)),
                    uib(160, 160, 3, 3, True, 1, 4, True, MHSA(4, 64, 64, 24)),
                    uib(160, 160, 3, 0, True, 1, 4, True)),
    "layer4": stage("uib",
                    uib(160, 256, 5, 5, True, 2, 6, True),
                    uib(256, 256, 5, 5, True, 1, 4, True),
                    uib(256, 256, 3, 5, True, 1, 4, True),
                    uib(256, 256, 3, 5, True, 1, 4, True),
                    uib(256, 256, 0, 0, True, 1, 2, True),
                    uib(256, 256, 3, 5, True, 1, 2, True),
                    uib(256, 256, 0, 0, True, 1, 2, True),
                    uib(256, 256, 0, 0, True, 1, 4, True, MHSA(4, 64, 64, 12)),
                    uib(256, 256, 3, 0, True, 1, 4, True, MHSA(4, 64, 64, 12)),
                    uib(256, 256, 5, 5, True, 1, 4, True, MHSA(4, 64, 64, 12)),
                    uib(256, 256, 5, 0, True, 1, 4, True, MHSA(4, 64, 64, 12)),
                    uib(256, 256, 5, 0, True, 1, 4, True)),
    "layer5": stage("convbn",
                    convbn(256, 960, 1, 1),
                    POOL(),
                    convbn(960, 1280, 1, 1)),
}

MNV4HybridConvLarge = lambda:{
    "conv0":  stage("convbn", convbn(3, 24, 3, 2)),
    "layer1": stage("fused_ib", fused_ib(24, 48, 2, 4.0, False, True)),
    "layer2": stage("uib",
                    uib(48, 96, 3, 5, True, 2, 4, True),
                    uib(96, 96, 3, 3, True, 1, 4, True)),
    "layer3": stage("uib",
                    uib(96, 192, 3, 5, True, 2, 4, True),
                    repeat(3, uib(192, 192, 3, 3, True, 1, 4, True)),
                    uib(192, 192, 3, 5, True, 1, 4, True),
                    repeat(2, uib(192, 192, 5, 3, True, 1, 4, True)),
                    repeat(4, uib(192, 192, 5, 3, True, 1, 4, True, MHSA(8, 48, 48, 24))),
                    uib(192, 192, 3, 0, True, 1, 4, True)),
    "layer4": stage("uib",
                    uib(192, 512, 5, 5, True, 2, 4, True),
                    repeat(3, uib(512, 512, 5, 5, True, 1, 4, True)),
                    uib(512, 512, 5, 0, True, 1, 4, True),
                    uib(512, 512, 5, 3, True, 1, 4, True),
                    repeat(2, uib(512, 512, 5, 0, True, 1, 4, True)),
                    uib(512, 512, 5, 3, True, 1, 4, True),
                    uib(512, 512, 5, 5, True, 1, 4, True, MHSA(8, 64, 64, 12)),
                    repeat(3, uib(512, 512, 5, 0, True, 1, 4, True, MHSA(8, 64, 64, 12))),
                    uib(512, 512, 5, 0, True, 1, 4, True)),
    "layer5": stage("convbn",
                    convbn(512, 960, 1, 1),
                    POOL(),
                    convbn(960, 1280, 1, 1)),
}

class MobileNetV4(nn.Module):
    MODEL_SPECS = {
        "MobileNetV4ConvSmall": MNV4ConvSmall,
        "MobileNetV4ConvMedium": MNV4ConvMedium,
        "MobileNetV4ConvLarge": MNV4ConvLarge,
        "MobileNetV4HybridMedium": MNV4HybridConvMedium,
        "MobileNetV4HybridLarge": MNV4HybridConvLarge
    }

    def __init__(self, model_name, num_classes):
        # MobileNetV4ConvSmall  MobileNetV4ConvMedium  MobileNetV4ConvLarge
        # MobileNetV4HybridMedium  MobileNetV4HybridLarge
        """Params to initiate MobilenNetV4
        Args:
            model : support 5 types of models as indicated in 
            "https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py"        
        """
        super().__init__()
        model_name = {
            "MobileNetV4ConvSmall":"MobileNetV4ConvSmall",
            "small":"MobileNetV4ConvSmall",
            "MobileNetV4ConvMedium":"MobileNetV4ConvMedium",
            "medium":"MobileNetV4ConvMedium",
            "MobileNetV4ConvLarge":"MobileNetV4ConvLarge",
            "large":"MobileNetV4ConvLarge",
            "MobileNetV4HybridMedium":"MobileNetV4HybridMedium",
            "hybrid_medium":"MobileNetV4HybridMedium",
            "MobileNetV4HybridLarge":"MobileNetV4HybridLarge",
            "hybrid_large":"MobileNetV4HybridLarge",
        }[model_name]
        assert model_name in MobileNetV4.MODEL_SPECS.keys()
        self.model_name = model_name
        self.num_classes = num_classes
        self.spec = MobileNetV4.MODEL_SPECS[self.model_name]()
       
        # conv0
        self.conv0 = self.spec['conv0']
        # layer1
        self.layer1 = self.spec['layer1']
        # layer2
        self.layer2 = self.spec['layer2']
        # layer3
        self.layer3 = self.spec['layer3']
        # layer4
        self.layer4 = self.spec['layer4']
        # layer5   
        self.layer5 = self.spec['layer5']

        # print("Check output shape ...")
        x = torch.rand(2, 3, 224, 224)
        y = self.feature(x)[-1]
        self.head = MobileNetV4Head(y.shape[1],num_classes=num_classes)
               
    def feature(self, x):
        x0 = self.conv0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return [x1, x2, x3, x4, x5]
        # return [x0, x1, x2, x3, x5]

    def forward(self, x):
        return self.head(self.feature(x)[-1])

    def clone(self) -> "MobileNetV4":
        return self.__class__(
                self.model_name,
                self.num_classes)

# for n in [
#         "small",
#         "medium",
#         "large",
#         "hybrid_medium",
#         "hybrid_large",
#     ]:
#     print(n)
#     model = MobileNetV4(n,200)
#     # Check the trainable params
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Number of parameters: {total_params}")
#     # Check the model's output shape
#     print("Check output shape ...")
#     x = torch.rand(2, 3, 64, 64)
#     y = model.feature(x)
#     for i in y: print(i.shape)

# ----------------------------
# MobileNetV4: builders + KD
# ----------------------------
def _parse_mnv4_tag(model_size: str):
    """
    Accepts: 'small', 'medium', 'large', 'hybrid_medium', 'hybrid_large'
             (and typo alias: 'hybrid_medium')
    Returns: family ('conv'|'hybrid'), size ('small'|'medium'|'large'), arch_tag string
    """
    s = (model_size or "").lower().strip()
    if s.startswith("hybrid_"):
        fam, sz = "hybrid", s.split("_", 1)[1]
    elif s in {"hybrid_medium", "hybrid_med", "hybrid_mid"}:
        fam, sz = "hybrid", "medium"
    else:
        fam, sz = ("hybrid", "medium") if s in {"midum", "med", "mid"} else ("conv", s)

    # normalize canonical sizes
    if sz not in {"small", "medium", "large"}:
        raise ValueError(f"Unsupported MobileNetV4 size '{sz}'. Use small|medium|large or hybrid_* variants.")
    arch_tag = f"mobilenetv4_{fam}_{sz}"
    return fam, sz, arch_tag


# ---------- timm model builders (student + teacher) ----------
def make_mobilenetv4_from_timm(
    model_size: str = "small",
    device: str = "cuda",
    pretrained: bool = True,
    model_name: str | None = None
):
    """
    Build a timm MobileNetV4 model. If model_name is provided, it wins.
    Otherwise we map (conv|hybrid, size) -> a reasonable default HF weight;
    fall back to bare arch when needed.
    """
    import timm
    fam, sz, arch_tag = _parse_mnv4_tag(model_size)

    if model_name is None:
        default_name_map = {
            ("conv", "small"):   "mobilenetv4_conv_small.e1200_r224_in1k",
            ("conv", "medium"):  "mobilenetv4_conv_medium.e500_r256_in1k",
            ("conv", "large"):   "mobilenetv4_conv_large.e600_r384_in1k",
            ("hybrid", "medium"): "mobilenetv4_hybrid_medium.ix_e550_r256_in1k",
            ("hybrid", "large"):  "mobilenetv4_hybrid_large.ix_e600_r384_in1k",
        }
        model_name = default_name_map.get((fam, sz), arch_tag)
        if pretrained and model_name == arch_tag:
            warnings.warn(
                f"No default pretrained weights mapped for '{arch_tag}'. "
                "Building architecture without pretrained weights."
            )

    try:
        print("create teacher model", model_name, pretrained)
        m = timm.create_model(model_name, pretrained=pretrained)
    except Exception:
        # try explicit HF hub path
        try:
            m = timm.create_model(f"hf-hub:timm/{model_name}", pretrained=pretrained)
        except Exception as e:
            if pretrained and "." in model_name:
                warnings.warn(
                    f"Could not load pretrained weights '{model_name}'. "
                    f"Falling back to bare arch '{arch_tag}'. Error: {e}"
                )
            m = timm.create_model(arch_tag, pretrained=False)

    return m.eval().to(device)


def make_mobilenetv4_teacher_for_dataset(
    size: str,
    dataset: str,
    num_classes: int,
    device: str = "cpu",
    pretrained: bool = True,
    model_name: str | None = None,
):
    """
    Teacher = MobileNetV4 with IN1K weights if available. If dataset classes != head,
    we replace the classifier via timm's reset_classifier (or manual).
    """
    t = make_mobilenetv4_from_timm(model_size=size, device=device, pretrained=pretrained, model_name=model_name)

    head_out = getattr(t, "num_classes", None)
    if head_out is None:
        getc = getattr(t, "get_classifier", None)
        if callable(getc):
            cls = getc()
            head_out = getattr(cls, "out_features", None)
    head_out = head_out or 1000

    if head_out != num_classes:
        ds = (dataset or "").lower().strip()
        if ds in {"timnet", "tiny", "tinyimagenet", "tiny-imagenet"} and num_classes == 200:
            mapping = load_tiny200_to_in1k_map("timnet_to_imagenet1k_indices.txt")
            adapter = IN1kToTiny200Adapter(mapping, temperature=2.0, renormalize=True).to(device)

            class Mobilenetv4WithIN1kToTiny200Adapter(nn.Module):
                def __init__(self, backbone: nn.Module, adapter: nn.Module, num_classes: int):
                    super().__init__()
                    self.backbone = backbone
                    self.adapter = adapter
                    self.num_classes = num_classes

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.adapter(self.backbone(x))

            return Mobilenetv4WithIN1kToTiny200Adapter(t, adapter, num_classes=num_classes).eval().to(device)

        # Generic fallback: reset classifier head to match classes (weights are random).
        reset = getattr(t, "reset_classifier", None)
        if callable(reset):
            reset(num_classes=num_classes)
            t.num_classes = num_classes
            return t.eval().to(device)

    return t.eval().to(device)

# ----------------------------
# LightningModule: KD + hints
# ----------------------------
def _maybe_backbone_and_prefix(teacher: nn.Module) -> tuple[nn.Module, str]:
    backbone = getattr(teacher, "backbone", None)
    if isinstance(backbone, nn.Module):
        return backbone, "backbone."
    return teacher, ""


def _infer_mnv4_hint_points(teacher: nn.Module) -> list[tuple[str, str]]:
    teacher_backbone, prefix = _maybe_backbone_and_prefix(teacher)
    tmods = set(dict(teacher_backbone.named_modules()).keys())

    hardcoded = ["blocks.0", "blocks.1", "blocks.2", "blocks.3"]
    if all(m in tmods for m in hardcoded):
        return [(f"layer{i}", f"{prefix}{m}") for i, m in enumerate(hardcoded, start=1)]

    feature_info = getattr(teacher_backbone, "feature_info", None)
    if feature_info is None or not hasattr(feature_info, "get_dicts"):
        return []

    names: list[str] = []
    for d in feature_info.get_dicts():
        m = d.get("module")
        if m and m not in names:
            names.append(str(m))

    names = [n for n in names if n in tmods]
    if len(names) < 2:
        return []

    # Prefer the last 4 feature taps (deepest, lowest-res).
    names = names[-4:]
    student_layers = ["layer1", "layer2", "layer3", "layer4"][-len(names):]
    return list(zip(student_layers, [f"{prefix}{n}" for n in names]))


class LitMobileNetV4KD(LitBit):
    @staticmethod
    def _infer_num_classes(model: nn.Module) -> Optional[int]:
        n = getattr(model, "num_classes", None)
        if isinstance(n, int) and n > 0:
            return n
        getc = getattr(model, "get_classifier", None)
        if callable(getc):
            cls = getc()
            out = getattr(cls, "out_features", None)
            if isinstance(out, int) and out > 0:
                return out
        cls = getattr(model, "classifier", None)
        out = getattr(cls, "out_features", None)
        if isinstance(out, int) and out > 0:
            return out
        return None

    def __init__(
        self,
        config: LitBitConfig,
        model_size: str,
        teacher_pretrained: bool = True,
        teacher_device: str = "cpu",
    ):
        config = LitBitConfig.model_validate(config)
        if config.dataset is None:
            raise ValueError("LitMobileNetV4KD requires config.dataset to be set (DataModuleConfig).")

        config.student = MobileNetV4(model_size, num_classes=config.dataset.num_classes)
        config.teacher = make_mobilenetv4_teacher_for_dataset(
            size=model_size,
            dataset=config.dataset.dataset_name,
            num_classes=config.dataset.num_classes,
            device=teacher_device,
            pretrained=teacher_pretrained,
        )
        t_nc = self._infer_num_classes(config.teacher)
        if t_nc is not None and t_nc != config.dataset.num_classes and config.alpha_kd > 0:
            warnings.warn(
                f"Teacher output classes ({t_nc}) != dataset num_classes ({config.dataset.num_classes}); "
                "disabling KD loss."
            )
            config.alpha_kd = 0.0

        config.hint_points = _infer_mnv4_hint_points(config.teacher)
        if config.alpha_hint > 0 and len(config.hint_points) == 0:
            warnings.warn("Could not infer MobileNetV4 teacher hint points; disabling hint loss.")
            config.alpha_hint = 0.0

        config.model_name = config.model_name or "mnv4"
        config.model_size = config.model_size or model_size

        super().__init__(config)

# ----------------------------
# CLI / main (MobileNetV4)
# ----------------------------
class Config(CommonTrainConfig):
    data: Optional[str] = Field(default=None, description="Alias for --data_dir (back-compat).")
    # For MobileNetV4 we accept conv + hybrid tags in one flag
    model_size: Literal[
        "small",
        "medium",
        "large",
        "hybrid_medium",
        "hybrid_large",
    ] = Field(
        default="small",
        description="MobileNetV4 variant.",
    )

    drop_path: float = Field(
        default=0.0,
        description="Stochastic depth drop-path rate.",
    )

    teacher_pretrained: bool = Field(
        default=True,
        description=(
            "Use ImageNet-pretrained teacher backbone when classes != 1000 "
            "(head is replaced)."
        ),
    )
    
    num_workers: int = 1
    batch_size:int=512
    lr:float=0.01
    alpha_kd:float=0.0
    alpha_hint:float=1e-5

def main_mnv4() -> None:
    parser = ArgumentParser(model=Config)
    args = parser.parse_typed_args()
    if args.data is not None:
        args.data_dir = args.data
    
    dm = DataModuleConfig.model_validate(args.model_dump())
    config = LitBitConfig.model_validate(args.model_dump())
    config.dataset = dm.model_copy()

    if "hybrid" in config.model_size.lower():
        print("[!!!Warning!!!]: hybrid models need image size >= 64x64")

    args.export_dir = config.export_dir = f"./ckpt_{config.dataset.dataset_name}_mnv4_{args.model_size}"
    config.model_name = "mnv4"
    config.model_size = str(args.model_size)
    
    lit = LitMobileNetV4KD(
        config=config,
        model_size=args.model_size,
        teacher_pretrained=args.teacher_pretrained,
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
    main_mnv4()
