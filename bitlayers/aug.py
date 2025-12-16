import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from torchvision.transforms import v2

from typing import Any, Callable, Optional, Sequence, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator

class ToTensor(BaseModel):
    dtype:str = "float32"
    scale:bool=True
    _torch_dtype:Any = None

    @model_validator(mode='after')
    def valid_model(self):
        self._torch_dtype = {
            "float32":torch.float32,
            "float16":torch.float16,
        }[self.dtype]

    def build(self):
        return v2.Compose([v2.ToImage(),
                           v2.ToDtype(self._torch_dtype, scale=self.scale)])

class Cutout(BaseModel):
    size: Union[int, Tuple[int, int]] = 16

    @model_validator(mode='after')
    def valid_model(self):
        size = self.size
        if isinstance(size, int):
            self.size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError(f"Cutout size must be int or (h, w) tuple, got: {size}")
            self.size = (int(size[0]), int(size[1]))
    def build(self):return CutoutModule(self)

class CutoutModule(nn.Module):
    """Simple Cutout transform for tensors [C, H, W].

    size:
        - int       -> square hole (size x size)
        - (h, w)    -> rectangular hole (height x width)
    """
    def __init__(self, config:Cutout):
        super().__init__()
        self.size = config.size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img is expected to be a Tensor after ToImage/ToDtype
        if not torch.is_tensor(img):
            return img

        _, h, w = img.shape
        if h <= 0 or w <= 0:
            return img

        mask_h, mask_w = self.size
        mask_h_half = mask_h // 2
        mask_w_half = mask_w // 2

        cy = torch.randint(0, h, (1,)).item()
        cx = torch.randint(0, w, (1,)).item()

        y1 = max(0, cy - mask_h_half)
        y2 = min(h, cy + mask_h_half)
        x1 = max(0, cx - mask_w_half)
        x2 = min(w, cx + mask_w_half)

        img = img.clone()  # safer than in-place
        img[:, y1:y2, x1:x2] = 0.0
        return img

class SimpleImageTrainAugment(BaseModel):
    # --- normalization ---
    norm_mean: Tuple[float, ...]
    norm_std: Tuple[float, ...]

    # --- crop ---
    crop_output_size: int = 32
    crop_padding_px: int = 4

    # --- horizontal flip ---
    enable_hflip: bool = True

    # --- probabilities (explicit) ---
    randaugment_apply_prob: float = Field(default=0.325, ge=0.0, le=1.0)
    random_erasing_apply_prob: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- RandAugment ---
    enable_randaugment: bool = True
    randaugment_num_ops: int = Field(default=2, ge=1)
    randaugment_magnitude: int = Field(default=7, ge=0)

    # --- RandomErasing ---
    erasing_area_fraction_range: Tuple[float, float] = (0.02, 0.2)
    erasing_aspect_ratio_range: Tuple[float, float] = (0.3, 3.3)
    erasing_fill: str = "random"  # torchvision expects "random" or a numeric value

    # --- dtype / tensor conversion ---
    dtype: str= "float32"

    def build(self) -> v2.Compose:
        ra = v2.RandAugment(
            num_ops=self.randaugment_num_ops,
            magnitude=self.randaugment_magnitude,
        )

        return v2.Compose([
            v2.RandomCrop(self.crop_output_size, padding=self.crop_padding_px),
            v2.RandomHorizontalFlip() if self.enable_hflip else v2.Identity(),

            v2.RandomApply([ra], p=self.randaugment_apply_prob)
            if self.enable_randaugment else v2.Identity(),

            ToTensor(dtype=self.dtype, scale=True).build(),

            # single probability here (no wrapper) ✅
            v2.RandomErasing(
                p=self.random_erasing_apply_prob,
                scale=self.erasing_area_fraction_range,
                ratio=self.erasing_aspect_ratio_range,
                value=self.erasing_fill,
            ),

            v2.Normalize(self.norm_mean, self.norm_std),
        ])

class SimpleImageValAugment(BaseModel):
    norm_mean: Tuple[float, ...]
    norm_std: Tuple[float, ...]

    dtype: str= "float32"

    def build(self) -> v2.Compose:
        return v2.Compose([
            ToTensor(self.dtype, scale=True).build(),
            v2.Normalize(self.norm_mean, self.norm_std),
        ])

class MixupCutmix(BaseModel):
    num_classes: int = Field(ge=2)

    enable_mixup: bool = False
    enable_cutmix: bool = False

    # Shared alpha (you can split later if you want)
    beta_alpha: float = Field(default=1.0, gt=0.0)
    apply_prob: float = Field(default=0.325, ge=0.0, le=1.0)


    def build(self) -> Optional[v2.Transform]:
        candidates = []
        if self.enable_mixup:
            candidates.append(v2.MixUp(num_classes=self.num_classes, alpha=self.beta_alpha))
        if self.enable_cutmix:
            candidates.append(v2.CutMix(num_classes=self.num_classes, alpha=self.beta_alpha))

        if not candidates:
            return v2.Identity()

        chosen = candidates[0] if len(candidates) == 1 else v2.RandomChoice(candidates)
        return v2.RandomApply([chosen], p=self.apply_prob)

class CIFAR100Augment(BaseModel):
    # --- normalization ---
    norm_mean: Tuple[float, ...] = (0.5071, 0.4867, 0.4408)
    norm_std: Tuple[float, ...] = (0.2675, 0.2565, 0.2761)
    # --- crop ---
    crop_output_size: int = 32
    crop_padding_px: int = 4
    def build(self):
        return (
            SimpleImageTrainAugment.model_validate(self.model_dump()).build(),
            SimpleImageValAugment.model_validate(self.model_dump()).build(),
            MixupCutmix(num_classes=100,enable_cutmix=True,enable_mixup=True).build(),
        )


# Helpers: 1/f fractal + blur + rotate
# -----------------------------
def generate_fractal_1_over_f(batch, c, h, w, beta=2.0, device=None, dtype=torch.float32):
    """
    Returns [B,C,H,W] in [0,1] (float32 by default).
    Uses rfft/irfft so the result is properly real-valued.
    """
    device = device or "cpu"
    fy = torch.fft.fftfreq(h, device=device, dtype=torch.float32).view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=device, dtype=torch.float32).view(1, -1)
    f = torch.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    amp = 1.0 / (f ** (beta / 2.0))  # [H, W//2+1]

    real = torch.randn(batch, c, h, w // 2 + 1, device=device, dtype=torch.float32)
    imag = torch.randn(batch, c, h, w // 2 + 1, device=device, dtype=torch.float32)
    spec = torch.complex(real, imag) * amp  # broadcast

    img = torch.fft.irfft2(spec, s=(h, w))  # [B,C,H,W], real
    img = img - img.amin(dim=(2, 3), keepdim=True)
    img = img / (img.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return img.to(dtype=dtype).clamp(0, 1)


def rotate_tensor(x, degrees):
    """
    Rotate a tensor [N,C,H,W] by degrees around center using grid_sample.
    """
    n, c, h, w = x.shape
    device, dtype = x.device, x.dtype
    angle = torch.tensor(degrees, device=device, dtype=dtype) * (math.pi / 180.0)
    ca, sa = torch.cos(angle), torch.sin(angle)

    theta = torch.zeros((n, 2, 3), device=device, dtype=dtype)
    theta[:, 0, 0] = ca
    theta[:, 0, 1] = -sa
    theta[:, 1, 0] = sa
    theta[:, 1, 1] = ca

    grid = F.affine_grid(theta, size=x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def normalize_map(m):
    m = m - m.amin(dim=(2, 3), keepdim=True)
    m = m / (m.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return m


# -----------------------------
# Integrated S²-FracMix module
# -----------------------------
class S2FracMix(BaseModel):
    scales: Tuple[float, ...] = Field(default=(0.2, 0.35, 0.5), min_length=1)
    theta_max: float = Field(default=30.0, ge=0.0)
    saliency_thresh: float = Field(default=0.5, ge=0.0, le=1.0)

    fractal_lambda: float = Field(default=0.2, ge=0.0, le=1.0)
    fractal_beta: float = Field(default=2.0, gt=0.0)

    blur_kernel: int = Field(default=7, ge=1)
    blur_sigma: Tuple[float, float] = Field(default=(0.1, 2.0))

    s2_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    apply_in_eval: bool = Field(default=False)
    return_saliency: bool = Field(default=False)

    model_input_fn: Optional[Callable] = Field(default=None, exclude=True)

    def build(self):return S2FracMixModule(self)    

class S2FracMixModule(nn.Module):
    """
    s2 = S2FracMix(
        s2_prob=0.5,
        model_input_fn=None,      # or your normalize() wrapper
        return_saliency=False
    ).cuda()

    model.train()
    images_aug, y_soft, used, _ = s2(model, images, labels)  # labels can be [B] ints

    logits = model(images_aug)
    loss = -(F.log_softmax(logits, dim=1) * y_soft).sum(dim=1).mean()

    Integrated augmenter:
      forward(model, images, labels_or_onehot) -> (images_aug, labels_onehot, used, saliency)

    images: [B,C,H,W] assumed in [0,1] (or at least bounded)
    labels_or_onehot: [B] int OR [B,K] float onehot/soft
    """
    def __init__(self,config:S2FracMix):
        super().__init__()
        self.scales = tuple(config.scales)
        self.theta_max = float(config.theta_max)
        self.t = float(config.saliency_thresh)
        self.lam_f = float(config.fractal_lambda)
        self.beta = float(config.fractal_beta)
        self.blur = GaussianBlur(kernel_size=config.blur_kernel, sigma=config.blur_sigma)
        self.s2_prob = float(config.s2_prob)
        self.apply_in_eval = bool(config.apply_in_eval)
        self.model_input_fn = config.model_input_fn
        self.return_saliency = bool(config.return_saliency)

    def _sample_salient_topleft(self, mask, ph, pw):
        # mask: [H,W] bool
        H, W = mask.shape
        ys, xs = torch.where(mask)
        if ys.numel() == 0:
            y = torch.randint(0, max(1, H - ph + 1), (1,), device=mask.device).item()
            x = torch.randint(0, max(1, W - pw + 1), (1,), device=mask.device).item()
            return y, x

        idx = torch.randint(0, ys.numel(), (1,), device=mask.device).item()
        cy, cx = ys[idx].item(), xs[idx].item()
        y = int(max(0, min(H - ph, cy - ph // 2)))
        x = int(max(0, min(W - pw, cx - pw // 2)))
        return y, x

    def _labels_to_onehot(self, labels_or_onehot, K, device, dtype):
        if labels_or_onehot.dim() == 1:
            return F.one_hot(labels_or_onehot.to(device=device), num_classes=K).to(dtype=dtype)
        if labels_or_onehot.dim() == 2:
            return labels_or_onehot.to(device=device, dtype=dtype)
        raise ValueError("labels_or_onehot must be [B] or [B,K]")

    def _compute_saliency(self, model, images, labels_onehot):
        """
        One backward wrt input only. Runs model in eval mode and disables param grads temporarily.
        """
        was_training = model.training
        req = [p.requires_grad for p in model.parameters()]
        try:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

            with torch.enable_grad(), torch.cuda.amp.autocast(enabled=False):
                x = images.detach().to(dtype=torch.float32).clone().requires_grad_(True)

                x_in = self.model_input_fn(x) if self.model_input_fn is not None else x
                logits = model(x_in)  # [B,K]

                # make labels match logits dtype
                y = labels_onehot.to(device=logits.device, dtype=logits.dtype)
                score = (logits * y).sum()

                grads = torch.autograd.grad(score, x, create_graph=False, retain_graph=False)[0]
                sal = grads.abs().sum(dim=1, keepdim=True)  # [B,1,H,W]
                sal = normalize_map(sal)
                return sal.detach(), logits.detach()
        finally:
            for p, r in zip(model.parameters(), req):
                p.requires_grad_(r)
            model.train(was_training)

    @torch.no_grad()
    def _augment_with_saliency(self, images, saliency):
        B, C, H, W = images.shape
        out = []
        n_scales = len(self.scales)

        for i in range(B):
            I = images[i:i+1]     # [1,C,H,W]
            S = saliency[i:i+1]   # [1,1,H,W]
            mask = (S[0, 0] >= self.t)

            Pm = torch.zeros_like(I)

            for s in self.scales:
                ph = max(1, int(math.floor(s * H)))
                pw = max(1, int(math.floor(s * W)))
                y0, x0 = self._sample_salient_topleft(mask, ph, pw)

                Pk = I[:, :, y0:y0+ph, x0:x0+pw]  # [1,C,ph,pw]
                Sk = S[:, :, y0:y0+ph, x0:x0+pw]  # [1,1,ph,pw]

                # FracMix inside patch
                Fk = generate_fractal_1_over_f(1, C, ph, pw, beta=self.beta,
                                               device=Pk.device, dtype=Pk.dtype)
                Pk_mix = self.lam_f * Fk + (1.0 - self.lam_f) * Pk

                # Transform: rotate + blur gated by saliency
                theta = (torch.rand((), device=Pk.device).item() * 2.0 - 1.0) * self.theta_max
                Pk_rot = rotate_tensor(Pk_mix, theta)
                Pk_blr = self.blur(Pk_mix)

                Tk = Pk_rot * (1.0 - Sk) + Pk_blr * Sk

                # Resize to full and accumulate
                Rk = F.interpolate(Tk, size=(H, W), mode="bilinear", align_corners=False)
                Pm = Pm + (1.0 / n_scales) * Rk

            alpha = torch.rand((), device=I.device).item()
            I_tilde = alpha * I + (1.0 - alpha) * Pm
            out.append(I_tilde)

        return torch.cat(out, dim=0).clamp(0, 1)

    def forward(self, model, images, labels_or_onehot):
        """
        Returns:
          images_out: [B,C,H,W]
          labels_onehot: [B,K]
          used: bool
          saliency (optional): [B,1,H,W] or None
        """
        if (not self.training) and (not self.apply_in_eval):
            # still convert labels to onehot for downstream
            with torch.no_grad():
                # infer K by a cheap forward if needed
                logits = model(self.model_input_fn(images) if self.model_input_fn else images)
                K = logits.shape[1]
                y = self._labels_to_onehot(labels_or_onehot, K, images.device, logits.dtype)
            return images, y, False, (None if not self.return_saliency else None)

        # decide apply
        if torch.rand((), device=images.device) >= self.s2_prob:
            with torch.no_grad():
                logits = model(self.model_input_fn(images) if self.model_input_fn else images)
                K = logits.shape[1]
                y = self._labels_to_onehot(labels_or_onehot, K, images.device, logits.dtype)
            return images, y, False, (None if not self.return_saliency else None)

        # run saliency and augment
        # infer K from model output during saliency pass
        with torch.no_grad():
            # (only to get K/dtype if labels are ints; keeps behavior consistent)
            tmp_logits = model(self.model_input_fn(images) if self.model_input_fn else images)
            K = tmp_logits.shape[1]
            logits_dtype = tmp_logits.dtype

        labels_onehot = self._labels_to_onehot(labels_or_onehot, K, images.device, logits_dtype)
        saliency, _ = self._compute_saliency(model, images, labels_onehot)
        images_aug = self._augment_with_saliency(images, saliency)

        return images_aug, labels_onehot, True, (saliency if self.return_saliency else None)
