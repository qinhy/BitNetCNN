import math
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
from torchvision.transforms import GaussianBlur

# ---- 1) Simple procedural fractal-ish generator (fast 1/f noise via FFT) ----
def generate_fractal_1_over_f(batch, c, h, w, beta=2.0, device="cuda"):
    # Returns [B,C,H,W] in [0,1]
    # Build frequency grid
    fy = torch.fft.fftfreq(h, device=device).view(-1, 1)
    fx = torch.fft.fftfreq(w, device=device).view(1, -1)
    f = torch.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0  # avoid div by zero
    amp = 1.0 / (f ** (beta / 2.0))

    # Random complex spectrum
    real = torch.randn(batch, c, h, w, device=device)
    imag = torch.randn(batch, c, h, w, device=device)
    spec = torch.complex(real, imag)

    # Shape amplitude and invert
    shaped = spec * amp
    img = torch.fft.ifft2(shaped).real

    # Normalize per-sample to [0,1]
    img = img - img.amin(dim=(2,3), keepdim=True)
    img = img / (img.amax(dim=(2,3), keepdim=True) + 1e-6)
    return img.clamp(0, 1)

# ---- 2) Gradient saliency (one backward wrt input only) ----
@torch.no_grad()
def _normalize_map(m):
    m = m - m.amin(dim=(2,3), keepdim=True)
    m = m / (m.amax(dim=(2,3), keepdim=True) + 1e-6)
    return m

def gradient_saliency_map(model, images, labels_onehot):
    """
    images: [B,C,H,W] float (requires_grad will be set)
    labels_onehot: [B,num_classes] float (one-hot)
    returns: saliency [B,1,H,W] in [0,1]
    """
    # freeze model params for this saliency pass (reduces memory)
    req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    x = images.detach().clone().requires_grad_(True)
    logits = model(x)  # [B,K]
    score = (logits * labels_onehot).sum()  # scalar
    grads = torch.autograd.grad(score, x, create_graph=False, retain_graph=False)[0]
    sal = grads.abs().sum(dim=1, keepdim=True)  # [B,1,H,W]
    sal = _normalize_map(sal)

    # restore
    for p, r in zip(model.parameters(), req):
        p.requires_grad_(r)

    return sal.detach()

# ---- 3) S²-FracMix augmenter ----
class S2FracMixAugment:
    def __init__(
        self,
        scales=(0.2, 0.35, 0.5),
        theta_max=30.0,
        saliency_thresh=0.5,
        fractal_lambda=0.2,
        blur_kernel=7,
        blur_sigma=(0.1, 2.0),
        device="cuda",
    ):
        self.scales = scales
        self.theta_max = theta_max
        self.t = saliency_thresh
        self.lam_f = fractal_lambda
        self.blur = GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)
        self.device = device

    def _sample_salient_topleft(self, mask, ph, pw):
        # mask: [H,W] bool
        H, W = mask.shape
        ys, xs = torch.where(mask)
        if ys.numel() == 0:
            # fallback: uniform anywhere
            y = torch.randint(0, max(1, H - ph + 1), (1,), device=mask.device).item()
            x = torch.randint(0, max(1, W - pw + 1), (1,), device=mask.device).item()
            return y, x

        idx = torch.randint(0, ys.numel(), (1,), device=mask.device).item()
        cy, cx = ys[idx].item(), xs[idx].item()
        y = int(max(0, min(H - ph, cy - ph // 2)))
        x = int(max(0, min(W - pw, cx - pw // 2)))
        return y, x

    def __call__(self, images, saliency):
        """
        images: [B,C,H,W] in [0,1]
        saliency: [B,1,H,W] in [0,1]
        returns: augmented images [B,C,H,W]
        """
        B, C, H, W = images.shape
        out = []

        for i in range(B):
            I = images[i:i+1]          # [1,C,H,W]
            S = saliency[i:i+1]        # [1,1,H,W]
            mask = (S[0,0] >= self.t)  # [H,W] bool

            Pm = torch.zeros_like(I)   # [1,C,H,W]

            for s in self.scales:
                ph = max(1, int(math.floor(s * H)))
                pw = max(1, int(math.floor(s * W)))

                y0, x0 = self._sample_salient_topleft(mask, ph, pw)

                Pk = I[:, :, y0:y0+ph, x0:x0+pw]          # [1,C,ph,pw]
                Sk = S[:, :, y0:y0+ph, x0:x0+pw]          # [1,1,ph,pw]

                # --- FracMix in patch (Eq. 6) ---
                Fk = generate_fractal_1_over_f(1, C, ph, pw, device=Pk.device)
                Pk = self.lam_f * Fk + (1.0 - self.lam_f) * Pk

                # --- Transform (Eq. 3): rotate + blur gated by saliency ---
                theta = (torch.rand(1, device=Pk.device).item() * 2 - 1) * self.theta_max
                Pk_rot = rotate(Pk, angle=theta)
                Pk_blr = self.blur(Pk)
                Tk = Pk_rot * (1.0 - Sk) + Pk_blr * Sk

                # --- Resize to full image and accumulate (Eq. 4, Alg.1) ---
                Rk = F.interpolate(Tk, size=(H, W), mode="bilinear", align_corners=False)
                Pm = Pm + (1.0 / len(self.scales)) * Rk

            alpha = torch.rand(1, device=I.device).item()
            I_tilde = alpha * I + (1.0 - alpha) * Pm  # Eq. 5
            out.append(I_tilde)

        return torch.cat(out, dim=0).clamp(0, 1)

# ---- 4) High-level mixing selector (S²-FracMix vs others) ----
def high_level_mix(images, labels_onehot, s2_aug_fn, s2_prob=0.5):
    """
    Minimal example: choose S²-FracMix or identity.
    Extend this to Mixup/CutMix/ResizeMix as in the paper’s high-level mixing idea.
    """
    if torch.rand(1).item() < s2_prob:
        return images, labels_onehot, True
    else:
        return images, labels_onehot, False
