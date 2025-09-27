# ===============================================
# CIFAR-100 KD: MobileNetV2 (teacher) -> BitMobileNetV2 (student)
# ===============================================
import os
import time
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# teacher weights from torch.hub
#   repo: chenyaofo/pytorch-cifar-models
#   models: "cifar100_mobilenetv2_x1_0", "cifar100_mobilenetv2_x1_4", etc.
from BitNetCNN import BitConv2d, BitLinear, convert_to_ternary_p2

# -------------------------------------------------------------
# CIFAR-friendly MobileNetV2 built on BitNet layers.
# Tweaks:
#  - Uses BitConv2d / BitLinear
#  - Removes conv biases when followed by BN
#  - Optional SiLU activations
#  - Weight initialization pass
#  - Hook-friendly stage "hint" names
#  - Utility to ternarize all Bit layers post-training
# -------------------------------------------------------------

# -------------- Utils --------------
def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# -------------- Building Blocks --------------
class ConvBNActBit(nn.Module):
    """
    Conv (BitConv2d) -> BN -> Act
    By default we set bias=False since BN follows.
    """
    def __init__(self, in_ch, out_ch, k, s, p, groups=1, scale_op="median", act="silu"):
        super().__init__()
        self.conv = BitConv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False,
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
    """
    Inverted residual block:
      - Optional 1x1 expand
      - 3x3 depthwise
      - 1x1 project (linear), BN only (no activation)
      - Residual if stride==1 and input channels == output channels
    """
    def __init__(self, inp, oup, stride, expand_ratio, scale_op="median", act="silu"):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)
        layers = []
        # pw (expand)
        if expand_ratio != 1:
            layers.append(ConvBNActBit(inp, hidden_dim, 1, 1, 0, scale_op=scale_op, act=act))
        # dw
        layers.append(
            ConvBNActBit(hidden_dim, hidden_dim, 3, stride, 1,
                         groups=hidden_dim, scale_op=scale_op, act=act)
        )
        # pw-linear (project): BN only afterwards, so bias=False
        layers.append(BitConv2d(hidden_dim, oup, 1, 1, 0, bias=False, scale_op=scale_op))
        layers.append(nn.BatchNorm2d(oup))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connect:
            out = x + out
        return out


# -------------- Model --------------
class BitMobileNetV2(nn.Module):
    """
    CIFAR-friendly BitMobileNetV2:
      - stem: 3x3 s=1 (no initial downsample; CIFAR is 32x32)
      - stride pattern approx. torchvision MobileNetV2 on ImageNet:
        [1, 2, 2, 2] across stages after the stem
    """
    def __init__(self, num_classes=100, width_mult=1.0, round_nearest=8,
                 scale_op="median", in_ch=3, act="silu", last_channel_override=None):
        super().__init__()
        # MobileNetV2 default setting: (t, c, n, s)
        setting = [
            # t,  c,  n,  s
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

        # Stem: 3x3 s=1
        self.stem = ConvBNActBit(in_ch, input_channel, k=3, s=1, p=1, scale_op=scale_op, act=act)

        # Stages
        features = []
        self.hint_names = []  # collect ids for hint hooks (after each stage)
        for t, c, n, s in setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidualBit(input_channel, output_channel, stride,
                                        expand_ratio=t, scale_op=scale_op, act=act)
                )
                input_channel = output_channel
            # mark hint after finishing this stage
            self.hint_names.append(f"features.{len(features)-1}")

        self.features = nn.Sequential(*features)

        # Head
        self.head_conv  = ConvBNActBit(input_channel, last_channel, k=1, s=1, p=0,
                                       scale_op=scale_op, act=act)
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = BitLinear(last_channel, num_classes, bias=True, scale_op=scale_op)

        # Init
        self._init_weights()

    # ---------- Utilities ----------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, BitConv2d):
                # He init is fine even with SiLU/ReLU6; BitConv2d should hold .weight
                if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, BitLinear):
                if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
                    nn.init.normal_(m.weight, 0.0, 0.01)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_submodule(self, name: str) -> nn.Module:
        """
        Retrieve a nested submodule by dotted path (e.g., 'features.12').
        """
        mod = self
        for attr in name.split('.'):
            mod = getattr(mod, attr)
        return mod

    # ---------- Forward ----------
    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head_conv(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x


# -------------- Factory --------------
def bit_mobilenetv2_cifar(num_classes=100, width_mult=1.0, scale_op="median",
                          in_ch=3, act="silu", last_channel_override=None):
    """
    Factory for BitMobileNetV2.
    - last_channel_override: set e.g. 1024 for smaller heads on tiny models.
    """
    return BitMobileNetV2(num_classes=num_classes,
                          width_mult=width_mult,
                          scale_op=scale_op,
                          in_ch=in_ch,
                          act=act,
                          last_channel_override=last_channel_override)
# -------------- KD losses --------------
class KDLoss(nn.Module):
    def __init__(self, T=4.0): super().__init__(); self.T=T
    def forward(self, z_s, z_t):
        T = self.T
        return F.kl_div(F.log_softmax(z_s/T,1), F.softmax(z_t/T,1), reduction="batchmean") * (T*T)

# Optional: feature hints (stage outputs)
class AdaptiveHintLoss(nn.Module):
    """
    Per-point learnable 1x1 projection + adaptive pooling so student feature
    matches teacher's (N, C_t, H_t, W_t) before SmoothL1.
    Stored in a ModuleDict using sanitized keys (no dots).
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.ModuleDict()  # safe_key -> Conv2d

    @staticmethod
    def _safe(name: str) -> str:
        # Module names cannot contain dots in ModuleDict keys
        return name.replace(".", "_")

    def forward(self, name, f_s, f_t):
        # 1) spatial match
        f_s = F.adaptive_avg_pool2d(f_s, f_t.shape[-2:])

        # 2) get/create a 1x1 projection for this hint point
        c_s, c_t = f_s.shape[1], f_t.shape[1]
        key = self._safe(name)

        if key not in self.proj:
            self.proj[key] = nn.Conv2d(c_s, c_t, kernel_size=1, bias=True).to(f_s.device)
        else:
            # if channels changed across re-runs / width multipliers, rebuild
            need_reset = (
                self.proj[key].in_channels  != c_s or
                self.proj[key].out_channels != c_t
            )
            if need_reset:
                self.proj[key] = nn.Conv2d(c_s, c_t, kernel_size=1, bias=True).to(f_s.device)

        f_s = self.proj[key](f_s)
        return F.smooth_l1_loss(f_s, f_t.detach())

def make_feature_hooks(module, names):
    feats = {}
    handles = []
    def hook(name):
        def fwd_hook(_m, _inp, out): feats[name] = out
        return fwd_hook
    for n, sub in module.named_modules():
        if n in names:
            handles.append(sub.register_forward_hook(hook(n)))
    return feats, handles

# -------------- Data (CIFAR-100) --------------
def cifar100_loaders(root, batch_size=128, workers=4, aug_cutmix=False, aug_mixup=False):
    mean = (0.5071,0.4867,0.4408); std=(0.2675,0.2565,0.2761)
    train_tf = [transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean,std),
                ]
    val_tf   = [transforms.ToTensor(), 
                transforms.Normalize(mean,std),
                ]
    train = datasets.CIFAR100(root=root, train=True, download=True, transform=transforms.Compose(train_tf))
    val   = datasets.CIFAR100(root=root, train=False, download=True, transform=transforms.Compose(val_tf))

    loader_train = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True, drop_last=True,
                                               persistent_workers=True if workers > 0 else False)
    loader_val   = torch.utils.data.DataLoader(val, batch_size=256, shuffle=False,
                                               num_workers=workers, pin_memory=True,
                                               persistent_workers=True if workers > 0 else False)

    if aug_cutmix or aug_mixup:
        def mix_collate(batch, alpha=1.0, cutmix=aug_cutmix, mixup=aug_mixup):
            import random
            xs, ys = zip(*batch); x = torch.stack(xs); y = torch.tensor(ys)
            if cutmix and random.random() < 0.5:
                lam = torch.distributions.Beta(alpha,alpha).sample().item()
                idx = torch.randperm(x.size(0))
                h,w = x.size(2), x.size(3)
                rx, ry = torch.randint(w,(1,)).item(), torch.randint(h,(1,)).item()
                rw = int(w*math.sqrt(1-lam)); rh = int(h*math.sqrt(1-lam))
                x1, y1 = max(rx-rw//2,0), max(ry-rh//2,0)
                x2, y2 = min(rx+rw//2,w), min(ry+rh//2,h)
                x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
                lam = 1 - ((x2-x1)*(y2-y1)/(w*h))
                y = (y, y[idx], lam)
            elif mixup:
                lam = torch.distributions.Beta(alpha,alpha).sample().item()
                idx = torch.randperm(x.size(0))
                x = lam*x + (1-lam)*x[idx]
                y = (y, y[idx], lam)
            return x, y
        loader_train = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                                   num_workers=workers, pin_memory=True, drop_last=True,
                                                   collate_fn=mix_collate)
    return loader_train, loader_val

# ---------- utils: pretty logging ----------
class AvgMeter:
    __slots__ = ("n","sum")
    def __init__(self): self.n=0; self.sum=0.0
    def add(self, v, k=1): self.sum += float(v)*k; self.n += int(k)
    @property
    def avg(self): return (self.sum / max(self.n,1))

def _cuda_mem():
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(i)/1024**2
        rsv   = torch.cuda.memory_reserved(i)/1024**2
        return f"{alloc:.0f}/{rsv:.0f} MiB"
    return "-"

def _banner(title, kv):
    bar = "="*max(32, len(title)+6)
    lines = [bar, f"== {title} ==", bar]
    for k,v in kv.items(): lines.append(f"{k:>20}: {v}")
    print("\n".join(lines))

def _get_lr(optim):
    return optim.param_groups[0]["lr"]

# -------------- Eval --------------
@torch.inference_mode()
def eval_top1(model, loader, renorm=None, device="cuda", amp=True):
    model = model.to(device).eval()
    tot = 0; corr = 0
    use_amp = bool(amp and str(device).startswith("cuda"))
    for x, y in loader:
        x,y=x.to(device,non_blocking=True), y.to(device,non_blocking=True)
        x_t = renorm(x) if renorm else x
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x_t)
        pred = logits.argmax(1)
        tot += y.size(0); corr += (pred == y).sum().item()
    return 100.0 * corr / max(1, tot)

# -------------- Teacher: MobileNetV2 (torch.hub) --------------
def make_mobilenetv2_teacher_from_hub(variant="cifar100_mobilenetv2_x1_4", device="cuda"):
    # fallback if hub not available: user can change variant to "cifar100_mobilenetv2_x1_0"
    teacher = torch.hub.load("chenyaofo/pytorch-cifar-models", variant, pretrained=True)
    return teacher.to(device).eval()

# -------------- Train & Distill --------------
def distill_cifar100(
    data_root="./data", out_dir="./checkpoints_c100_mbv2",
    epochs=200, batch_size=128, lr=0.2, wd=5e-4, label_smoothing=0.1,
    alpha_kd=0.7, alpha_hint=0.05, T=4.0, amp=True,
    scale_op="median", width_mult=1.0, device=None,
    teacher_variant="cifar100_mobilenetv2_x1_4"
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- Data --------------------
    train_loader, val_loader = cifar100_loaders(data_root, batch_size=batch_size, workers=1)

    # -------------------- Teacher --------------------
    teacher = make_mobilenetv2_teacher_from_hub(teacher_variant, device)

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # -------------------- Student --------------------
    best_model = student = bit_mobilenetv2_cifar(num_classes=100, width_mult=width_mult, scale_op=scale_op).to(device)

    # -------------------- Losses --------------------
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    kd = KDLoss(T=T)

    # Hints: tap the last block of each MobileNetV2 stage
    # Student hint names are populated inside BitMobileNetV2 as "features.{idx}"
    s_hint_points = student.hint_names
    # For teacher, pick roughly corresponding modules: try to find "features.{idx}" too (torchvision-like)
    # Fallback: if names don't exist, hooks dict will stay partially empty and it's fine.
    t_hint_points = []
    for n, _m in teacher.named_modules():
        if n.startswith("features."):
            t_hint_points.append(n)
    # choose a coarse stage boundary subset for teacher: last block indices of similar stages
    # We'll map by counting lengths.
    # Build teacher stage ends by scanning changes in stride or channel if available is complex;
    # for simplicity, pick evenly spaced 7 marks if possible.
    t_feature_names_sorted = sorted([n for n in t_hint_points if n.count(".")==1], key=lambda x:int(x.split(".")[1]))
    # pick up to len(s_hint_points) from the end of each teacher stage-ish division
    if len(t_feature_names_sorted) >= len(s_hint_points) and len(s_hint_points)>0:
        step = len(t_feature_names_sorted)/len(s_hint_points)
        chosen = [t_feature_names_sorted[int(round(step*(i+1))-1)] for i in range(len(s_hint_points))]
        t_hint_points = chosen
    else:
        # fallback to last N features
        t_hint_points = t_feature_names_sorted[-len(s_hint_points):] if len(s_hint_points)>0 else []

    t_feats, t_handles = make_feature_hooks(teacher, t_hint_points)
    s_feats, s_handles = make_feature_hooks(student, s_hint_points)
    hint = AdaptiveHintLoss()

    # -------------------- Optim/Sched/Scaler --------------------
    opt = torch.optim.SGD(
        list(student.parameters()) + list(hint.parameters()),
        lr=lr, momentum=0.9, weight_decay=wd, nesterov=True
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and str(device).startswith("cuda")))

    # -------------------- Banner --------------------
    _banner("Distillation Run (MBv2 -> BitMBv2)", {
        "device": device,
        "AMP": amp,
        "epochs": epochs,
        "batch_size": batch_size,
        "init LR": lr,
        "weight_decay": wd,
        "alpha_kd": alpha_kd,
        "alpha_hint": alpha_hint,
        "T (KD)": T,
        "scale_op": scale_op,
        "width_mult": width_mult,
        "teacher": teacher_variant,
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "cuda_mem": _cuda_mem()
    })

    best = 0.0
    os.makedirs(out_dir, exist_ok=True)

    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        student = student.to(memory_format=torch.channels_last)
        try:
            teacher = teacher.to(memory_format=torch.channels_last)
        except Exception:
            pass

    # (Optional) cache an estimated teacher acc (for prints only)
    try:
        teach_top1_est = 75.98 if "x1_4" in teacher_variant else 74.0
    except Exception:
        teach_top1_est = -1

    for epoch in range(1, epochs+1):
        m_loss, m_ce, m_kd, m_hint = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        start = time.time()

        student.train()
        teacher.eval()

        for step, batch in enumerate(train_loader, 1):
            x, y = batch
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)

            is_mix = isinstance(y, tuple)
            if is_mix:
                y_a, y_b, lam = y
                y_a, y_b = y_a.to(device), y_b.to(device)
            else:
                y = y.to(device)

            opt.zero_grad(set_to_none=True)
            use_amp = bool(amp and str(device).startswith("cuda"))

            # Student step
            with torch.amp.autocast("cuda", enabled=use_amp):
                z_s = student(x)

                # CE
                if is_mix:
                    loss_ce = lam * ce(z_s, y_a) + (1 - lam) * ce(z_s, y_b)
                else:
                    loss_ce = ce(z_s, y)

                # KD in fp32
                loss_kd = 0.0
                if alpha_kd>0:
                    # KD targets
                    with torch.inference_mode():
                        z_t = teacher(x)
                    with torch.amp.autocast("cuda", enabled=False):
                        loss_kd = kd(z_s.float(), z_t.float())

                # Hints
                loss_hint = 0.0
                if alpha_hint>0:
                    if (len(s_hint_points) > 0) and (len(t_hint_points) > 0):
                        # pair by index
                        for s_name, t_name in zip(s_hint_points, t_hint_points):
                            if (s_name in s_feats) and (t_name in t_feats):
                                loss_hint += hint(s_name, s_feats[s_name].float(), t_feats[t_name].float())


                loss = (1.0 - alpha_kd) * loss_ce + alpha_kd * loss_kd + alpha_hint * loss_hint

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            m_loss.add(loss.item(), bs)
            m_ce.add(loss_ce.item(), bs)

            if alpha_kd>0:
                m_kd.add(loss_kd.item(), bs)
                
            if alpha_hint>0:
                m_hint.add(float(loss_hint), bs)

        sched.step()

        # Metrics
        student_top1 = eval_top1(student, val_loader, device=device, amp=amp)
        tern = convert_to_ternary_p2(copy.deepcopy(student))
        top1 = eval_top1(tern, val_loader, device=device, amp=amp)

        # Save best (float + ternary export)
        is_best = top1 > best
        if is_best:
            best = top1
            best_model = copy.deepcopy(student).cpu().eval()

            ckpt = f"{out_dir}/bit_mbv2_c100_kd_best.pt"
            torch.save({"model": best_model.state_dict(), "top1": best}, ckpt)
            print(f"✓ saved {ckpt} (top1={best:.2f})")

            tern = tern.cpu().eval()
            tpath = f"{out_dir}/bit_mbv2_c100_kd_ternary.pt"
            torch.save({"model": tern.state_dict(), "top1": best}, tpath)
            print(f"✓ exported ternary PoT → {tpath}")

        epoch_time = time.time() - start
        imgs = len(train_loader.dataset)
        ips = imgs / max(epoch_time, 1e-6)

        print(
            f"[{epoch:03d}/{epochs}] "
            f"loss {m_loss.avg:.4f} (CE {m_ce.avg:.4f}, KD {m_kd.avg:.4f}, Hint {m_hint.avg:.4f}) | "
            f"top1 {top1:.2f} | float top1 {student_top1:.2f} | best {best:.2f} | "
            f"teach_top1 {teach_top1_est:.2f} | "
            f"lr {_get_lr(opt):.4f} | "
            f"time {epoch_time:.1f}s | {ips:.0f} img/s | cuda { _cuda_mem() }"
        )

    for h in t_handles + s_handles:
        try: h.remove()
        except Exception: pass

    print(f"Best Top-1 (ternary eval): {best:.2f}")
    return best_model

# --- quick run (adjust paths & hparams) ---
if __name__ == "__main__":
    best_model = distill_cifar100(
        data_root="./data",
        out_dir="./ckpt_c100_kd_mbv2",
        epochs=200, batch_size=1024,
        lr=0.2, wd=5e-4, label_smoothing=0.1,
        alpha_kd=0.5, alpha_hint=0.1, T=4.0,
        amp=True, scale_op="median",
        width_mult=1.0,  # student width; try 1.0 or 0.75 for lighter
        teacher_variant="cifar100_mobilenetv2_x1_4"
    )
