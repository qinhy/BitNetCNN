
# ===============================================
# CIFAR-100 KD: ResNet-18 (teacher) -> BitResNet-18 (student)
# ===============================================
import os
import time
import numpy as np
import math, copy, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet, BasicBlock
from huggingface_hub import hf_hub_download

from BitNetCNN import BitConv2d, BitLinear, convert_to_ternary_p2

# -------- BitResNet18 with CIFAR stem (3x3 s=1, no maxpool) --------
class BasicBlockBit(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, scale_op="median"):
        super().__init__()
        self.conv1 = BitConv2d(inplanes, planes, 3, stride=stride, padding=1,
                               bias=True, scale_op=scale_op)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.SiLU(inplace=True)
        self.conv2 = BitConv2d(planes, planes, 3, stride=1, padding=1,
                               bias=True, scale_op=scale_op)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class BitResNetCIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=100, scale_op="median", in_ch=3):
        super().__init__()
        self.inplanes = 64
        # CIFAR stem
        self.stem = nn.Sequential(
            BitConv2d(in_ch, self.inplanes, kernel_size=3, stride=1, padding=1,
                      bias=True, scale_op=scale_op),
            nn.BatchNorm2d(self.inplanes),
            nn.SiLU(inplace=True),
        )
        # No maxpool for CIFAR
        # MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, scale_op=scale_op)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, scale_op=scale_op)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, scale_op=scale_op)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, scale_op=scale_op)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            BitLinear(512, num_classes, bias=True, scale_op=scale_op)
        )

    def _make_layer(self, block, planes, blocks, stride, scale_op):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:            
            downsample = nn.Sequential(
                BitConv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=True,
                            scale_op=scale_op),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, scale_op=scale_op)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, scale_op=scale_op))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.head(x)

def bit_resnet18_cifar(num_classes=100, scale_op="median"):
    return BitResNetCIFAR(BasicBlockBit, [2,2,2,2], num_classes, scale_op)

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
    Creates one conv per hint name lazily on first use.
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.ModuleDict()  # name -> Conv2d

    def forward(self, name, f_s, f_t):
        # match spatial size
        f_s = F.adaptive_avg_pool2d(f_s, f_t.shape[-2:])

        # make / get 1x1 conv for this hint point
        c_s, c_t = f_s.shape[1], f_t.shape[1]
        if name not in self.proj:
            self.proj[name] = nn.Conv2d(c_s, c_t, kernel_size=1, bias=True).to(f_s.device)

        # if channels change later (unlikely), rebuild
        elif self.proj[name].in_channels != c_s or self.proj[name].out_channels != c_t:
            self.proj[name] = nn.Conv2d(c_s, c_t, kernel_size=1, bias=True).to(f_s.device)

        f_s = self.proj[name](f_s)
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
    # simple wrappers for mixup/cutmix (optional)
    mix = None
    if aug_cutmix or aug_mixup:
        def mix_collate(batch, alpha=1.0, cutmix=aug_cutmix, mixup=aug_mixup):
            import random
            xs, ys = zip(*batch); x = torch.stack(xs); y = torch.tensor(ys)
            lam = 1.0
            if cutmix and random.random()<0.5:
                lam = torch.distributions.Beta(alpha,alpha).sample().item()
                idx = torch.randperm(x.size(0))
                h,w = x.size(2), x.size(3)
                rx, ry = torch.randint(w,(1,)).item(), torch.randint(h,(1,)).item()
                rw = int(w*math.sqrt(1-lam)); rh = int(h*math.sqrt(1-lam))
                x1, y1 = max(rx-rw//2,0), max(ry-rh//2,0)
                x2, y2 = min(rx+rw//2,w), min(ry+rh//2,h)
                x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
                y = (y, y[idx], 1 - ((x2-x1)*(y2-y1)/(w*h)))
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

# -------------- Train & Eval --------------
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

# --- Teacher: ResNet-18 with CIFAR stem (3x3 s=1, no maxpool) + load HF weights ---
def make_resnet18_cifar_teacher_from_hf(device="cuda"):

    # Build an ImageNet ResNet-18 *skeleton* but with CIFAR stem
    class ResNet18CIFAR(ResNet):
        def __init__(self, num_classes=100):
            super().__init__(block=BasicBlock, layers=[2,2,2,2], num_classes=num_classes)
            # overwrite stem: 3x3 s=1, no maxpool
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
            self.maxpool = nn.Identity()

    model = ResNet18CIFAR(num_classes=100)

    # Download HF weights and load strictly
    ckpt_path = hf_hub_download(repo_id="edadaltocg/resnet18_cifar100", filename="pytorch_model.bin")
    state = torch.load(ckpt_path, map_location="cpu",weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Some repos save classifier as 'fc.weight/bias' (expected), but if there are
    # 'classifier.*' or bn tracking keys, strict=False handles it.

    if missing:
        print(f"[teacher] Missing keys: {missing}")
    if unexpected:
        print(f"[teacher] Unexpected keys: {unexpected}")

    return model.to(device).eval()

# -------------- Train & Distill with detailed prints --------------
def distill_cifar100(
    data_root="./data", out_dir="./checkpoints_c100",
    epochs=200, batch_size=128, lr=0.2, wd=5e-4, label_smoothing=0.1,
    alpha_kd=0.7, alpha_hint=0.05, T=4.0, amp=True,
    scale_op="median", device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- Data --------------------
    train_loader, val_loader = cifar100_loaders(data_root, batch_size=batch_size, workers=1)

    # -------------------- Teacher (HF CIFAR-100 weights, CIFAR stem) --------------------
    teacher = make_resnet18_cifar_teacher_from_hf(device)

    # Freeze teacher for faster distill
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # -------------------- Student --------------------
    best_model = student = bit_resnet18_cifar(num_classes=100, scale_op=scale_op).to(device)

    # -------------------- Losses / KD --------------------
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    kd = KDLoss(T=T)

    # Hints
    hint_points = ["layer1", "layer2", "layer3", "layer4"]
    t_feats, t_handles = make_feature_hooks(teacher, hint_points)
    s_feats, s_handles = make_feature_hooks(student, hint_points)
    hint = AdaptiveHintLoss()

    # -------------------- Student Optim/Sched/Scaler --------------------
    opt = torch.optim.SGD(
        list(student.parameters()) + list(hint.parameters()),
        lr=lr, momentum=0.9, weight_decay=wd, nesterov=True
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and str(device).startswith("cuda")))

    # -------------------- Banner --------------------
    _banner("Distillation Run", {
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
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "cuda_mem": _cuda_mem()
    })

    best = 0.0
    os.makedirs(out_dir, exist_ok=True)

    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Optional memory/throughput boost:
        student = student.to(memory_format=torch.channels_last)
        teacher = teacher.to(memory_format=torch.channels_last)

    for epoch in range(1, epochs+1):

        m_loss, m_ce, m_kd, m_hint = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        start = time.time()

        # Train loop
        student.train()
        teacher.eval()  # always eval for KD

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

            # ---------- KD targets ----------
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_amp):
                z_t = teacher(x)

            # ---------- Student step ----------
            with torch.amp.autocast("cuda", enabled=use_amp):
                z_s = student(x)

                # CE
                if is_mix:
                    loss_ce = lam * ce(z_s, y_a) + (1 - lam) * ce(z_s, y_b)
                else:
                    loss_ce = ce(z_s, y)

                # KD (in fp32)
                with torch.amp.autocast("cuda", enabled=False):
                    loss_kd = kd(z_s.float(), z_t.float())

                # Hints
                loss_hint = 0.0
                if alpha_hint > 0:
                    for n in hint_points:
                        if (n in s_feats) and (n in t_feats):
                            loss_hint = loss_hint + hint(n, s_feats[n].float(), t_feats[n].float())

                loss = (1.0 - alpha_kd) * loss_ce + alpha_kd * loss_kd + alpha_hint * loss_hint

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            m_loss.add(loss.item(), bs)
            m_ce.add(loss_ce.item(), bs)
            m_kd.add(loss_kd.item(), bs)
            m_hint.add(float(loss_hint), bs)

        # ---- End epoch ----
        sched.step()

        # Teacher eval top-1 (no renorm)
        teach_top1 = 79.28
        # teach_top1 = eval_top1(teacher, val_loader, renorm=None, device=device, amp=amp)

        # Student eval (export to ternary for metric, like your original)
        student_top1 = eval_top1(student, val_loader, device=device, amp=amp)
        tern = convert_to_ternary_p2(copy.deepcopy(student))
        top1 = eval_top1(tern, val_loader, device=device, amp=amp)

        is_best = top1 > best
        if is_best:
            best = top1
            best_model = copy.deepcopy(student).cpu().eval()

            ckpt = f"{out_dir}/bit_resnet18_c100_kd_best.pt"
            torch.save({"model": best_model.state_dict(), "top1": best}, ckpt)
            print(f"✓ saved {ckpt} (top1={best:.2f})")

            tern = tern.cpu().eval()
            tpath = f"{out_dir}/bit_resnet18_c100_kd_ternary.pt"
            torch.save({"model": tern.state_dict(), "top1": best}, tpath)
            print(f"✓ exported ternary PoT → {tpath}")

        epoch_time = time.time() - start
        imgs = len(train_loader.dataset)
        ips = imgs / max(epoch_time, 1e-6)

        print(
            f"[{epoch:03d}/{epochs}] "
            f"loss {m_loss.avg:.4f} (CE {m_ce.avg:.4f} | KD {m_kd.avg:.4f} | Hint {m_hint.avg:.4f}) | "
            f"top1 {top1:.2f} | float top1 {student_top1:.2f} | best {best:.2f} | teach_top1 {teach_top1:.2f} | "
            f"lr {_get_lr(opt):.4f} | "
            f"time {epoch_time:.1f}s | {ips:.0f} img/s | cuda { _cuda_mem() }"
        )

    for h in t_handles + s_handles:
        try: h.remove()
        except Exception: pass

    print(f"Best Top-1: {best:.2f}")
    return best_model


# --- quick run (adjust paths & hparams) ---
if __name__ == "__main__":
    best_model = distill_cifar100(
        data_root="./data",
        out_dir="./ckpt_c100_kd",
        epochs=200, batch_size=1024,
        lr=0.2, wd=5e-4, label_smoothing=0.1,
        alpha_kd=0.7, alpha_hint=0.05, T=4.0,
        amp=True, scale_op="median",
    )


# ================================
# == Distillation Run ==
# ================================
#               device: cuda
#                  AMP: True
#               epochs: 200
#           batch_size: 1280
#              init LR: 0.2
#         weight_decay: 0.0005
#             alpha_kd: 0.7
#           alpha_hint: 0.05
#               T (KD): 4.0
#             scale_op: mean
#        train_batches: 39
#          val_batches: 40
#             cuda_mem: 87/108 MiB
# ✓ saved ./ckpt_c100_kd/bit_resnet18_c100_kd_best.pt (top1=5.00)
# ✓ exported ternary PoT → ./ckpt_c100_kd/bit_resnet18_c100_kd_ternary.pt
# [001/200] loss 4.9688 (CE 4.0350 | KD 5.2924 | Hint 1.0722) | top1 5.00 | best 5.00 | teach_top1 79.28 | lr 0.2000 | time 32.7s | 1529 img/s | cuda 819/6296 MiB
# ✓ saved ./ckpt_c100_kd/bit_resnet18_c100_kd_best.pt (top1=9.71)
# ✓ exported ternary PoT → ./ckpt_c100_kd/bit_resnet18_c100_kd_ternary.pt
# [002/200] loss 4.2185 (CE 3.3563 | KD 4.5172 | Hint 0.9923) | top1 9.71 | best 9.71 | teach_top1 79.28 | lr 0.2000 | time 32.8s | 1525 img/s | cuda 818/6296 MiB
# ✓ saved ./ckpt_c100_kd/bit_resnet18_c100_kd_best.pt (top1=19.19)
# ✓ exported ternary PoT → ./ckpt_c100_kd/bit_resnet18_c100_kd_ternary.pt
# [003/200] loss 3.6535 (CE 2.9118 | KD 3.9046 | Hint 0.9358) | top1 19.19 | best 19.19 | teach_top1 79.28 | lr 0.1999 | time 33.2s | 1506 img/s | cuda 818/6296 MiB
# ✓ saved ./ckpt_c100_kd/bit_resnet18_c100_kd_best.pt (top1=26.06)
# ✓ exported ternary PoT → ./ckpt_c100_kd/bit_resnet18_c100_kd_ternary.pt
# [004/200] loss 3.1588 (CE 2.5582 | KD 3.3533 | Hint 0.8811) | top1 26.06 | best 26.06 | teach_top1 79.28 | lr 0.1998 | time 33.2s | 1505 img/s | cuda 818/6296 MiB
# ✓ saved ./ckpt_c100_kd/bit_resnet18_c100_kd_best.pt (top1=32.08)
# ✓ exported ternary PoT → ./ckpt_c100_kd/bit_resnet18_c100_kd_ternary.pt
# .
# .
# .
# [133/200] loss 0.4076 (CE 1.1442 | KD 0.0808 | Hint 0.1553) | top1 62.00 | best 64.65 | teach_top1 79.28 | lr 0.0505 | time 32.9s | 1522 img/s | cuda 829/6298 MiB
# [134/200] loss 0.4064 (CE 1.1447 | KD 0.0789 | Hint 0.1552) | top1 60.47 | best 64.65 | teach_top1 79.28 | lr 0.0491 | time 33.6s | 1487 img/s | cuda 830/6298 MiB
# [135/200] loss 0.4058 (CE 1.1445 | KD 0.0781 | Hint 0.1552) | top1 60.33 | best 64.65 | teach_top1 79.28 | lr 0.0478 | time 33.1s | 1512 img/s | cuda 832/6298 MiB
# [136/200] loss 0.4049 (CE 1.1448 | KD 0.0767 | Hint 0.1551) | top1 60.65 | best 64.65 | teach_top1 79.28 | lr 0.0464 | time 32.5s | 1541 img/s | cuda 829/6298 MiB
# [137/200] loss 0.4044 (CE 1.1447 | KD 0.0761 | Hint 0.1551) | top1 58.38 | best 64.65 | teach_top1 79.28 | lr 0.0451 | time 33.0s | 1516 img/s | cuda 830/6298 MiB
# [138/200] loss 0.4039 (CE 1.1454 | KD 0.0750 | Hint 0.1551) | top1 55.54 | best 64.65 | teach_top1 79.28 | lr 0.0438 | time 32.5s | 1536 img/s | cuda 832/6298 MiB
# [139/200] loss 0.4033 (CE 1.1446 | KD 0.0745 | Hint 0.1550) | top1 56.73 | best 64.65 | teach_top1 79.28 | lr 0.0425 | time 33.2s | 1508 img/s | cuda 829/6298 MiB
# [140/200] loss 0.4026 (CE 1.1451 | KD 0.0733 | Hint 0.1550) | top1 56.17 | best 64.65 | teach_top1 79.28 | lr 0.0412 | time 33.2s | 1507 img/s | cuda 830/6298 MiB
