# ===============================================
# CIFAR-100 KD: ResNet-50 (teacher) -> BitResNet-18 (student)
# ===============================================
import os
import numpy as np
import math, copy, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms, models

from BitNetCNN import BitConv2d, BitLinear, convert_to_ternary_p2

# -------- BitResNet18 with CIFAR stem (3x3 s=1, no maxpool) --------
class BasicBlockBit(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, act_bits=8, scale_op="mean"):
        super().__init__()
        self.conv1 = BitConv2d(inplanes, planes, 3, stride=stride, padding=1,
                               bias=False, act_bits=act_bits, scale_op=scale_op)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = BitConv2d(planes, planes, 3, stride=1, padding=1,
                               bias=False, act_bits=act_bits, scale_op=scale_op)
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
    def __init__(self, block, layers, num_classes=100, act_bits=8, scale_op="mean", in_ch=3,
                 first_last_float=False):
        super().__init__()
        self.inplanes = 64
        # CIFAR stem
        if first_last_float:
            self.conv1 = nn.Conv2d(in_ch, 64, 3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = BitConv2d(in_ch, 64, 3, stride=1, padding=1,
                                   bias=False, act_bits=act_bits, scale_op=scale_op)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # No maxpool for CIFAR
        # MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, act_bits=act_bits, scale_op=scale_op)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, act_bits=act_bits, scale_op=scale_op)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, act_bits=act_bits, scale_op=scale_op)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, act_bits=act_bits, scale_op=scale_op)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if first_last_float:
            self.fc = nn.Linear(512, num_classes, bias=True)
        else:
            self.fc = BitLinear(512, num_classes, bias=True, act_bits=None, scale_op=scale_op)

    def _make_layer(self, block, planes, blocks, stride, act_bits, scale_op):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if isinstance(self.conv1, nn.Conv2d):  # first_last_float -> keep downsample float too
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    BitConv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False,
                              act_bits=act_bits, scale_op=scale_op),
                    nn.BatchNorm2d(planes * block.expansion),
                )
        layers = [block(self.inplanes, planes, stride, downsample, act_bits=act_bits, scale_op=scale_op)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, act_bits=act_bits, scale_op=scale_op))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def bit_resnet18_cifar(num_classes=100, act_bits=8, scale_op="mean", first_last_float=False):
    return BitResNetCIFAR(BasicBlockBit, [2,2,2,2], num_classes, act_bits, scale_op,
                          first_last_float=first_last_float)

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
            self.proj[name] = nn.Conv2d(c_s, c_t, kernel_size=1, bias=False).to(f_s.device)

        # if channels change later (unlikely), rebuild
        elif self.proj[name].in_channels != c_s or self.proj[name].out_channels != c_t:
            self.proj[name] = nn.Conv2d(c_s, c_t, kernel_size=1, bias=False).to(f_s.device)

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

class ReNormalize(nn.Module):
    def __init__(self, mean_src, std_src, mean_tgt, std_tgt):
        super().__init__()
        ms, ss = torch.tensor(mean_src).view(1,-1,1,1), torch.tensor(std_src).view(1,-1,1,1)
        mt, st = torch.tensor(mean_tgt).view(1,-1,1,1), torch.tensor(std_tgt).view(1,-1,1,1)
        # from (x - ms)/ss  ->  (x - mt)/st   with x already normalized by (ms, ss)
        self.a = (ss / st)
        self.b = (ms - mt) / st
    def forward(self, x):
        # x is CIFAR-normalized; turn it into ImageNet-normalized view
        return self.a.to(x.device) * x + self.b.to(x.device)
    
# -------------- Data (CIFAR-100) --------------
def cifar100_loaders(root, batch_size=128, workers=4, aug_cutmix=False, aug_mixup=False):
    mean = (0.5071,0.4867,0.4408); std=(0.2675,0.2565,0.2761)
    train_tf = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), transforms.Normalize(mean,std)]
    val_tf   = [transforms.ToTensor(), transforms.Normalize(mean,std)]
    train = datasets.CIFAR100(root=root, train=True, download=True, transform=transforms.Compose(train_tf))
    val   = datasets.CIFAR100(root=root, train=False, download=True, transform=transforms.Compose(val_tf))

    loader_train = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True, drop_last=True)
    loader_val   = torch.utils.data.DataLoader(val, batch_size=256, shuffle=False,
                                               num_workers=workers, pin_memory=True)
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

# -------------- Train & Eval --------------
@torch.inference_mode()
def eval_top1(model, loader, device="cuda", amp=True):
    tot=0; corr=0
    model = convert_to_ternary_p2(copy.deepcopy(model)).to(device).eval()
    for x,y in loader:
        x,y=x.to(device,non_blocking=True), y.to(device,non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(amp and device.startswith("cuda"))):
            logits=model(x)
        pred=logits.argmax(1)
        tot+=y.size(0); corr+=(pred==y).sum().item()
    return 100.0*corr/tot

def one_hot(labels, num_classes):
    y = torch.zeros(labels.size(0), num_classes, device=labels.device)
    return y.scatter_(1, labels.view(-1,1), 1.)
import copy, os, time, math, torch, torch.nn as nn
from torchvision import models

# ---------- utils: pretty logging ----------
def _num(x):
    # compact numeric formatter
    if isinstance(x, (int,)) or (isinstance(x, float) and (x==0 or 1e-3<=abs(x)<1e4)):
        return f"{x:.4g}" if isinstance(x, float) else str(x)
    if isinstance(x, float):
        return f"{x:.3e}"
    return str(x)

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

_HAS_TQDM = False

# -------------- Eval (unchanged logic, clearer prints) --------------
@torch.inference_mode()
def eval_top1(model, loader, device="cuda", amp=True, desc="eval"):
    tot, corr = 0, 0
    model = convert_to_ternary_p2(copy.deepcopy(model)).to(device).eval()
    use_amp = bool(amp and str(device).startswith("cuda"))
    iters = len(loader)
    iterator = loader
    if _HAS_TQDM:
        iterator = tqdm(loader, desc=f"{desc} (ternary)", leave=False, ncols=100)
    for x,y in iterator:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
        pred = logits.argmax(1)
        tot += y.size(0); corr += (pred == y).sum().item()
    return 100.0 * corr / max(1, tot)

# -------------- Train & Distill with detailed prints --------------
def distill_cifar100(
    data_root="./data", out_dir="./checkpoints_c100",
    epochs=200, batch_size=128, lr=0.2, wd=5e-4, label_smoothing=0.1,
    alpha_kd=0.7, alpha_hint=0.05, T=4.0, amp=True,
    act_bits=8, scale_op="mean", first_last_float=False, device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader = cifar100_loaders(data_root, batch_size=batch_size, workers=8)

    # Teacher: ResNet-18 pretrained on ImageNet, replace head for 100 classes
    teacher = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    teacher.fc = nn.Linear(512, 100)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval().to(device)
    
    # Stats and re-normalizer (CIFAR->ImageNet for teacher)
    cifar_mean = (0.5071, 0.4867, 0.4408); cifar_std = (0.2675, 0.2565, 0.2761)
    imnet_mean = (0.485, 0.456, 0.406);     imnet_std = (0.229, 0.224, 0.225)
    renorm = ReNormalize(cifar_mean, cifar_std, imnet_mean, imnet_std).to(device)

    # Teacher warmup (first 20 epochs only)
    teach_opt = torch.optim.SGD(teacher.parameters(), lr=0.05, momentum=0.9, weight_decay=wd)
    teach_sched = torch.optim.lr_scheduler.CosineAnnealingLR(teach_opt, T_max=20)

    # Student
    best_model = student = bit_resnet18_cifar(
        num_classes=100, act_bits=act_bits, scale_op=scale_op, first_last_float=first_last_float
    ).to(device)

    # KD components
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    kd = KDLoss(T=T)
    hint_points = ["layer1", "layer2", "layer3", "layer4"]
    t_feats, t_handles = make_feature_hooks(teacher, hint_points)
    s_feats, s_handles = make_feature_hooks(student, hint_points)
    hint = AdaptiveHintLoss()

    # Optimizer/scheduler + AMP scaler
    opt = torch.optim.SGD(
        list(student.parameters()) + list(hint.parameters()),
        lr=lr, momentum=0.9, weight_decay=wd, nesterov=True
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and str(device).startswith("cuda")))

    # --- run banner ---
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
        "act_bits": act_bits,
        "scale_op": scale_op,
        "first_last_float": first_last_float,
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "cuda_mem": _cuda_mem()
    })

    best = 0.0
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        student.train()
        warmup = (epoch <= 20)
        if warmup: teacher.train()
        else: teacher.eval()

        # meters
        m_loss, m_ce, m_kd, m_hint = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        m_time, m_data = AvgMeter(), AvgMeter()
        start = time.time()

        iterator = enumerate(train_loader, 1)
        if _HAS_TQDM:
            bar = tqdm(iterator, total=len(train_loader), ncols=110,
                       desc=f"epoch {epoch:03d}/{epochs} | lr {_get_lr(opt):.4f}", leave=False)
        else:
            bar = iterator

        end = time.time()
        for step, batch in bar:
            m_data.add(time.time() - end)
            x, y = batch
            x = x.to(device, non_blocking=True)
            x_t = renorm(x)  # teacher view

            # support mixup/cutmix label tuples
            is_mix = isinstance(y, tuple)
            if is_mix:
                y_a, y_b, lam = y
                y_a, y_b = y_a.to(device), y_b.to(device)
            else:
                y = y.to(device)

            opt.zero_grad(set_to_none=True)
            if warmup:
                teach_opt.zero_grad(set_to_none=True)

            use_amp = bool(amp and str(device).startswith("cuda"))
            with torch.amp.autocast("cuda", enabled=use_amp):
                z_t = teacher(x_t)
                z_s = student(x)

                # CE
                if is_mix:
                    loss_ce = lam * ce(z_s, y_a) + (1 - lam) * ce(z_s, y_b)
                else:
                    loss_ce = ce(z_s, y)

                # KD
                loss_kd = kd(z_s, z_t)

                # Hints
                loss_hint = 0.0
                if alpha_hint > 0:
                    for n in hint_points:
                        fs = s_feats[n].float()
                        ft = t_feats[n].float()
                        loss_hint = loss_hint + hint(n, fs, ft)

                loss = (1.0 - alpha_kd) * loss_ce + alpha_kd * loss_kd + alpha_hint * loss_hint

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if warmup:
                teach_opt.step()

            bs = x.size(0)
            m_loss.add(loss.item(), bs)
            m_ce.add(loss_ce.item(), bs)
            m_kd.add(loss_kd.item(), bs)
            m_hint.add(float(loss_hint), bs)

            # timing
            m_time.add(time.time() - end)
            end = time.time()

            if _HAS_TQDM:
                bar.set_postfix({
                    "loss": f"{m_loss.avg:.4f}",
                    "CE": f"{m_ce.avg:.4f}",
                    "KD": f"{m_kd.avg:.4f}",
                    "Hint": f"{m_hint.avg:.4f}",
                    "mem": _cuda_mem()
                })

        # end epoch
        sched.step()
        if warmup: teach_sched.step()

        epoch_time = time.time() - start
        imgs = len(train_loader.dataset)
        ips = imgs / epoch_time if epoch_time>0 else float("nan")

        top1 = eval_top1(student, val_loader, device=device, amp=amp, desc=f"val e{epoch:03d}")
        is_best = top1 > best
        if is_best:
            best = top1
            best_model = student
            ckpt = f"{out_dir}/bit_resnet18_c100_kd_best.pt"
            torch.save({"model": student.state_dict(), "top1": best}, ckpt)
            print(f"✓ saved {ckpt} (top1={best:.2f})")
            # export frozen ternary (PoT)
            tern = convert_to_ternary_p2(copy.deepcopy(student)).cpu().eval()
            tpath = f"{out_dir}/bit_resnet18_c100_kd_ternary.pt"
            torch.save({"model": tern.state_dict(), "top1": best}, tpath)
            print(f"✓ exported ternary PoT → {tpath}")

        # neat epoch summary
        print(
            f"[{epoch:03d}/{epochs}] "
            f"loss {m_loss.avg:.4f} (CE {m_ce.avg:.4f} | KD {m_kd.avg:.4f} | Hint {m_hint.avg:.4f}) | "
            f"top1 {top1:.2f} | best {best:.2f} | "
            f"lr {_get_lr(opt):.4f} | warmup {str(warmup):<5} | "
            f"time {epoch_time:.1f}s | {ips:.0f} img/s | cuda { _cuda_mem() }"
        )

    # cleanup hooks
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
        epochs=200, batch_size=128,
        lr=0.2, wd=5e-4, label_smoothing=0.1,
        alpha_kd=0.7, alpha_hint=0.05, T=4.0,
        amp=True, act_bits=None, scale_op="mean",
        first_last_float=True
    )
