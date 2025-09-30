# ===============================================
# CIFAR-100 KD (PyTorch Lightning):
#   MobileNetV2 (teacher, torch.hub)
#   -> BitMobileNetV2 (student, Bit.Conv2d/Bit.Linear)
# ===============================================
import os, math, copy, argparse
from functools import partial

import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Lightning ---
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics.classification import MulticlassAccuracy

# --- your bit modules ---
from BitNetCNN import Bit, convert_to_ternary

EPS = 1e-12

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

class ConvBNActBit(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, groups=1, scale_op="median", act="silu"):
        super().__init__()
        self.conv = Bit.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False,
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
    def __init__(self, inp, oup, stride, expand_ratio, scale_op="median", act="silu"):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNActBit(inp, hidden_dim, 1, 1, 0, scale_op=scale_op, act=act))
        layers.append(ConvBNActBit(hidden_dim, hidden_dim, 3, stride, 1,
                                   groups=hidden_dim, scale_op=scale_op, act=act))
        layers.append(Bit.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False, scale_op=scale_op))
        layers.append(nn.BatchNorm2d(oup))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connect:
            out = x + out
        return out

class BitMobileNetV2(nn.Module):
    """
    CIFAR-friendly BitMobileNetV2:
      - 3x3 s=1 stem
      - MobileNetV2 setting (CIFAR-sized)
      - Collects stage-end names in self.hint_names for hooks
    """
    def __init__(self, num_classes=100, width_mult=1.0, round_nearest=8,
                 scale_op="median", in_ch=3, act="silu", last_channel_override=None):
        super().__init__()
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

        self.stem = ConvBNActBit(in_ch, input_channel, k=3, s=1, p=1, scale_op=scale_op, act=act)

        features = []
        self.hint_names = []
        for t, c, n, s in setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidualBit(input_channel, output_channel, stride,
                                        expand_ratio=t, scale_op=scale_op, act=act)
                )
                input_channel = output_channel
            self.hint_names.append(f"features.{len(features)-1}")
        self.features = nn.Sequential(*features)

        self.head_conv  = ConvBNActBit(input_channel, last_channel, k=1, s=1, p=0,
                                       scale_op=scale_op, act=act)
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

def bit_mobilenetv2_cifar(num_classes=100, width_mult=1.0, scale_op="median",
                          in_ch=3, act="silu", last_channel_override=None):
    return BitMobileNetV2(num_classes=num_classes,
                          width_mult=width_mult,
                          scale_op=scale_op,
                          in_ch=in_ch,
                          act=act,
                          last_channel_override=last_channel_override)

# ----------------------------
# KD losses & feature hints
# ----------------------------
class KDLoss(nn.Module):
    def __init__(self, T=4.0): super().__init__(); self.T=T
    def forward(self, z_s, z_t):
        T = self.T
        return F.kl_div(F.log_softmax(z_s/T,1), F.softmax(z_t/T,1), reduction="batchmean") * (T*T)

class AdaptiveHintLoss(nn.Module):
    """Learnable 1x1 per hint; auto matches spatial size then SmoothL1."""
    def __init__(self):
        super().__init__()
        self.proj = nn.ModuleDict()
    def _safe(self, name: str) -> str:
        return name.replace(".", "_")
    def forward(self, name, f_s, f_t):
        f_s = F.adaptive_avg_pool2d(f_s, f_t.shape[-2:])
        c_s, c_t = f_s.shape[1], f_t.shape[1]
        key = self._safe(name)
        if key not in self.proj or \
           self.proj[key].in_channels != c_s or self.proj[key].out_channels != c_t:
            self.proj[key] = nn.Conv2d(c_s, c_t, kernel_size=1, bias=True).to(f_s.device)
        f_s = self.proj[key](f_s)
        return F.smooth_l1_loss(f_s, f_t.detach())

class SaveOutputHook:
    __slots__ = ("store", "key")
    def __init__(self, store: dict, key: str):
        self.store = store; self.key = key
    def __call__(self, module, module_in, module_out):
        self.store[self.key] = module_out

def make_feature_hooks(module: nn.Module, names, store: dict):
    handles = []
    name_set = set(names)
    for n, sub in module.named_modules():
        if n in name_set:
            handles.append(sub.register_forward_hook(SaveOutputHook(store, n)))
    return handles

# ----------------------------
# CIFAR-100 DataModule (mixup/cutmix optional)
# ----------------------------
def mix_collate(batch, *, aug_cutmix: bool, aug_mixup: bool, alpha: float):
    xs, ys = zip(*batch)
    x = torch.stack(xs); y = torch.tensor(ys)
    if not (aug_cutmix or aug_mixup):
        return x, y
    import random
    if aug_cutmix and random.random() < 0.5:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        idx = torch.randperm(x.size(0))
        h, w = x.size(2), x.size(3)
        rx, ry = torch.randint(w, (1,)).item(), torch.randint(h, (1,)).item()
        rw = int(w * math.sqrt(1 - lam)); rh = int(h * math.sqrt(1 - lam))
        x1, y1 = max(rx - rw//2, 0), max(ry - rh//2, 0)
        x2, y2 = min(rx + rw//2, w), min(ry + rh//2, h)
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        return x, (y, y[idx], lam)
    else:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        idx = torch.randperm(x.size(0))
        x = lam * x + (1 - lam) * x[idx]
        return x, (y, y[idx], lam)

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 4,
                 aug_mixup: bool = False, aug_cutmix: bool = False, alpha: float = 1.0):
        super().__init__()
        self.data_dir, self.batch_size, self.num_workers = data_dir, batch_size, num_workers
        self.aug_mixup, self.aug_cutmix, self.alpha = aug_mixup, aug_cutmix, alpha

    def setup(self, stage=None):
        mean = (0.5071,0.4867,0.4408); std=(0.2675,0.2565,0.2761)
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.train_ds = datasets.CIFAR100(root=self.data_dir, train=True,  download=True, transform=train_tf)
        self.val_ds   = datasets.CIFAR100(root=self.data_dir, train=False, download=True, transform=val_tf)

    def train_dataloader(self):
        collate = partial(mix_collate, aug_cutmix=self.aug_cutmix, aug_mixup=self.aug_mixup, alpha=self.alpha)
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True,
                          persistent_workers=True if self.num_workers > 0 else False,
                          collate_fn=collate)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=256, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True if self.num_workers > 0 else False)

# ----------------------------
# Teacher: MobileNetV2 from torch.hub
# ----------------------------
def make_mobilenetv2_teacher_from_hub(variant="cifar100_mobilenetv2_x1_4", device="cuda"):
    teacher = torch.hub.load("chenyaofo/pytorch-cifar-models", variant, pretrained=True)
    return teacher.to(device).eval()

def _choose_teacher_hint_points(teacher: nn.Module, n_points: int):
    # collect top-level "features.{i}" modules if present; pick evenly spaced n_points
    candidates = sorted([n for n,_ in teacher.named_modules() if n.count(".")==1 and n.startswith("features.")],
                        key=lambda x: int(x.split(".")[1]) if x.split(".")[1].isdigit() else 0)
    if n_points <= 0 or len(candidates) == 0:
        return []
    if len(candidates) >= n_points:
        step = len(candidates)/n_points
        chosen = [candidates[int(round(step*(i+1))-1)] for i in range(n_points)]
        return chosen
    # fallback to last few
    return candidates[-n_points:]

# ----------------------------
# LightningModule: KD + hints + ternary eval/export
# ----------------------------
class LitBitMBv2KD(pl.LightningModule):
    def __init__(self, lr, wd, epochs, label_smoothing=0.1,
                 alpha_kd=0.7, alpha_hint=0.05, T=4.0, scale_op="median",
                 width_mult=1.0, amp=True, teacher_variant="cifar100_mobilenetv2_x1_4",
                 export_dir="./checkpoints_c100_mbv2"):
        super().__init__()
        self.save_hyperparameters(ignore=['teacher','_t_feats','_s_feats','_t_handles','_s_handles','_ternary_snapshot'])
        self.scale_op = scale_op
        self.student = bit_mobilenetv2_cifar(num_classes=100, width_mult=width_mult, scale_op=scale_op)
        self.teacher = None
        if alpha_kd>0:
            self.teacher = make_mobilenetv2_teacher_from_hub(teacher_variant, device="cpu")  # move in setup
            for p in self.teacher.parameters(): p.requires_grad_(False)

        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing).eval()
        self.kd = KDLoss(T=T).eval()
        self.hint = AdaptiveHintLoss().eval()
        self.s_hint_points = list(self.student.hint_names)  # e.g., ["features.0", "features.2", ...]
        self.t_hint_points = []  # decided in setup by inspecting teacher
        self.acc_fp = MulticlassAccuracy(num_classes=100).eval()
        self.acc_tern = MulticlassAccuracy(num_classes=100).eval()
        self._ternary_snapshot = None
        self._t_feats = {}
        self._s_feats = {}

    def setup(self, stage=None):
        if self.teacher:
            self.teacher = self.teacher.to(self.device).eval()
            # choose teacher hint points to match number of student points
            self.t_hint_points = _choose_teacher_hint_points(self.teacher, len(self.s_hint_points))
            # register hooks
            self._t_feats, self._s_feats = {}, {}
            self._t_handles = make_feature_hooks(self.teacher, self.t_hint_points, self._t_feats)
            self._s_handles = make_feature_hooks(self.student, self.s_hint_points, self._s_feats)

    def teardown(self, stage=None):
        for h in getattr(self, "_t_handles", []):
            try: h.remove()
            except: pass
        for h in getattr(self, "_s_handles", []):
            try: h.remove()
            except: pass

    def forward(self, x):
        return self.student(x)

    def on_fit_start(self):
        n_params = sum(p.numel() for p in self.student.parameters())
        acc_name = self.trainer.accelerator.__class__.__name__
        strategy_name = self.trainer.strategy.__class__.__name__
        num_devices = getattr(self.trainer, "num_devices", None) or len(self.trainer.devices or [])
        dev_str = str(self.device)
        cuda_name = ""
        if torch.cuda.is_available() and "cuda" in dev_str:
            try: cuda_name = f" | CUDA: {torch.cuda.get_device_name(self.device.index or 0)}"
            except: pass
        self.print(f"Model params: {n_params/1e6:.2f}M | Accelerator: {acc_name} | Devices: {num_devices} | Strategy: {strategy_name} | Device: {dev_str}{cuda_name}")

    @torch.no_grad()
    def _clone_student(self):
        clone = bit_mobilenetv2_cifar(num_classes=100, width_mult=self.hparams.width_mult, scale_op=self.scale_op)
        clone.load_state_dict(self.student.state_dict(), strict=True)
        clone = convert_to_ternary(clone)
        return clone.eval().to(self.device)

    def on_validation_epoch_start(self):
        print()
        self._ternary_snapshot = self._clone_student()

    def training_step(self, batch, batch_idx):
        x, y = batch
        is_mix = isinstance(y, tuple)
        if is_mix:
            y_a, y_b, lam = y
        use_amp = bool(self.hparams.amp and "cuda" in str(self.device))

        with torch.amp.autocast("cuda", enabled=use_amp):
            z_s = self.student(x)

            # CE
            if is_mix:
                loss_ce = lam * self.ce(z_s, y_a) + (1 - lam) * self.ce(z_s, y_b)
            else:
                loss_ce = self.ce(z_s, y)

            # KD (compute in fp32 for stability)
            loss_kd = 0.0
            if self.hparams.alpha_kd>0:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    z_t = self.teacher(x)
                with torch.amp.autocast("cuda", enabled=False):
                    loss_kd = self.kd(z_s.float(), z_t.float())

            # Hints
            loss_hint = 0.0
            if self.hparams.alpha_hint>0 and len(self.s_hint_points)>0 and len(self.t_hint_points)>0:
                for s_name, t_name in zip(self.s_hint_points, self.t_hint_points):
                    if (s_name in self._s_feats) and (t_name in self._t_feats):
                        loss_hint = loss_hint + self.hint(s_name, self._s_feats[s_name].float(), self._t_feats[t_name].float())

            loss = (1.0 - self.hparams.alpha_kd) * loss_ce + self.hparams.alpha_kd * loss_kd + self.hparams.alpha_hint * loss_hint

        self.log_dict({
            "train/loss": loss,
            "train/ce": loss_ce,
            "train/kd": loss_kd,
            "train/hint": torch.as_tensor(loss_hint, device=self.device, dtype=torch.float32),
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            logits_fp = self.student(x)
            acc_fp = self.acc_fp(logits_fp.softmax(1), y)
            self.log("val/acc_fp", acc_fp, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

            if self._ternary_snapshot is not None:
                logits_t = self._ternary_snapshot(x)
                acc_t = self.acc_tern(logits_t.softmax(1), y)
                self.log("val/acc_tern", acc_t, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

            if self.hparams.alpha_kd>0:
                # optional: show a static teacher acc estimate as in the other script
                self.log("val/t_acc_fp", 0.760, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            list(self.student.parameters()) + list(self.hint.parameters()),
            lr=self.hparams.lr, momentum=0.9, weight_decay=self.hparams.wd, nesterov=True
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch", "monitor": "val/acc_tern"},
        }

# ----------------------------
# Export callback (save best FP & ternary)
# ----------------------------
class ExportBestTernary(Callback):
    def __init__(self, out_dir: str, monitor: str = "val/acc_tern", mode: str = "max"):
        super().__init__()
        self.out_dir, self.monitor, self.mode = out_dir, monitor, mode
        self.best = None
        os.makedirs(out_dir, exist_ok=True)

    def _is_better(self, current, best):
        if best is None: return True
        return (current > best) if self.mode == "max" else (current < best)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: LitBitMBv2KD):
        metrics = trainer.callback_metrics
        if self.monitor not in metrics: return
        current = metrics[self.monitor].item()
        if self._is_better(current, self.best):
            self.best = current
            # save FP student
            best_fp = copy.deepcopy(pl_module.student).cpu().eval()
            fp_path = os.path.join(self.out_dir, "bit_mbv2_c100_kd_best_fp.pt")
            torch.save({"model": best_fp.state_dict(), "acc_tern": current}, fp_path)
            pl_module.print(f"✓ saved {fp_path} (val/acc_tern={current*100:.2f}%)")
            # save ternary PoT export
            tern = convert_to_ternary(copy.deepcopy(best_fp)).cpu().eval()
            tern_path = os.path.join(self.out_dir, "bit_mbv2_c100_kd_ternary.pt")
            torch.save({"model": tern.state_dict(), "acc_tern": current}, tern_path)
            pl_module.print(f"✓ exported ternary PoT → {tern_path}")

# ----------------------------
# CLI / main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--out",  type=str, default="./ckpt_c100_kd_mbv2")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--alpha-kd", type=float, default=0.3)   # set >0 to enable KD
    p.add_argument("--alpha-hint", type=float, default=0.05) # set >0 to enable feature hints
    p.add_argument("--T", type=float, default=4.0)
    p.add_argument("--scale-op", type=str, default="median", choices=["mean","median"])
    p.add_argument("--width-mult", type=float, default=1.4)
    p.add_argument("--teacher-variant", type=str, default="cifar100_mobilenetv2_x1_4")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--mixup", action="store_true")
    p.add_argument("--cutmix", action="store_true")
    p.add_argument("--mix-alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    dm = CIFAR100DataModule(
        data_dir=args.data, batch_size=args.batch_size, num_workers=4,
        aug_mixup=args.mixup, aug_cutmix=args.cutmix, alpha=args.mix_alpha
    )

    lit = LitBitMBv2KD(
        lr=args.lr, wd=args.wd, epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        alpha_kd=args.alpha_kd, alpha_hint=args.alpha_hint, T=args.T,
        scale_op=args.scale_op, width_mult=args.width_mult,
        amp=args.amp, teacher_variant=args.teacher_variant,
        export_dir=args.out
    )

    os.makedirs(args.out, exist_ok=True)
    logger = CSVLogger(save_dir=args.out, name="logs")
    ckpt_cb = ModelCheckpoint(monitor="val/acc_tern", mode="max", save_top_k=1, save_last=True)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    export_cb = ExportBestTernary(args.out, monitor="val/acc_tern", mode="max")
    callbacks = [ckpt_cb, lr_cb, export_cb]

    accelerator = "cpu" if args.cpu else "auto"
    precision = "16-mixed" if args.amp else "32-true"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        deterministic=False,
    )

    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)

if __name__ == "__main__":
    main()
