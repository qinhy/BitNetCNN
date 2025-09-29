# cifar100_kd_lightning.py
from functools import partial
import os, math, copy, argparse
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet, BasicBlock
from huggingface_hub import hf_hub_download

# --- Lightning ---
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics.classification import MulticlassAccuracy

# --- your bit modules ---
from BitNetCNN import Bit, convert_to_ternary_p2

EPS = 1e-12

# ----------------------------
# BitResNet-18 (CIFAR stem)
# ----------------------------
class BasicBlockBit(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, scale_op="median"):
        super().__init__()
        self.conv1 = Bit.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=True, scale_op=scale_op)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = Bit.Conv2d(planes, planes, 3, stride=1, padding=1, bias=True, scale_op=scale_op)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)

class BitResNetCIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=100, scale_op="median", in_ch=3):
        super().__init__()
        self.inplanes = 64
        self.stem = nn.Sequential(
            Bit.Conv2d(in_ch, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True, scale_op=scale_op),
            nn.BatchNorm2d(self.inplanes),
            nn.SiLU(inplace=True),
        )
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, scale_op=scale_op)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, scale_op=scale_op)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, scale_op=scale_op)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, scale_op=scale_op)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            Bit.Linear(512, num_classes, bias=True, scale_op=scale_op)
        )

    def _make_layer(self, block, planes, blocks, stride, scale_op):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Bit.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=True, scale_op=scale_op),
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
    def forward(self, name, f_s, f_t):
        f_s = F.adaptive_avg_pool2d(f_s, f_t.shape[-2:])
        c_s, c_t = f_s.shape[1], f_t.shape[1]
        if name not in self.proj or \
           self.proj[name].in_channels != c_s or self.proj[name].out_channels != c_t:
            self.proj[name] = nn.Conv2d(c_s, c_t, kernel_size=1, bias=True).to(f_s.device)
        f_s = self.proj[name](f_s)
        return F.smooth_l1_loss(f_s, f_t.detach())

class SaveOutputHook:
    """Picklable forward hook that stores outputs into a dict under a given key."""
    __slots__ = ("store", "key")
    def __init__(self, store: dict, key: str):
        self.store = store
        self.key = key
    def __call__(self, module, module_in, module_out):
        self.store[self.key] = module_out

def make_feature_hooks(module: nn.Module, names, store: dict):
    """Register picklable forward hooks; returns list of handles."""
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
    lam = 1.0
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
        collate = partial(
            mix_collate,
            aug_cutmix=self.aug_cutmix,
            aug_mixup=self.aug_mixup,
            alpha=self.alpha,
        )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate,
        )


    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=256, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True if self.num_workers > 0 else False)

# ----------------------------
# Teacher: ResNet-18 (CIFAR stem) from HF
# ----------------------------
class ResNet18CIFAR(ResNet):
    def __init__(self, num_classes=100):
        super().__init__(block=BasicBlock, layers=[2,2,2,2], num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.Identity()
        
def make_resnet18_cifar_teacher_from_hf(device="cuda"):
    model = ResNet18CIFAR(num_classes=100)
    ckpt_path = hf_hub_download(repo_id="edadaltocg/resnet18_cifar100", filename="pytorch_model.bin")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:   print(f"[teacher] Missing keys: {missing}")
    if unexpected: print(f"[teacher] Unexpected keys: {unexpected}")
    return model.eval().to(device)

# ----------------------------
# LightningModule: KD + hints + ternary eval/export
# ----------------------------
class LitBitResNetKD(pl.LightningModule):
    def __init__(self, lr, wd, epochs, label_smoothing=0.1,
                 alpha_kd=0.7, alpha_hint=0.05, T=4.0, scale_op="median",
                 amp=True, export_dir="./checkpoints_c100"):
        super().__init__()
        self.save_hyperparameters(ignore=['self._t_feats','self._s_feats',
                                          'self._t_handles','self._s_handles','self.teacher',
                                          'self._ternary_snapshot'])
        self.scale_op = scale_op
        self.student = BitResNetCIFAR(BasicBlockBit, [2,2,2,2], 100, scale_op)
        self.teacher = make_resnet18_cifar_teacher_from_hf(device="cpu")  # will move in setup
        for p in self.teacher.parameters(): p.requires_grad_(False)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing).eval()
        self.kd = KDLoss(T=T).eval()
        self.hint = AdaptiveHintLoss().eval()
        self.hint_points = ["layer1", "layer2", "layer3", "layer4"]
        self.acc_fp = MulticlassAccuracy(num_classes=100).eval()
        self.acc_tern = MulticlassAccuracy(num_classes=100).eval()
        self.best_tern = -1.0
        self._ternary_snapshot = None
        self._t_feats = {}
        self._s_feats = {}

    def setup(self, stage=None):
        # move teacher to device once strategy is set
        self.teacher = self.teacher.to(self.device).eval()
        # register hooks AFTER moving
        self._t_feats, self._s_feats = {}, {}
        self._t_handles = make_feature_hooks(self.teacher, self.hint_points, self._t_feats)
        self._s_handles = make_feature_hooks(self.student, self.hint_points, self._s_feats)


    def teardown(self, stage=None):
        # clean hooks
        for h in getattr(self, "_t_handles", []): 
            try: h.remove()
            except: pass
        for h in getattr(self, "_s_handles", []):
            try: h.remove()
            except: pass

    def forward(self, x):  # student forward
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
        clone = BitResNetCIFAR(BasicBlockBit, [2,2,2,2], 100, self.scale_op)
        clone.load_state_dict(self.student.state_dict(), strict=True)
        clone = convert_to_ternary_p2(clone)
        return clone.eval().to(self.device)
    
    def on_validation_epoch_start(self):
        print()
        # build ternary PoT snapshot for this epoch
        self._ternary_snapshot = self._clone_student()

    def training_step(self, batch, batch_idx):
        x, y = batch
        is_mix = isinstance(y, tuple)
        if is_mix:
            y_a, y_b, lam = y
        use_amp = bool(self.hparams.amp and "cuda" in str(self.device))

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                z_t = self.teacher(x)

        with torch.amp.autocast("cuda", enabled=use_amp):
            z_s = self.student(x)
            if is_mix:
                loss_ce = lam * self.ce(z_s, y_a) + (1 - lam) * self.ce(z_s, y_b)
            else:
                loss_ce = self.ce(z_s, y)

            # KD in fp32 for stability
            with torch.amp.autocast("cuda", enabled=False):
                loss_kd = self.kd(z_s.float(), z_t.float())

            loss_hint = 0.0
            if self.hparams.alpha_hint > 0:
                for n in self.hint_points:
                    if (n in self._s_feats) and (n in self._t_feats):
                        loss_hint = loss_hint + self.hint(n, self._s_feats[n].float(), self._t_feats[n].float())

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
                        
            # logits_fp = self.teacher(x)
            # acc_fp = self.acc_fp(logits_fp.softmax(1), y)
            self.log("val/t_acc_fp", 0.750, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            list(self.student.parameters()) + list(self.hint.parameters()),
            lr=self.hparams.lr, momentum=0.9, weight_decay=self.hparams.wd, nesterov=True
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "monitor": "val/acc_tern",
            },
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

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: LitBitResNetKD):
        metrics = trainer.callback_metrics
        if self.monitor not in metrics: return
        current = metrics[self.monitor].item()
        if self._is_better(current, self.best):
            self.best = current
            # save FP student
            best_fp = copy.deepcopy(pl_module.student).cpu().eval()
            fp_path = os.path.join(self.out_dir, "bit_resnet18_c100_kd_best_fp.pt")
            torch.save({"model": best_fp.state_dict(), "acc_tern": current}, fp_path)
            pl_module.print(f"✓ saved {fp_path} (val/acc_tern={current*100:.2f}%)")
            # save ternary PoT export
            tern = convert_to_ternary_p2(copy.deepcopy(best_fp)).cpu().eval()
            tern_path = os.path.join(self.out_dir, "bit_resnet18_c100_kd_ternary.pt")
            torch.save({"model": tern.state_dict(), "acc_tern": current}, tern_path)
            pl_module.print(f"✓ exported ternary PoT → {tern_path}")

# ----------------------------
# CLI / main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--out",  type=str, default="./ckpt_c100_kd")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--alpha-kd", type=float, default=0.7)
    p.add_argument("--alpha-hint", type=float, default=0.05)
    p.add_argument("--T", type=float, default=4.0)
    p.add_argument("--scale-op", type=str, default="median", choices=["mean","median"])
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

    lit = LitBitResNetKD(
        lr=args.lr, wd=args.wd, epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        alpha_kd=args.alpha_kd, alpha_hint=args.alpha_hint, T=args.T,
        scale_op=args.scale_op, amp=args.amp, export_dir=args.out
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
