# cifar100_convnextv2_lightning.py
from functools import partial
import os, math, copy, argparse
import torch

from BitNetCNN import Bit, convert_to_ternary
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.layers import trunc_normal_, DropPath

# --- Lightning ---
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics.classification import MulticlassAccuracy

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
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = Bit.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = Bit.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = Bit.Linear(4 * dim, dim)
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
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            Bit.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    Bit.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = Bit.Linear(dims[-1], num_classes)

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
    
    @staticmethod
    def convnextv2(size='atto',num_classes=100, drop_path_rate=0.3):
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
        return ConvNeXtV2(**params[size],num_classes=num_classes,drop_path_rate=drop_path_rate)

# ----------------------------
# KD losses & feature hints (unchanged from your pattern)
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
    
    @staticmethod
    def _k(name: str) -> str:
        # bijective mapping so we never collide with real underscores etc.
        # U+2027 (Hyphenation Point) is printable and allowed in names.
        return name.replace('.', '\u2027')

    def forward(self, name, f_s, f_t):
        f_s = F.adaptive_avg_pool2d(f_s, f_t.shape[-2:])
        c_s, c_t = f_s.shape[1], f_t.shape[1]
        k = self._k(name)
        if (k not in self.proj or
            self.proj[k].in_channels != c_s or
            self.proj[k].out_channels != c_t):
            self.proj[k] = nn.Conv2d(c_s, c_t, kernel_size=1, bias=True).to(f_s.device)
        f_s = self.proj[k](f_s)
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
# CIFAR-100 DataModule (unchanged)
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
# Teacher: convnextv2 from HF
# ----------------------------
def make_convnextv2_from_timm(size='pico',num_classes=100, device="cuda", pretrained=True):
    import timm
    m = timm.create_model(f'convnextv2_{size}.fcmae_ft_in1k', pretrained=pretrained)
    return m.eval().to(device)

# ----------------------------
# LightningModule: KD + hints (no ternary)
# ----------------------------
class LitConvNeXtV2KD(pl.LightningModule):
    def __init__(self, lr, wd, epochs, model_size="convnextv2_pico",
                 label_smoothing=0.1, alpha_kd=0.0, alpha_hint=0.0, T=4.0,
                 amp=True, export_dir="./checkpoints_c100_convnextv2", drop_path_rate=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model_size = model_size
        self.drop_path_rate = drop_path_rate
        self.student = ConvNeXtV2.convnextv2(model_size,num_classes=100, drop_path_rate=drop_path_rate)
        self.teacher = None
        if alpha_kd > 0 or alpha_hint > 0:
            self.teacher = make_convnextv2_from_timm(model_size,device="cpu")  # move in setup
            for p in self.teacher.parameters(): p.requires_grad_(False)

        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing).eval()
        self.kd = KDLoss(T=T).eval()
        self.hint = AdaptiveHintLoss().eval()

        # Hook points: after each stage (and stem output)
        self.hint_points = ["stages.0", "stages.1", "stages.2", "stages.3"]

        self.acc_fp = MulticlassAccuracy(num_classes=100).eval()
        self.acc_tern = MulticlassAccuracy(num_classes=100).eval()
        self._t_feats = {}
        self._s_feats = {}

    def setup(self, stage=None):
        if self.teacher:
            # move teacher to device once strategy is set
            self.teacher = self.teacher.to(self.device).eval()
            self._t_handles = make_feature_hooks(self.teacher, self.hint_points, self._t_feats)
            self._s_handles = make_feature_hooks(self.student, self.hint_points, self._s_feats)

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        is_mix = isinstance(y, tuple)
        if is_mix:
            y_a, y_b, lam = y
        use_amp = bool(self.hparams.amp and "cuda" in str(self.device))

        with torch.amp.autocast("cuda", enabled=use_amp):
            z_s = self.student(x)
            if is_mix:
                loss_ce = lam * self.ce(z_s, y_a) + (1 - lam) * self.ce(z_s, y_b)
            else:
                loss_ce = self.ce(z_s, y)

            if self.hparams.alpha_kd > 0 or self.hparams.alpha_hint > 0:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    z_t = self.teacher(x)

            loss_kd = torch.tensor(0.0, device=self.device)
            if self.hparams.alpha_kd > 0:
                with torch.amp.autocast("cuda", enabled=False):
                    loss_kd = self.kd(z_s.float(), z_t.float())

            loss_hint = torch.tensor(0.0, device=self.device)
            if self.hparams.alpha_hint > 0 and self.teacher:
                for n in self.hint_points:
                    if (n in self._s_feats) and (n in self._t_feats):
                        # print(n,self._s_feats[n].shape,self._t_feats[n].shape)
                        loss_hint = loss_hint + self.hint(n, self._s_feats[n].float(), self._t_feats[n].float())
                        # print(n,loss_hint)


            loss = (1.0 - self.hparams.alpha_kd) * loss_ce + self.hparams.alpha_kd * loss_kd + self.hparams.alpha_hint * loss_hint

        self.log_dict({
            "train/loss": loss,
            "train/ce": loss_ce,
            "train/kd": loss_kd,
            "train/hint": loss_hint,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=x.size(0))
        return loss

    @torch.no_grad()
    def _clone_student(self):        
        clone = ConvNeXtV2.convnextv2(self.model_size,num_classes=100, drop_path_rate=self.drop_path_rate)
        clone.load_state_dict(self.student.state_dict(), strict=True)
        clone = convert_to_ternary(clone)
        return clone.eval().to(self.device)
    
    def on_validation_epoch_start(self):
        print()
        # build ternary PoT snapshot for this epoch
        self._ternary_snapshot = self._clone_student()

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
            if self.hparams.alpha_kd>0:
                self.log("val/t_acc_fp", -1.0, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

    def configure_optimizers(self):
        # standard recipe: SGD w/ cosine also works well for ConvNeXt on CIFAR
        opt = torch.optim.SGD(
            self.student.parameters(),
            lr=self.hparams.lr, momentum=0.9, weight_decay=self.hparams.wd, nesterov=True
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "monitor": "val/acc_fp",
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

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: LitConvNeXtV2KD):
        metrics = trainer.callback_metrics
        if self.monitor not in metrics: return
        current = metrics[self.monitor].item()
        if self._is_better(current, self.best):
            self.best = current
            # save FP student
            best_fp = copy.deepcopy(pl_module.student).cpu().eval()
            fp_path = os.path.join(self.out_dir, f"bit_convnextv2_{pl_module.model_size}_c100_kd_best_fp.pt")
            torch.save({"model": best_fp.state_dict(), "acc_tern": current}, fp_path)
            pl_module.print(f"✓ saved {fp_path} (val/acc_tern={current*100:.2f}%)")
            # save ternary PoT export
            tern = convert_to_ternary(copy.deepcopy(best_fp)).cpu().eval()
            tern_path = os.path.join(self.out_dir, f"bit_convnextv2_{pl_module.model_size}_c100_kd_ternary.pt")
            torch.save({"model": tern.state_dict(), "acc_tern": current}, tern_path)
            pl_module.print(f"✓ exported ternary PoT → {tern_path}")

# ----------------------------
# CLI / main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--out",  type=str, default="./ckpt_c100_convnextv2")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.2)  # works well for SGD + cosine on CIFAR
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--alpha-kd", type=float, default=0.0)   # enable to use teacher
    p.add_argument("--alpha-hint", type=float, default=0.1) # enable to use feature hints
    p.add_argument("--T", type=float, default=4.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--mixup", action="store_true")
    p.add_argument("--cutmix", action="store_true")
    p.add_argument("--mix-alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-size", type=str, default="nano", choices=["atto","femto","pico","nano","tiny","base","large","huge"])
    p.add_argument("--drop-path", type=float, default=0.1)
    return p.parse_args()

def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    dm = CIFAR100DataModule(
        data_dir=args.data, batch_size=args.batch_size, num_workers=4,
        aug_mixup=args.mixup, aug_cutmix=args.cutmix, alpha=args.mix_alpha
    )

    lit = LitConvNeXtV2KD(
        lr=args.lr, wd=args.wd, epochs=args.epochs,
        model_size=args.model_size,
        label_smoothing=args.label_smoothing,
        alpha_kd=args.alpha_kd, alpha_hint=args.alpha_hint, T=args.T,
        amp=args.amp, export_dir=args.out, drop_path_rate=args.drop_path
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
