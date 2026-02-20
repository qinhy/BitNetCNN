import random
from pydanticV2_argparse import ArgumentParser
import torch

from bitlayers.dinov3.models.vision_transformer import DinoVisionTransformer, DinoVisionTransformerTRM, vit_micro
from trainer import AccelTrainer, CommonTrainConfig, LitBit, LitBitConfig, Metrics
import torch.nn as nn

# Import from common_utils
from common_utils import *
from dataset import DataModuleConfig

# ----------------------------
# Simple BitNet block & model
# ----------------------------
class TinyViT(nn.Module):
    def __init__(self, num_classes=10, drop_p=0.1, bias=True, scale_op="median"):
        super().__init__()
        self.back = vit_micro(
            img_size=28,      # <-- important for MNIST
            patch_size=7,
            in_chans=1,
            cls=DinoVisionTransformerTRM,
            drop_path_rate=0.0,  # <-- good to force off for MNIST
        )
        self.back.init_weights()   # <-- VERY important unless vit_micro already does this

        self.head = nn.Sequential(
            nn.Dropout(drop_p),
            nn.Linear(self.back.embed_dim, num_classes, bias=bias)
        )
        nn.init.normal_(self.head[1].weight, std=0.02)
        if bias:
            nn.init.zeros_(self.head[1].bias)

        self.num_classes = num_classes
        self.drop_p = drop_p
        self.bias = bias

    def forward(self, x, solution=None, latent=None, n=1, T=1, full_output=False):
        out = self.back.forward_features(
            x,
            solution=solution,
            latent=latent,
            n=n,
            T=T,
            # track_latent_grads=True,  # if your patched TRM class supports it
        )
        y = out["x_prenorm"]   # solution state
        z = out["z_latent"]    # latent state
        logits = self.head(out["x_norm_clstoken"])
        if full_output:
            return logits, y, z
        else:
            return logits

    def clone(self):
        return self.__class__(self.num_classes, self.drop_p, self.bias)


# ----------------------------
# LightningModule wrapper using LitBit
# ----------------------------
class LitNetViT(LitBit):
    def training_step(self, batch, batch_idx):
        if type(self.student) is DinoVisionTransformer:
            return super().training_step(batch, batch_idx)

        x, y_answer = batch
        logd = {}

        loss_ce = 0.0
        solution, latent = None, None

        student: TinyViT = self.student

        # Start simple but actually use TRM
        N_supervision = random.randint(1,4)   # try 1 first, then 4
        for s in range(N_supervision):
            T = random.randint(1,2)               # outer recursion passes
            n = random.randint(1,2)               # latent refinement steps
            logits, solution, latent = student(
                x,
                solution=solution,
                latent=latent,
                n=n,
                T=T,
                full_output=True
            )
            loss_ce = loss_ce + self.ce_hard(logits, y_answer)

            # Important if doing deep supervision (>1 step):
            if s < N_supervision - 1:
                solution = solution.detach()
                latent = latent.detach()

        loss_ce = loss_ce / N_supervision
        logd["train/loss_ce"] = loss_ce.detach()

        y_idx = y_answer.argmax(dim=1) if y_answer.ndim == 2 else y_answer
        acc = (logits.argmax(dim=1) == y_idx).float().mean()
        logd["train/acc"] = acc.detach()

        return Metrics(loss=loss_ce, metrics=logd)
    
    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Metrics:
        if type(self.student) is DinoVisionTransformer:
            return super().validation_step(batch, batch_idx)
        
        x, y = batch
        y_idx = y.argmax(dim=1) if y.ndim == 2 else y 

        z_fp = self.student(x)
        z_tern = self._ternary_snapshot(x)

        vloss = F.cross_entropy(z_fp, y_idx.long())
        acc_fp = (z_fp.argmax(dim=1) == y_idx).float().mean()
        acc_tern = (z_tern.argmax(dim=1) == y_idx).float().mean()
        metrics = {"val/acc_fp": acc_fp, "val/acc_tern": acc_tern}
        return Metrics(loss=vloss, metrics=metrics)
    
    def configure_optimizers(self,trainer=None):
        # Use AdamW for MNIST instead of SGD
        opt = torch.optim.AdamW(
            self.configure_optimizer_params(),
            lr=self.lr, weight_decay=self.wd
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=self.lr*0.01)
        return opt, sched, "epoch"

# ----------------------------
# CLI / main
# ----------------------------
class Config(CommonTrainConfig):
    data:str="./data"
    dataset_name:str='mnist'
    export_dir:Optional[str]="./ckpt_tViT_mnist"
    epochs:int=50
    batch_size:int=1024
    lr: float = 3e-4
    wd:float=1e-4
    label_smoothing:float=0.0
    amp: bool = True

def main():
    parser = ArgumentParser(model=Config)
    args = parser.parse_typed_args()
    
    dm = DataModuleConfig.model_validate(args.model_dump())
    config = LitBitConfig.model_validate(args.model_dump())
    config.dataset = dm.model_copy()    
    config.student = TinyViT(num_classes=dm.num_classes,scale_op=config.scale_op)    
    config.model_name='vit'
    config.model_size='tiny'
    lit = LitNetViT(config)

    trainer = AccelTrainer(
        max_epochs=args.epochs,
        mixed_precision="bf16" if args.amp else "no",
        gradient_accumulation_steps=1,
        log_every_n_steps=10,
        enable_ema=True
    )
    trainer.fit(lit, datamodule=dm.build())


if __name__ == "__main__":
    main()
