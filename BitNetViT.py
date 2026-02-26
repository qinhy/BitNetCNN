import random
from pydanticV2_argparse import ArgumentParser
import torch

from bitlayers.dinov3.models.vision_transformer import DinoVisionTransformer, DinoVisionTransformerTRM, vit_femto, vit_micro, vit_tiny
from trainer import AccelTrainer, CommonTrainConfig, LitBit, LitBitConfig, Metrics
import torch.nn as nn

# Import from common_utils
from common_utils import *
from dataset import DataModuleConfig

# ----------------------------
# Simple BitNet block & model
# ----------------------------
class TinyViT(nn.Module):
    def __init__(
        self, model_size=vit_femto, num_classes=10, drop_p=0.1, bias=True, scale_op="median"):
        super().__init__()
        self.model_size = model_size
        self.num_classes = num_classes
        self.drop_p = drop_p
        self.bias = bias
        self.scale_op = scale_op

        self.back = DinoVisionTransformerTRM(
            img_size=28,      # MNIST
            patch_size=4,
            in_chans=1,
            drop_path_rate=0.0, # no drop for recursion
            embed_dim=72,
            depth=3,
            num_heads=3,
            ffn_ratio=2.5,
        )
        self.back.init_weights()

        build_head = lambda num_out:nn.Sequential(
            nn.Dropout(drop_p),
            nn.Linear(self.back.embed_dim, num_out, bias=True),
        )

        # Classification head (output_head in your pseudocode)
        self.head = build_head(num_classes)
        nn.init.normal_(self.head[1].weight, std=0.02)
        nn.init.zeros_(self.head[1].bias)

        # Confidence / correctness head (Q_head in your pseudocode)
        # Produces logits for BCEWithLogitsLoss; shape [B, 1]
        self.q_head = build_head(1)
        nn.init.normal_(self.q_head[1].weight, std=0.02)
        nn.init.zeros_(self.q_head[1].bias)
            
    def forward(self, x, solution=None, latent=None, n=1, T=1, full_output=False):
        if type(self.back) is DinoVisionTransformer:
            out = self.back.forward_features(x)
            y = None
            z = None
        else:
            out = self.back.forward_features(
                x,n=n,T=T,
                solution=solution,
                latent=latent,
                # track_latent_grads=True,  # if your patched TRM class supports it
            )
            y = out["x_prenorm"]   # solution state
            z = out["z_latent"]    # latent state

        # Heads use cls token representation
        cls = out["x_norm_clstoken"]  # shape [B, D]
        y_logits = self.head(cls)     # [B, num_classes]
        q_logits = self.q_head(cls)   # [B, 1] (logits)
        if full_output:
            return y, z, y_logits, q_logits
        else:
            return y_logits
        
    def thinking(self,x,n=1,T=2,max_supervision=16,q_threshold=0.99):
        logits_parts = []
        idx_parts = []

        idx = torch.arange(x.size(0), device=x.device)  # original positions
        solution, latent = None, None

        for step in range(max_supervision):
            solution, latent, logits, q_logits = self.forward(
                x, n=n, T=T,
                solution=solution, latent=latent,
                full_output=True
            )

            q_prob = torch.sigmoid(q_logits).view(-1)
            keep = (q_prob < q_threshold)   # keep running these
            done = ~keep                   # finalize these

            # store finished samples
            if done.any():
                logits_parts.append(logits[done])
                idx_parts.append(idx[done])

            # if all finished, stop
            if not keep.any():
                break

            # otherwise keep only unfinished and continue
            x, solution, latent, idx = x[keep], solution[keep], latent[keep], idx[keep]

        else:
            # hit max_supervision: keep whatever is left from the last step
            logits_parts.append(logits)
            idx_parts.append(idx)

        # restore original order
        logits_all = torch.cat(logits_parts, dim=0)
        idx_all = torch.cat(idx_parts, dim=0)
        return logits_all[idx_all.argsort()]

    def clone(self):
        return self.__class__(self.model_size, self.num_classes, self.drop_p, self.bias)


# ----------------------------
# LightningModule wrapper using LitBit
# ----------------------------
class LitNetViT(LitBit):
    def training_step(self, batch, batch_idx):
        if type(self.student.back) is DinoVisionTransformer:
            return super().training_step(batch, batch_idx)
        x, y_answer = batch        
        x, y_answer = x.to(self.device), y_answer.to(self.device)
        logd = {}

        loss = 0.0
        acc_list = []
        loss_list = []
        q_threshold = 0.99
        solution, latent = None, None

        student: TinyViT = self.student

        # Start simple but actually use TRM
        N_supervision = 16   # try 1 first, then 4
        T = 2            # outer recursion passes
        n = 1            # latent refinement steps
        for s in range(N_supervision):
            solution, latent, logits, q_logits = student(
                x,n=n,T=T,
                solution=solution,latent=latent,
                full_output=True
            )
            # q-target = whether prediction is correct
            pred = logits.argmax(dim=-1)  # [B]
            acc = (pred == y_answer).float()
            q_target = acc.unsqueeze(-1)  # [B,1]

            # Confidence / correctness loss (q_hat are logits)
            loss = self.ce_hard(logits, y_answer) + F.binary_cross_entropy_with_logits(q_logits, q_target)

            # Important if doing deep supervision (>1 step):
            if s < N_supervision - 1:
                self._trainer.accelerator.backward(loss)
                self._trainer.optimizer.step()
                self._trainer.optimizer.zero_grad(set_to_none=True)
                loss = loss.detach()
                loss_list.append(loss.cpu().item()/len(x))
                solution = solution.detach()
                latent = latent.detach()

            # Optional early stop if model is confident enough
            q_prob = torch.sigmoid(q_logits)
            keep_idx = (q_prob < q_threshold).flatten()
            
            x = x[keep_idx]
            y_answer = y_answer[keep_idx]
            solution = solution[keep_idx]
            latent = latent[keep_idx]

            if keep_idx.sum() == 0:
                acc_list.append(acc.flatten())    
                break
            elif keep_idx.sum() > 0 and s == N_supervision - 1:
                acc_list.append(acc.flatten())
            else:
                acc_list.append(acc[~keep_idx].flatten())
 
        logd["train/loss"] = torch.Tensor(loss_list).mean()
        logd["train/acc"] = torch.cat(acc_list, dim=0).mean()
        return Metrics(loss=loss, metrics=logd)
    
    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Metrics:
        if type(self.student.back) is DinoVisionTransformer:
            return super().validation_step(batch, batch_idx)
        
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_idx = y.argmax(dim=1) if y.ndim == 2 else y 

        # z_fp = self.student.thinking(x,n=1,T=2)
        # z_tern = self._ternary_snapshot.thinking(x,n=1,T=2)
        z_fp = self.student(x,n=0,T=1)
        z_tern = self._ternary_snapshot(x,n=0,T=1)

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
    epochs:int=1023
    batch_size:int=1024*4
    num_workers:int=8
    lr:float=3e-4
    wd:float=1e-4
    label_smoothing:float=0.0
    amp:bool=True

def main():
    parser = ArgumentParser(model=Config)
    args = parser.parse_typed_args()
    
    dm = DataModuleConfig.model_validate(args.model_dump())
    config = LitBitConfig.model_validate(args.model_dump())
    config.dataset = dm.model_copy()    
    config.student = TinyViT(model_size=vit_femto, num_classes=dm.num_classes,scale_op=config.scale_op)    
    config.model_name='vit'
    config.model_size='femto'
    lit = LitNetViT(config)

    trainer = AccelTrainer(
        max_epochs=args.epochs,
        mixed_precision="bf16" if args.amp else "no",
        gradient_accumulation_steps=1,
        log_every_n_steps=10,
        enable_ema=0.99**(args.batch_size//128)
    )
    trainer.fit(lit, datamodule=dm.build())


if __name__ == "__main__":
    main()
