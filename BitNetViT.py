from fractions import Fraction
import random
from typing import Optional, Tuple

from pydanticV2_argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlayers.dinov3.models.vision_transformer import (
    DinoVisionTransformer,
    DinoVisionTransformerTRM,
    vit_femto,
)
from dataset.base import DataSetModule
from trainer import AccelTrainer, CommonTrainConfig, LitBit, LitBitConfig, Metrics, RawFraction

# Project-local imports
from common_utils import *  # noqa: F403
from dataset import DataModuleConfig


# ----------------------------
# TinyViT (TRM backbone + 2 heads)
# ----------------------------
class TinyViT(nn.Module):
    def __init__(
        self,
        model_size=vit_femto,
        num_classes: int = 10,
        drop_p: float = 0.1,
        bias: bool = True,
        scale_op: str = "median",
    ):
        super().__init__()
        self.model_size = model_size
        self.num_classes = num_classes
        self.drop_p = drop_p
        self.bias = bias
        self.scale_op = scale_op

        # MNIST-ish settings
        self.back = DinoVisionTransformerTRM(
            img_size=28,
            patch_size=4,
            in_chans=1,
            drop_path_rate=0.0,  # no drop for recursion
            embed_dim=72,
            depth=3,
            num_heads=3,
            ffn_ratio=2.5,
            no_conv=True,
        )
        self.back.init_weights()

        def build_head(num_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Dropout(self.drop_p),
                Bit.Linear(self.back.embed_dim, num_out, bias=self.bias),
            )

        # Classification head
        self.head = build_head(num_classes)
        nn.init.normal_(self.head[1].weight, std=0.02)
        if self.head[1].bias is not None:
            nn.init.zeros_(self.head[1].bias)

        # Correctness/confidence head (logits for BCEWithLogitsLoss), shape [B, 1]
        self.q_head = build_head(1)
        nn.init.normal_(self.q_head[1].weight, std=0.02)
        if self.q_head[1].bias is not None:
            nn.init.zeros_(self.q_head[1].bias)

        # Tracks best (n, T, supervision) combo
        self.combo = {}

    def forward(
        self,
        x: torch.Tensor,
        solution: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        n: int = 1,
        T: int = 1,
        full_output: bool = False,
    ):
        is_trm = type(self.back) is DinoVisionTransformerTRM

        if is_trm:
            out = self.back.forward_features(
                x, n=n, T=T,
                solution=solution,
                latent=latent,
            )
            y = out.get("x_prenorm", None)  # solution state
            z = out.get("z_latent", None)   # latent state
        else:
            # Base DinoVisionTransformer path
            out = self.back.forward_features(x)
            y, z = None, None

        cls = out["x_norm_clstoken"]     # [B, D]
        y_logits = self.head(cls)        # [B, num_classes]
        q_logits = self.q_head(cls)      # [B, 1]

        if full_output:
            return y, z, y_logits, q_logits
        return y_logits

    def record_combo(self, n: int, T: int, supervision: int, correct_cnt: float, total_cnt: int) -> None:
        cnt, t_cnt = self.combo.get((n, T, supervision), (0.0, 0))
        self.combo[(n, T, supervision)] = (cnt + float(correct_cnt), t_cnt + int(total_cnt))

    def get_best_combo(self) -> Tuple[int, int, int]:
        # Important: validation may run before any training steps
        if not self.combo:
            return (0, 1, 1)

        def acc(item):
            correct, total = item[1]
            return (correct / total) if total > 0 else -1.0

        best_key, _ = max(self.combo.items(), key=acc)
        return best_key

    @torch.no_grad()
    def thinking(
        self,
        x: torch.Tensor,
        n: int = 1,
        T: int = 2,
        max_supervision: int = 16,
        q_threshold: float = 0.99,
    ) -> torch.Tensor:
        """
        Early-exit inference:
        - Run up to max_supervision steps
        - At each step, accept samples with sigmoid(q_logits) >= q_threshold
        - Return logits in original order, shape [B, num_classes]
        """
        logits_parts = []
        idx_parts = []

        idx = torch.arange(x.size(0), device=x.device)
        solution, latent = None, None

        for step in range(max_supervision):
            solution, latent, logits, q_logits = self.forward(
                x, n=n, T=T,
                solution=solution,
                latent=latent,
                full_output=True,
            )

            q_prob = torch.sigmoid(q_logits).view(-1)
            keep = q_prob < q_threshold
            done = ~keep

            if done.any():
                logits_parts.append(logits[done])
                idx_parts.append(idx[done])

            if not keep.any():
                break

            x, solution, latent, idx = x[keep], solution[keep], latent[keep], idx[keep]

        else:
            # Hit max_supervision: keep whatever is left
            logits_parts.append(logits)
            idx_parts.append(idx)

        logits_all = torch.cat(logits_parts, dim=0)
        idx_all = torch.cat(idx_parts, dim=0)
        return logits_all[idx_all.argsort()]

    def clone(self):
        return self.__class__(
            model_size=self.model_size,
            num_classes=self.num_classes,
            drop_p=self.drop_p,
            bias=self.bias,
            scale_op=self.scale_op,
        )


# ----------------------------
# Lightning-ish module wrapper
# ----------------------------
class LitNetViT(LitBit):
    """
    Notes:
    - This keeps your original "truncated deep supervision" behavior:
      intermediate steps do their own optimizer update, and we return ONE final loss
      to the trainer for a final update.
    - To avoid returning a detached loss (which can break backprop), we treat the step
      that finishes the batch as the "final" step and do NOT manually step on it.
    """

    def _manual_step_intermediate(self, loss: torch.Tensor) -> None:
        self._trainer.accelerator.backward(loss)
        self._trainer.optimizer.step()
        self._trainer.optimizer.zero_grad(set_to_none=True)

    def on_train_epoch_start(self, epoch: int):
        self.current_epoch = epoch

    def training_step(self, batch, batch_idx):
        # If not TRM, defer to base
        if type(self.student.back) is DinoVisionTransformer:
            return super().training_step(batch, batch_idx)

        x, y_answer = batch
        x, y_answer = x.to(self.device), y_answer.to(self.device)

        student: TinyViT = self.student

        q_threshold = 0.99

        # Randomize recursion / refinement / supervision depth
        # if self.current_epoch<1:
        #     N_supervision = 1
        #     T = 1
        #     n = 0
        # else:
        N_supervision = 16#random.randint(1, 16)
        T = random.randint(1, 8)
        n = random.randint(0, 4)

        # Track final accuracy on the ORIGINAL batch (even with early exit)
        orig_idx = torch.arange(x.size(0), device=x.device)
        correct_parts = []
        idx_parts = []

        # Track losses for logging only
        step_losses = []

        solution, latent = None, None

        # Work on a shrinking subset
        x_cur, y_cur, idx_cur = x, y_answer, orig_idx

        final_loss = None

        for step in range(N_supervision):
            solution, latent, logits, q_logits = student(
                x_cur, n=n, T=T,
                solution=solution,
                latent=latent,
                full_output=True,
            )

            pred = logits.argmax(dim=-1)
            correct = (pred == y_cur).float()        # [B_cur]
            q_target = correct.unsqueeze(-1)         # [B_cur, 1]

            # Record combo stats on the current subset
            student.record_combo(n, T, step + 1, correct.sum().item(), int(correct.numel()))

            loss = self.ce_hard(logits, y_cur) + F.binary_cross_entropy_with_logits(q_logits, q_target)
            step_losses.append(loss.detach())

            # Early-exit decision on current subset
            q_prob = torch.sigmoid(q_logits).view(-1)
            keep = q_prob < q_threshold
            done = ~keep

            # If we're at the last supervision step, force all remaining samples to be "done"
            if step == N_supervision - 1:
                done = torch.ones_like(done, dtype=torch.bool)
                keep = ~done

            # Save correctness for done samples (in original order later)
            if done.any():
                correct_parts.append(correct[done])
                idx_parts.append(idx_cur[done])

            # If nothing remains, this step is the final step for the batch
            if not keep.any():
                final_loss = loss
                break

            # If there will be another step, do an intermediate optimizer update and truncate states
            # (IMPORTANT: do not do this on the final step, otherwise you'd return a detached loss)
            self._manual_step_intermediate(loss)

            # Truncate state/history for next step
            solution = solution.detach()
            latent = latent.detach()

            # Continue with only the "hard" samples
            x_cur = x_cur[keep]
            y_cur = y_cur[keep]
            idx_cur = idx_cur[keep]
            solution = solution[keep]
            latent = latent[keep]

        # Safety: should always be set
        if final_loss is None:
            final_loss = step_losses[-1].to(self.device)

        # Compute final accuracy over the original batch
        if correct_parts:
            corr_all = torch.cat(correct_parts, dim=0)
            idx_all = torch.cat(idx_parts, dim=0)
            corr_all = corr_all[idx_all.argsort()]
            # acc = corr_all[idx_all.argsort()].mean()
            acc = RawFraction.acc(corr_all)
        else:
            # Extremely unlikely, but keep it defined
            acc = RawFraction(0,0)

        n,T,S = student.get_best_combo()
        logd = {
            "train/loss": torch.stack(step_losses).mean() if step_losses else final_loss.detach(),
            "train/acc": acc,"train/n": n,"train/T": T,"train/S": S,
        }

        # Return ONE final loss so the outer trainer can do the final optimizer step
        return Metrics(loss=final_loss, metrics=logd)

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Metrics:
        if type(self.student.back) is DinoVisionTransformer:
            return super().validation_step(batch, batch_idx)

        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_idx = y.argmax(dim=1) if y.ndim == 2 else y
        
        n, T, S = self.student.get_best_combo()

        z_fp = self.student.thinking(x, n=n, T=T, max_supervision=S, q_threshold=0.99)
        z_tern = self._ternary_snapshot.thinking(x, n=n, T=T, max_supervision=S, q_threshold=0.99)

        vloss = F.cross_entropy(z_fp, y_idx.long())
        acc_fp = RawFraction.acc(z_fp.argmax(dim=1) == y_idx)
        acc_tern = RawFraction.acc(z_tern.argmax(dim=1) == y_idx)

        metrics = {"val/acc_fp": acc_fp, "val/acc_tern": acc_tern}
        return Metrics(loss=vloss, metrics=metrics)

    def configure_optimizers(self, trainer=None):
        # AdamW for MNIST
        self.lr: float = 3e-4
        self.wd: float = 1e-4
        opt = torch.optim.AdamW(
            self.configure_optimizer_params(),
            lr=self.lr,
            weight_decay=self.wd,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=self.lr * 0.01
        )
        return opt, sched, "epoch"

# ----------------------------
# CLI / main
# ----------------------------
class Config(CommonTrainConfig):
    data: str = "./data"
    dataset_name: str = "mnist"
    export_dir: Optional[str] = "./ckpt_tViT_mnist"

    epochs: int = 1023
    batch_size: Union[int,Tuple[int, int]] = (128, 5000)
    num_workers: int = 8

    label_smoothing: float = 0.0
    amp: bool = True


def main():
    parser = ArgumentParser(model=Config)
    args = parser.parse_typed_args()

    dm = DataModuleConfig.model_validate(args.model_dump())
    config = LitBitConfig.model_validate(args.model_dump())
    config.dataset = dm.model_copy()

    config.student = TinyViT(
        model_size=vit_femto,
        num_classes=dm.num_classes,
        scale_op=config.scale_op,
    )

    config.model_name = "vit"
    config.model_size = "femto"

    lit = LitNetViT(config)
    datamodule:DataSetModule=dm.build()
    trainer = AccelTrainer(
        max_epochs=args.epochs,
        # mixed_precision="bf16" if args.amp else "no",
        gradient_accumulation_steps=1,
        log_every_n_steps=10,
        enable_ema=0.99** (datamodule.batch_size[0] // (128 if datamodule.batch_size[0] > 128 else datamodule.batch_size[0])),
    )
    datamodule.setup()
    tl,vl = datamodule.train_dataloader(repeats=0.1),datamodule.val_dataloader()
    trainer.fit(lit, train_dataloader=tl, val_dataloader=vl, val_first=False)


if __name__ == "__main__":
    main()