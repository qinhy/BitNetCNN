# mnist_bitnet_lightning.py
import argparse
import torch

torch.set_float32_matmul_precision('high')
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy

# Import from common_utils
from common_utils import *
from bitlayers import convs
from bitlayers.acts import ActModels
from bitlayers.norms import NormModels
# ----------------------------
# Simple BitNet block & model
# ----------------------------
InvertedResidual = convs.Conv2dModels.InvertedResidual
InvertedResidualModule = convs.Conv2dModules.InvertedResidual
# class InvertedResidual(nn.Module):
#     def __init__(self, in_channels, out_channels, exp_ratio, 
#                  stride, bias=True, scale_op="median", conv2d=Bit.Conv2d,
#                  drop_path_rate=0.0):
#         super().__init__()
#         hid = int(in_channels * exp_ratio)
#         self.use_res = (stride == 1 and in_channels == out_channels)

#         self.pw1 = nn.Sequential(
#             conv2d(in_channels, hid, kernel_size=1, bias=bias, scale_op=scale_op),
#             nn.BatchNorm2d(hid),
#             nn.SiLU(inplace=True),
#         )
#         self.dw = nn.Sequential(
#             conv2d(hid, hid, kernel_size=3, stride=stride, padding=1, groups=hid,
#                       bias=bias, scale_op=scale_op),
#             nn.BatchNorm2d(hid),
#             nn.SiLU(inplace=True),
#         )
#         # no activation here
#         self.pw2 = nn.Sequential(
#             conv2d(hid, out_channels, kernel_size=1, bias=bias, scale_op=scale_op),
#             nn.BatchNorm2d(out_channels),
#         )
#     def build(self):
#         return self
    
#     def forward(self, x):
#         y = self.pw2(self.dw(self.pw1(x)))
#         return x + y if self.use_res else y


class NetCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, expand_ratio=5,
                 drop2d_p=0.05, drop_p=0.1, scale_op="median", bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.expand_ratio = expand_ratio
        self.drop2d_p = drop2d_p
        self.drop_p = drop_p
        self.scale_op = scale_op
        def act(): return ActModels.SiLU(inplace=True)
        def norm(): return NormModels.BatchNorm2d(num_features=-1)
        
        self.stem:convs.Conv2dModules.Conv2dNormAct = convs.Conv2dModels.Conv2dNormAct(
                            in_channels=in_channels,
                            out_channels=2**expand_ratio,
                            kernel_size=3, stride=1, padding=1,
                            bias=bias, scale_op=scale_op,norm=norm(),act=act()).build()        
        
        def cconvs():
            return dict(conv_pw_exp_layer=convs.Conv2dModels.Conv2dPointwiseNormAct(
                            in_channels=-1,norm=norm(),act=act()),
                        conv_dw_layer=convs.Conv2dModels.Conv2dDepthwiseNormAct(
                            in_channels=-1,norm=norm(),act=act()),
                        conv_pw_layer=convs.Conv2dModels.Conv2dPointwiseNorm(
                            in_channels=-1,norm=norm()),)
        
        self.stage1 = InvertedResidual(in_channels=2**expand_ratio,out_channels=2**(expand_ratio+1),
                                        **cconvs(),
                                        padding=0,exp_ratio=2.0,stride=2,scale_op=scale_op,bias=bias).build()
        
        self.stage2 = InvertedResidual(in_channels=2**(expand_ratio+1),out_channels=2**(expand_ratio+2),
                                        **cconvs(),
                                          padding=0,exp_ratio=2.0,stride=2,scale_op=scale_op,bias=bias).build()
        
        self.stage3 = InvertedResidual(in_channels=2**(expand_ratio+2),out_channels=2**(expand_ratio+3),
                                        **cconvs(),
                                          padding=0,exp_ratio=2.0,stride=2,scale_op=scale_op,bias=bias).build()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=drop_p),
            Bit.Linear(2**(expand_ratio+3), num_classes, bias=bias, scale_op=scale_op),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)

    def clone(self):
        return NetCNN(self.in_channels, self.num_classes, self.expand_ratio,
                     self.drop2d_p, self.drop_p, self.scale_op)
# summ(convert_to_ternary(NetCNN()))
# ----------------------------
# LightningModule wrapper using LitBit
# ----------------------------
class LitNetCNNKD(LitBit):
    def __init__(self, lr, wd, epochs, label_smoothing=0.0,
                 alpha_kd=0.0, alpha_hint=0.0, T=4.0, scale_op="median",
                 width_mult=1.0, amp=True, export_dir="./ckpt_mnist"):
        # No teacher, no KD for MNIST (simple model)
        student = NetCNN(in_channels=1, num_classes=10, expand_ratio=5, scale_op=scale_op)
        super().__init__(
            lr=lr, wd=wd, epochs=epochs, label_smoothing=label_smoothing,
            alpha_kd=alpha_kd, alpha_hint=alpha_hint, T=T, scale_op=scale_op,
            width_mult=width_mult, amp=amp, export_dir=export_dir,
            dataset_name='mnist',
            model_name='netcnn',
            model_size='small',
            hint_points=[],
            student=student,
            teacher=None,
            num_classes=10,
        )
        # Override metrics for MNIST (10 classes instead of 100)
        self.acc_fp = MulticlassAccuracy(num_classes=10).eval()
        self.acc_tern = MulticlassAccuracy(num_classes=10).eval()

    def configure_optimizers(self):
        # Use AdamW for MNIST instead of SGD
        opt = torch.optim.AdamW(
            list(self.student.parameters()) + list(self.hint.parameters()),
            lr=self.hparams.lr, weight_decay=self.hparams.wd
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs, eta_min=self.hparams.lr*0.01)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch", "monitor": "val/acc_tern"},
        }

# ----------------------------
# CLI / main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p = add_common_args(p)
    p.set_defaults(
        data="./data",
        out="./ckpt_mnist",
        epochs=50,
        batch_size=512,
        lr=2e-3,
        wd=1e-4,
        label_smoothing=0.0
    )
    return p.parse_args()

def main():
    args = parse_args()

    lit = LitNetCNNKD(
        lr=args.lr, wd=args.wd, epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        alpha_kd=0.0, alpha_hint=0.0, T=args.T,
        scale_op=args.scale_op, amp=args.amp, export_dir=args.out
    )

    trainer, dm = setup_trainer(args, lit)

    # Override datamodule with MNIST
    dm = MNISTDataModule(data_dir=args.data, batch_size=args.batch_size, num_workers=4)

    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)

if __name__ == "__main__":
    main()

# uv run BitNetCNN.py
# Seed set to 42
# GPU available: True (cuda), used: True
# TPU available: False, using: 0 TPU cores
# HPU available: False, using: 0 HPUs
# LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

#   | Name     | Type               | Params | Mode
# --------------------------------------------------------
# 0 | model    | NetCNN             | 140 K  | train
# 1 | crit     | CrossEntropyLoss   | 0      | train
# 2 | acc_fp   | MulticlassAccuracy | 0      | train
# 3 | acc_tern | MulticlassAccuracy | 0      | train
# --------------------------------------------------------
# 140 K     Trainable params
# 0         Non-trainable params
# 140 K     Total params
# 0.560     Total estimated model params size (MB)
# 62        Modules in train mode
# 0         Modules in eval mode
# ✓ Exported frozen ternary model to ./ckpt_mnist\mnist_bitnet_ternary.pt (val_acc_ternary=9.90%)
# Epoch 0: 100%|█████████| 30/30 [00:09<00:00,  3.08it/s, v_num=18]
# ✓ Exported frozen ternary model to ./ckpt_mnist\mnist_bitnet_ternary.pt (val_acc_ternary=15.14%)
# Epoch 1: 100%|█████████| 30/30 [00:06<00:00,  4.45it/s, v_num=18, val_loss=2.840, val_acc=0.152, val_acc_ternary=0.151, train_loss=1.760]
# ✓ Exported frozen ternary model to ./ckpt_mnist\mnist_bitnet_ternary.pt (val_acc_ternary=85.76%)
# Epoch 2: 100%|█████████| 30/30 [00:06<00:00,  4.52it/s, v_num=18, val_loss=0.444, val_acc=0.858, val_acc_ternary=0.858, train_loss=0.863]
# ✓ Exported frozen ternary model to ./ckpt_mnist\mnist_bitnet_ternary.pt (val_acc_ternary=91.30%)
# Epoch 3: 100%|█████████| 30/30 [00:06<00:00,  4.52it/s, v_num=18, val_loss=0.290, val_acc=0.913, val_acc_ternary=0.913, train_loss=0.392]
# Epoch 4: 100%|█████████| 30/30 [00:06<00:00,  4.49it/s, v_num=18, val_loss=0.290, val_acc=0.909, val_acc_ternary=0.909, train_loss=0.240]
# ✓ Exported frozen ternary model to ./ckpt_mnist\mnist_bitnet_ternary.pt (val_acc_ternary=95.99%)
# Epoch 5: 100%|█████████| 30/30 [00:06<00:00,  4.49it/s, v_num=18, val_loss=0.128, val_acc=0.960, val_acc_ternary=0.960, train_loss=0.185]
# ✓ Exported frozen ternary model to ./ckpt_mnist\mnist_bitnet_ternary.pt (val_acc_ternary=96.89%)
# Epoch 6: 100%|████████| 30/30 [00:06<00:00,  4.45it/s, v_num=18, val_loss=0.0958, val_acc=0.969, val_acc_ternary=0.969, train_loss=0.149]
# ✓ Exported frozen ternary model to ./ckpt_mnist\mnist_bitnet_ternary.pt (val_acc_ternary=97.69%)
# Epoch 7: 100%|████████| 30/30 [00:06<00:00,  4.53it/s, v_num=18, val_loss=0.0767, val_acc=0.977, val_acc_ternary=0.977, train_loss=0.130]
# Epoch 8: 100%|████████| 30/30 [00:06<00:00,  4.48it/s, v_num=18, val_loss=0.0817, val_acc=0.975, val_acc_ternary=0.975, train_loss=0.119]
# Epoch 9: 100%|████████| 30/30 [00:06<00:00,  4.37it/s, v_num=18, val_loss=0.0928, val_acc=0.971, val_acc_ternary=0.971, train_loss=0.106]
# Epoch 10: 100%|███████| 30/30 [00:06<00:00,  4.58it/s, v_num=18, val_loss=0.0788, val_acc=0.975, val_acc_ternary=0.975, train_loss=0.102]
# ✓ Exported frozen ternary model to ./ckpt_mnist\mnist_bitnet_ternary.pt (val_acc_ternary=97.70%)
# Epoch 11: 100%|██████| 30/30 [00:06<00:00,  4.58it/s, v_num=18, val_loss=0.0681, val_acc=0.977, val_acc_ternary=0.977, train_loss=0.0913]
# ✓ Exported frozen ternary model to ./ckpt_mnist\mnist_bitnet_ternary.pt (val_acc_ternary=98.30%)
# Epoch 12: 100%|██████| 30/30 [00:06<00:00,  4.58it/s, v_num=18, val_loss=0.0517, val_acc=0.983, val_acc_ternary=0.983, train_loss=0.0872]
# Epoch 13: 100%|██████| 30/30 [00:06<00:00,  4.56it/s, v_num=18, val_loss=0.0523, val_acc=0.982, val_acc_ternary=0.982, train_loss=0.0827]
# Epoch 14: 100%|███████| 30/30 [00:06<00:00,  4.57it/s, v_num=18, val_loss=0.0559, val_acc=0.982, val_acc_ternary=0.982, train_loss=0.080]
# Epoch 15: 100%|██████| 30/30 [00:06<00:00,  4.52it/s, v_num=18, val_loss=0.0556, val_acc=0.982, val_acc_ternary=0.982, train_loss=0.0739]
# Epoch 16: 100%|██████| 30/30 [00:06<00:00,  4.59it/s, v_num=18, val_loss=0.0746, val_acc=0.977, val_acc_ternary=0.977, train_loss=0.0738]
# Epoch 17: 100%|███████| 30/30 [00:06<00:00,  4.58it/s, v_num=18, val_loss=0.0554, val_acc=0.981, val_acc_ternary=0.981, train_loss=0.072]
# ✓ Exported frozen ternary model to ./ckpt_mnist\mnist_bitnet_ternary.pt (val_acc_ternary=98.44%)
# Epoch 18: 100%|████████| 30/30 [00:06<00:00,  4.52it/s, v_num=18, val_loss=0.046, val_acc=0.985, val_acc_ternary=0.984, train_loss=0.067]
# Epoch 19: 100%|██████| 30/30 [00:06<00:00,  4.55it/s, v_num=18, val_loss=0.0504, val_acc=0.984, val_acc_ternary=0.984, train_loss=0.0668]
# ✓ Exported frozen ternary model to ./ckpt_mnist\mnist_bitnet_ternary.pt (val_acc_ternary=98.78%)
# Epoch 20: 100%|██████| 30/30 [00:06<00:00,  4.48it/s, v_num=18, val_loss=0.0386, val_acc=0.988, val_acc_ternary=0.988, train_loss=0.0647]
# Epoch 21: 100%|██████| 30/30 [00:06<00:00,  4.53it/s, v_num=18, val_loss=0.0571, val_acc=0.982, val_acc_ternary=0.982, train_loss=0.0622]
# Epoch 22: 100%|██████| 30/30 [00:06<00:00,  4.57it/s, v_num=18, val_loss=0.0476, val_acc=0.985, val_acc_ternary=0.985, train_loss=0.0595]
# Epoch 23: 100%|███████| 30/30 [00:06<00:00,  4.56it/s, v_num=18, val_loss=0.0644, val_acc=0.978, val_acc_ternary=0.978, train_loss=0.059]
# Epoch 24: 100%|██████| 30/30 [00:06<00:00,  4.54it/s, v_num=18, val_loss=0.0508, val_acc=0.982, val_acc_ternary=0.982, train_loss=0.0606]
# Epoch 25: 100%|██████| 30/30 [00:06<00:00,  4.55it/s, v_num=18, val_loss=0.0406, val_acc=0.985, val_acc_ternary=0.986, train_loss=0.0537]
# Epoch 26: 100%|██████| 30/30 [00:06<00:00,  4.50it/s, v_num=18, val_loss=0.0642, val_acc=0.979, val_acc_ternary=0.979, train_loss=0.0541]
# Epoch 27: 100%|██████| 30/30 [00:06<00:00,  4.60it/s, v_num=18, val_loss=0.0659, val_acc=0.978, val_acc_ternary=0.978, train_loss=0.0538]
# Epoch 28: 100%|██████| 30/30 [00:06<00:00,  4.55it/s, v_num=18, val_loss=0.0394, val_acc=0.986, val_acc_ternary=0.986, train_loss=0.0541]
# Epoch 29: 100%|██████| 30/30 [00:06<00:00,  4.58it/s, v_num=18, val_loss=0.0498, val_acc=0.983, val_acc_ternary=0.983, train_loss=0.0504]
# Epoch 30: 100%|██████| 30/30 [00:06<00:00,  4.29it/s, v_num=18, val_loss=0.0503, val_acc=0.982, val_acc_ternary=0.982, train_loss=0.0518]
# Epoch 31: 100%|██████| 30/30 [00:06<00:00,  4.38it/s, v_num=18, val_loss=0.0559, val_acc=0.983, val_acc_ternary=0.983, train_loss=0.0532]
# Epoch 32: 100%|██████| 30/30 [00:06<00:00,  4.36it/s, v_num=18, val_loss=0.0683, val_acc=0.979, val_acc_ternary=0.979, train_loss=0.0502]
# Epoch 33: 100%|██████| 30/30 [00:06<00:00,  4.50it/s, v_num=18, val_loss=0.0409, val_acc=0.987, val_acc_ternary=0.987, train_loss=0.0522]
# ✓ Exported frozen ternary model to ./ckpt_mnist\mnist_bitnet_ternary.pt (val_acc_ternary=98.79%)
# Epoch 34: 100%|██████| 30/30 [00:06<00:00,  4.47it/s, v_num=18, val_loss=0.0372, val_acc=0.988, val_acc_ternary=0.988, train_loss=0.0493]
# ✓ Exported frozen ternary model to ./ckpt_mnist\mnist_bitnet_ternary.pt (val_acc_ternary=98.94%)
# Epoch 35: 100%|██████| 30/30 [00:06<00:00,  4.43it/s, v_num=18, val_loss=0.0339, val_acc=0.989, val_acc_ternary=0.989, train_loss=0.0471]
# Epoch 36: 100%|██████| 30/30 [00:06<00:00,  4.46it/s, v_num=18, val_loss=0.0361, val_acc=0.989, val_acc_ternary=0.988, train_loss=0.0473]
# ✓ Exported frozen ternary model to ./ckpt_mnist\mnist_bitnet_ternary.pt (val_acc_ternary=99.08%)
# Epoch 37: 100%|██████| 30/30 [00:06<00:00,  4.50it/s, v_num=18, val_loss=0.0265, val_acc=0.991, val_acc_ternary=0.991, train_loss=0.0452]
# Epoch 38: 100%|██████| 30/30 [00:06<00:00,  4.47it/s, v_num=18, val_loss=0.0607, val_acc=0.982, val_acc_ternary=0.982, train_loss=0.0451]
# Epoch 39: 100%|██████| 30/30 [00:06<00:00,  4.52it/s, v_num=18, val_loss=0.0336, val_acc=0.990, val_acc_ternary=0.990, train_loss=0.0482]
# Epoch 40: 100%|███████| 30/30 [00:06<00:00,  4.43it/s, v_num=18, val_loss=0.0502, val_acc=0.985, val_acc_ternary=0.985, train_loss=0.046]