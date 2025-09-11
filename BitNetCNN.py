import torch
import torch.nn as nn
import torch.nn.functional as F
import math

EPS = 1e-8

def _reduce_abs(x, keep_dim, op="mean"):
    # reduce |x| over all dims except keep_dim
    dims = [d for d in range(x.dim()) if d != keep_dim]
    a = x.abs()
    if op == "mean":
        s = a.mean(dim=dims, keepdim=True)
    elif op == "median":
        # median over flattened other dims
        perm = (keep_dim,) + tuple(d for d in range(x.dim()) if d != keep_dim)
        flat = a.permute(perm).contiguous().view(a.size(keep_dim), -1)
        s = flat.median(dim=1).values.view([a.size(keep_dim)] + [1]*(x.dim()-1))
        inv_perm = tuple(sorted(range(x.dim()), key=lambda i: perm[i]))
        s = s.permute(inv_perm).contiguous()
    else:
        raise ValueError("op must be 'mean' or 'median'")
    return s.clamp_min(EPS)

class ActQuant(nn.Module):
    """Symmetric k-bit per-tensor activation quantizer (k in {4,8,...})."""
    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits
        assert bits in [None, 4, 8], "Use 4 or 8 bits (or None for float)."

    def forward(self, x):
        if self.bits is None:
            return x
        # Optional range limiter (helps stability)
        x = torch.clamp(x, -8.0, 8.0)
        qmax = (1 << (self.bits - 1)) - 1  # e.g., 127 for 8-bit
        s = x.detach().abs().amax().clamp_min(EPS) / qmax
        x_int = torch.round(x / s).clamp(-qmax, qmax)
        x_q = x_int * s
        # STE
        return x + (x_q - x).detach()

class Bit1p58Weight(nn.Module):
    """
    1.58-bit (ternary) weight quantizer with AbsMean/AbsMedian scaling.
    Per-output-channel scaling for Conv2d, per-row for Linear.
    """
    def __init__(self, dim=0, scale_op="mean"):
        super().__init__()
        self.dim = dim
        self.scale_op = scale_op  # "mean" or "median"

    def forward(self, w):
        s = _reduce_abs(w, keep_dim=self.dim, op=self.scale_op)  # scale
        w_bar = (w / s).detach()  # don't let scale backprop through quant
        # Round-to-nearest in {-1, 0, +1}, then clip
        w_q = torch.round(w_bar).clamp_(-1, 1)
        # Dequantize for normal PyTorch kernels
        w_hat = w_q * s
        # STE (straight-through) for gradients wrt original w
        return w + (w_hat - w).detach()

class BitConv2d(nn.Module):
    """
    Drop-in Conv2d with 1.58-bit weights + k-bit activations.
    Handles ANY kernel size including 1×1 (just set kernel_size=1).
    """
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, act_bits=8, scale_op="mean"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_c, in_c // groups, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_c)) if bias else None
        self.act_q = ActQuant(bits=act_bits)
        self.w_q = Bit1p58Weight(dim=0, scale_op=scale_op)
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups

    def forward(self, x):
        xq = self.act_q(x)
        wq = self.w_q(self.weight)
        return F.conv2d(xq, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)

class BitLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=True, act_bits=None, scale_op="mean"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        self.act_q = ActQuant(bits=act_bits)  # often None for classifier input (after GAP)
        self.w_q = Bit1p58Weight(dim=0, scale_op=scale_op)

    def forward(self, x):
        xq = self.act_q(x) if self.act_q.bits is not None else x
        wq = self.w_q(self.weight)
        return F.linear(xq, wq, self.bias)

class InvertedResidualBit(nn.Module):
    def __init__(self, in_c, out_c, expand, stride, act_bits=8, scale_op="mean"):
        super().__init__()
        hid = in_c * expand
        self.use_res = stride == 1 and in_c == out_c
        self.pw1 = nn.Sequential(
            BitConv2d(in_c, hid, 1, bias=False, act_bits=act_bits, scale_op=scale_op),
            nn.BatchNorm2d(hid),
            nn.Hardtanh(-1, 1)  # Bi-Real style limiter (pre-quant helps)
        )
        self.dw = nn.Sequential(
            BitConv2d(hid, hid, 3, stride=stride, padding=1, groups=hid,
                      bias=False, act_bits=act_bits, scale_op=scale_op),
            nn.BatchNorm2d(hid),
            nn.Hardtanh(-1, 1)
        )
        self.pw2 = nn.Sequential(
            BitConv2d(hid, out_c, 1, bias=False, act_bits=act_bits, scale_op=scale_op),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        y = self.pw2(self.dw(self.pw1(x)))
        return x + y if self.use_res else y

class BitNetCNN(nn.Module):
    def __init__(self, num_classes=100, act_bits=8, scale_op="mean"):
        super().__init__()
        self.stem = nn.Sequential(
            BitConv2d(3, 32, 3, stride=1, padding=1, bias=False, act_bits=act_bits, scale_op=scale_op),
            nn.BatchNorm2d(32),
            nn.Hardtanh(-1, 1)
        )
        self.stage1 = InvertedResidualBit(32, 64, expand=2, stride=2, act_bits=act_bits, scale_op=scale_op)
        self.stage2 = InvertedResidualBit(64, 128, expand=2, stride=2, act_bits=act_bits, scale_op=scale_op)
        self.stage3 = InvertedResidualBit(128, 256, expand=2, stride=2, act_bits=act_bits, scale_op=scale_op)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # You can keep this float by using act_bits=None in BitLinear
            BitLinear(256, num_classes, bias=True, act_bits=None, scale_op=scale_op)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x)
        return self.head(x)

# Example
model = BitNetCNN(num_classes=100, act_bits=8, scale_op="mean")
x = torch.randn(4, 3, 32, 32)
logits = model(x)
print(logits.shape)  # [4, 100]
