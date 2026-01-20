"""
Common utilities for BitNetCNN implementations.
This module contains shared components used across different BitNet model implementations.
"""

import os
import re
import tempfile
import warnings
import zipfile

from typing import Dict, Optional, Sequence, Tuple, List, Union
import torch

from bitlayers.bit import Bit
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

def snap_model(model):
    def snapshot_params(m):
        return {k: v.detach().clone().cpu() for k, v in m.named_parameters()}
    param = snapshot_params(model)
    def snapshot_buffers(m):
        return {k: v.detach().clone().cpu() for k, v in m.named_buffers()}
    buf = snapshot_buffers(model)
    return param,buf

@torch.no_grad()
def recover_snap(model: nn.Module, params_snap, bufs_snap):
    # restore parameters
    for name, p in model.named_parameters():
        if name in params_snap:
            p.copy_(params_snap[name].to(p.device))

    # restore buffers (e.g. BN running_mean/var)
    for name, b in model.named_buffers():
        if name in bufs_snap:
            b.copy_(bufs_snap[name].to(b.device))

# ----------------------------
# Quantization Utilities
# ----------------------------
def summ(model, verbose=True, include_buffers=True):
    info = []
    for name, module in model.named_modules():
        # parameters for this module only (no children)
        params = list(module.parameters(recurse=False))
        nparams = sum(p.numel() for p in params)

        # collect dtypes from params (and optionally buffers)
        tensors = params
        if include_buffers:
            tensors += list(module.buffers(recurse=False))

        dtypes = {t.dtype for t in tensors}
        if dtypes:
            # e.g. "float32", "float16, int8"
            dtype_str = ", ".join(
                sorted(str(dt).replace("torch.", "") for dt in dtypes)
            )
        else:
            dtype_str = "-" + " "*14

        out_chs = ""
        if hasattr(module,'out_channels'):
            out_chs = f"out_channels={module.out_channels}"
            
        row = (name, module.__class__.__name__, nparams, dtype_str)
        info.append(row)

        if verbose:
            print(f"{name:35} {module.__class__.__name__:25} "
                  f"params={nparams:8d}  dtypes={dtype_str:15} {out_chs}")
    return info

# ----------------------------
# Model-wide conversion helpers
# ----------------------------
@torch.no_grad()
def convert_to_ternary(module: nn.Module) -> nn.Module:
    """
    Recursively replace Bit.Conv2d/Bit.Linear with Ternary*Infer modules.
    Returns a new nn.Module (original left untouched if you deepcopy before).
    """
    for name, child in list(module.named_children()):
        if hasattr(child, 'to_ternary'):
            setattr(module, name, child.to_ternary())
        else:
            convert_to_ternary(child)
    return module

@torch.no_grad()
def convert_to_ternary_p2(module: nn.Module) -> nn.Module:
    """
    Recursively replace Bit.Conv2d/Bit.Linear with their PoT inference counterparts.
    """
    for name, child in list(module.named_children()):
        if hasattr(child, 'to_ternary_p2'):
            setattr(module, name, child.to_ternary_p2())
        else:
            convert_to_ternary_p2(child)
    return module

def replace_all2Bit(model: nn.Module, scale_op: str = "median", wrap_same: bool = True) -> Tuple[int, int]:
    convs, linears = 0, 0

    for name, child in list(model.named_children()):
        # Recurse first
        c_cnt, l_cnt = replace_all2Bit(child, scale_op, wrap_same)
        convs += c_cnt
        linears += l_cnt

        # Determine if child is already a Bit layer (if Bit exists)
        try:
            is_bit_conv = isinstance(child, Bit.Conv2d)
            is_bit_linear = isinstance(child, Bit.Linear)
        except Exception:
            is_bit_conv = False
            is_bit_linear = False

        new_child = None

        # Replace Conv2d with Bit.Conv2d (respect wrap_same)
        if isinstance(child, nn.Conv2d) and (wrap_same or not is_bit_conv):
            dev = child.weight.device
            dt = child.weight.dtype

            new_child = Bit.Conv2d(
                in_c=child.in_channels,
                out_c=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None),
                scale_op=scale_op,
            ).to(device=dev, dtype=dt)

            # Best-effort: copy weights/bias if attributes are compatible
            # with torch.no_grad():
            #     if hasattr(new_child, "weight") and new_child.weight.shape == child.weight.shape:
            #         new_child.weight.copy_(child.weight)
            #     if child.bias is not None and hasattr(new_child, "bias") and new_child.bias is not None:
            #         new_child.bias.copy_(child.bias)

            convs += 1

        # Replace Linear with Bit.Linear (respect wrap_same)
        elif isinstance(child, nn.Linear) and (wrap_same or not is_bit_linear):
            dev = child.weight.device
            dt = child.weight.dtype

            new_child = Bit.Linear(
                in_f=child.in_features,
                out_f=child.out_features,
                bias=(child.bias is not None),
                scale_op=scale_op,
            ).to(device=dev, dtype=dt)

            # Best-effort: copy weights/bias if attributes are compatible
            # with torch.no_grad():
            #     if hasattr(new_child, "weight") and new_child.weight.shape == child.weight.shape:
            #         new_child.weight.copy_(child.weight)
            #     if child.bias is not None and hasattr(new_child, "bias") and new_child.bias is not None:
            #         new_child.bias.copy_(child.bias)

            linears += 1

        if new_child is not None:
            setattr(model, name, new_child)

    return convs, linears

def GUI_tool(model,
             resize=None,
             class_names=None,
             preprocess=None,
             device=None,
             topk=5,
             logits_fn=None,
             title="Classifier GUI (no resize to model)"):
    """
    model:      torch.nn.Module (put in eval mode automatically)
    class_names:list[str] or None
    preprocess: callable(PIL.Image) -> torch.Tensor (C,H,W); default = ToTensor() only
    device:     torch.device or str; default = model's first parameter device (else cpu)
    topk:       int
    logits_fn:  callable(tensor[N,C,H,W]) -> logits[N,num_classes]; default = model(x)
                (useful to inject an adapter, e.g. lambda x: adapter(model(x)))
    title:      window title

    IMPORTANT: The image fed to the model is NOT resized here.
               The preview image may be scaled ONLY for display.
    """
    # --- Setup model/device ---

    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
    from PIL import Image, ImageTk
    
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    if isinstance(device, str):
        device = torch.device(device)

    # Default preprocess: just ToTensor (0..1), no resize
    if preprocess is None:
        preprocess = transforms.ToTensor()

    # Helpers
    def load_image(path,resize=resize):
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if resize:
            img = img.resize(resize)
        return img

    def tensor_from_pil(img):
        # preprocess returns CxHxW
        t = preprocess(img)
        if t.ndim == 3:
            t = t.unsqueeze(0)  # NxCxHxW
        return t

    # --- Tkinter UI ---
    root = tk.Tk()
    root.title(title)

    # Main frames
    frm_top = ttk.Frame(root, padding=8)
    frm_top.pack(side=tk.TOP, fill=tk.X)

    frm_mid = ttk.Frame(root, padding=8)
    frm_mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    frm_right = ttk.Frame(frm_mid, padding=(8, 0))
    frm_right.pack(side=tk.RIGHT, fill=tk.Y)

    # Canvas to show image (scaled only for display)
    canvas_size = 448  # display only, not used for model input
    canvas = tk.Canvas(frm_mid, width=canvas_size, height=canvas_size, bg="#222")
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Controls
    btn_open = ttk.Button(frm_top, text="Open image…")
    btn_predict = ttk.Button(frm_top, text="Predict", state=tk.DISABLED)
    lbl_path = ttk.Label(frm_top, text="No file selected", width=60)

    btn_open.pack(side=tk.LEFT)
    btn_predict.pack(side=tk.LEFT, padx=(8, 0))
    lbl_path.pack(side=tk.LEFT, padx=(12, 0))

    # Results box
    lbl_res = ttk.Label(frm_right, text="Top-K", font=("TkDefaultFont", 10, "bold"))
    lbl_res.pack(anchor="nw")
    txt = tk.Text(frm_right, width=40, height=24, wrap="word")
    txt.pack(fill=tk.Y, expand=False)

    # State
    state = {"pil": None, "path": None, "photo": None}

    def show_image_on_canvas(pil_img):
        # scale to fit canvas while keeping aspect ratio (DISPLAY ONLY)
        w, h = pil_img.size
        scale = min(canvas_size / max(w, 1), canvas_size / max(h, 1), 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)
        preview = pil_img if (disp_w == w and disp_h == h) else pil_img.resize((disp_w, disp_h))
        photo = ImageTk.PhotoImage(preview)
        canvas.delete("all")
        canvas.create_image(canvas_size // 2, canvas_size // 2, image=photo, anchor="center")
        state["photo"] = photo  # keep ref

    def on_open():
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp;*.tif;*.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            img = load_image(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            return
        state["pil"] = img
        state["path"] = path
        lbl_path.config(text=os.path.basename(path))
        show_image_on_canvas(img)
        btn_predict.config(state=tk.NORMAL)

    def safe_class_name(i):
        if class_names is None:
            return str(i)
        try:
            return str(class_names[i])
        except Exception:
            return str(i)

    @torch.no_grad()
    def on_predict():
        if state["pil"] is None:
            return
        img = state["pil"]

        try:
            x = tensor_from_pil(img).to(device)  # NxCxHxW, original size
        except Exception as e:
            messagebox.showerror("Preprocess error", f"Failed to preprocess image:\n{e}")
            return

        try:
            logits = logits_fn(x) if callable(logits_fn) else model(x)
        except Exception as e:
            messagebox.showerror(
                "Model error",
                f"Forward pass failed.\n\nIf your model requires a fixed size, "
                f"you must pass a preprocess that resizes/crops accordingly.\n\nError:\n{e}"
            )
            return

        # top-K
        probs = F.softmax(logits, dim=-1)
        k = min(topk, probs.shape[-1])
        vals, idxs = probs.topk(k, dim=-1)
        vals, idxs = vals[0].tolist(), idxs[0].tolist()

        # Render results
        txt.delete("1.0", tk.END)
        for rank, (p, i) in enumerate(zip(vals, idxs), start=1):
            txt.insert(tk.END, f"{rank:>2}. {safe_class_name(i)}  —  {p*100:.2f}%\n")

    btn_open.config(command=on_open)
    btn_predict.config(command=on_predict)

    root.mainloop()

def load_tiny200_to_in1k_map(path: str, expected_out: int = 200) -> List[List[int]]:
    """
    Reads a mapping text file where each *line* corresponds to one Tiny-ImageNet class (0..199),
    and contains one or more ImageNet-1k indices (0..999) to aggregate from.
    Lines can use commas/spaces/colons, and may include comments after '#'.

    Example lines:
      8
      65, 67
      42 417 901   # comment
      12: 12

    Returns: list of length 200; each item is a sorted list of unique int indices.
    """
    mapping: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            # extract all integers on the line
            ids = [int(s) for s in re.findall(r"\d+", line)]
            ids = sorted(set(ids))
            mapping.append(ids)

    if len(mapping) != expected_out:
        warnings.warn(
            f"Expected {expected_out} mapping rows, found {len(mapping)}. "
            "Ensure the file has one (non-empty) line per Tiny-ImageNet class."
        )

    # sanity checks
    for i, ids in enumerate(mapping):
        if not ids:
            raise ValueError(f"Mapping row {i} is empty (no ImageNet-1k indices).")
        for k in ids:
            if not (0 <= k < 1000):
                raise ValueError(f"Invalid IN1k index {k} in row {i}; expected 0..999.")
    return mapping

def build_projection_matrix(mapping: Sequence[Sequence[int]],
                            in_dim: int = 1000,
                            dtype: torch.dtype = torch.float32,
                            device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Builds a (out_dim x in_dim) binary matrix M where M[j, i] = 1 if IN1k idx i maps to Tiny200 cls j.
    Use with probability vectors: p200 = p1k @ M.T  (or p1k.matmul(M.T))
    """
    out_dim = len(mapping)
    M = torch.zeros((out_dim, in_dim), dtype=dtype, device=device)
    for j, ids in enumerate(mapping):
        M[j, ids] = 1.0
    # Optional: warn if any IN1k index is assigned to multiple Tiny classes (overlap)
    overlaps = (M.sum(0) > 1).nonzero(as_tuple=False).flatten()
    if overlaps.numel() > 0:
        warnings.warn(f"{overlaps.numel()} ImageNet-1k indices map to multiple Tiny classes; "
                      "their probability mass will be counted multiple times before renormalization.")
    return M

class IN1kToTiny200Adapter(nn.Module):
    """
    Adapts teacher outputs from 1000 classes to 200 using a mapping.
    Default behavior:
      - apply temperature (T) to 1k logits,
      - softmax -> p1k,
      - aggregate: p200_raw = p1k @ M.T,
      - renormalize to sum to 1 (per sample),
      - return either probabilities (p200) or logits (log p200).
    """
    def __init__(self,
                 mapping: Sequence[Sequence[int]],
                 temperature: float = 1.0,
                 renormalize: bool = True,
                 in_dim: int = 1000,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.temperature = float(temperature)
        self.renormalize = bool(renormalize)
        M = build_projection_matrix(mapping, in_dim=in_dim, device=device)
        # register as buffer so it moves with .to(device) and gets saved in state_dict
        self.register_buffer("proj", M)  # shape: (200, 1000)

    @torch.no_grad()
    def probs(self, logits_1k: torch.Tensor) -> torch.Tensor:
        """Return 200-way probabilities (with temperature applied at 1k-level)."""
        t = self.temperature
        if t != 1.0:
            logits_1k = logits_1k / t
        p1k = logits_1k.softmax(dim=-1)                  # (N, 1000)
        p200 = p1k.matmul(self.proj.t())                 # (N, 200), sum probabilities per mapping
        if self.renormalize:
            denom = p200.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            p200 = p200 / denom
        return p200

    @torch.no_grad()
    def forward(self, logits_1k: torch.Tensor, return_logits: bool = True) -> torch.Tensor:
        """
        If return_logits=True: returns log(p200 + eps), which are valid "logits"
        (since softmax(log p) = p). Else returns probabilities p200.
        """
        p200 = self.probs(logits_1k)
        if return_logits:
            return (p200 + 1e-12).log()
        return p200

def set_export_mode(m, flag=True):
    if hasattr(m, "export_mode"):
        m.export_mode = flag
    for c in m.children():
        set_export_mode(c, flag)

def export_onnx(model, dummy_input, path="model.onnx"):
    model = model.eval()
    set_export_mode(model, True)
    exported = torch.onnx.dynamo_export(model, dummy_input)
    exported.save(path)

def load_zip_weights(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu"
) -> Optional[Dict[str, torch.Tensor]]:
    state_dict = None

    if checkpoint_path.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(checkpoint_path, "r") as zf:
                pt_files = [f for f in zf.namelist() if f.endswith(".pt")]
                if not pt_files:
                    raise ValueError(f"No .pt file found in zip: {checkpoint_path}")
                pt_file = pt_files[0]
                zf.extract(pt_file, tmpdir)
                extracted = os.path.join(tmpdir, pt_file)
                raw = torch.load(extracted, map_location=device, weights_only=False)
    else:
        raw = torch.load(extracted, map_location=device, weights_only=False)
    
    if checkpoint_path:
        try:
            if isinstance(raw, dict):
                # Prefer EMA if present
                for key in ("model_ema", "ema", "ema_state_dict"):
                    if key in raw and isinstance(raw[key], dict):
                        state_dict = raw[key]
                # Then standard keys
                for key in ("model", "state_dict"):
                    if key in raw and isinstance(raw[key], dict):
                        state_dict = raw[key]
            # Might already be a raw state_dict (e.g., OrderedDict)
            if hasattr(raw, "keys") and hasattr(raw, "items"):
                state_dict = raw  # type: ignore
            else:
                raise ValueError("Unrecognized checkpoint format; no state_dict found.")
        except Exception as e:
            print(f"[warn] Could not load local checkpoint '{checkpoint_path}': {e}")
    return state_dict

# if __name__ == '__main__':
#     freeze_support()
#     dm = TinyImageNetDataModule("./data",4)
#     dm.setup()
#     for data,label in dm.train_dataloader():
#         break
#     print(data)
#     print(label)