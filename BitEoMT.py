import math
from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from bitlayers.dinov3.layers.block import SelfAttentionBlock as DinoSelfAttentionBlock
from bitlayers.dinov3.models.vision_transformer import DinoVisionTransformer, vit_base
from bitlayers.norms import NormModels
from bitlayers.convs import Conv2dModels
from bitlayers.linear import LinearModels
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from common_utils import convert_to_ternary, summ


class ScaleBlock(nn.Module):
    def __init__(self, embed_dim, conv1_layer=Conv2dModels.ConvTranspose2d):
        super().__init__()

        self.conv1 = conv1_layer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=2,
            stride=2,
        ).build()
        self.act = nn.GELU()
        self.conv2 = Conv2dModels.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        ).build()
        self.norm = NormModels.LayerNorm2d(num_features=embed_dim).build()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x

class DinoEoMT(nn.Module):
    """
    Author-style EoMT:
      - Insert queries only for last `num_blocks`
      - Predict mask/class logits per masked block + final
      - Build query->patch attention masks from predicted masks
      - Anneal via attn_mask_probs (prob per masked block)

    Works with:
      - encoder being a wrapper with `.backbone` (author style), OR
      - encoder being the backbone itself (DinoVisionTransformer)
    """

    def __init__(
        self,
        encoder: nn.Module,     # DinoVisionTransformer or wrapper with .backbone
        num_classes: int,
        num_queries: int,
        num_blocks: int = 4,
        masked_attn_enabled: bool = True,
        pixel_mean: Optional[torch.Tensor] = None,
        pixel_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.backbone:Union[nn.Module, DinoVisionTransformer] = encoder.backbone if hasattr(encoder, "backbone") else encoder

        self.num_q = num_queries
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        # Annealing probs per masked block (length = num_blocks). Start all-1 (fully masked).
        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        embed_dim = self.backbone.embed_dim if hasattr(self.backbone, "embed_dim") else self.backbone.num_features

        # Queries as embedding (author style)
        self.q = nn.Embedding(num_queries, embed_dim)

        # Heads (match author's structure)
        self.class_head = LinearModels.Linear(in_features=embed_dim, out_features=num_classes + 1).build()

        self.mask_head = nn.Sequential(
            LinearModels.Linear(in_features=embed_dim, out_features=embed_dim).build(),
            nn.GELU(),
            LinearModels.Linear(in_features=embed_dim, out_features=embed_dim).build(),
            nn.GELU(),
            LinearModels.Linear(in_features=embed_dim, out_features=embed_dim).build(),
        )

        # Determine patch_size and num_upscale (author logic)
        patch_size = getattr(self.backbone, "patch_size", None)
        if patch_size is None and hasattr(self.backbone, "patch_embed"):
            patch_size = getattr(self.backbone.patch_embed, "patch_size", 16)
        if isinstance(patch_size, tuple):
            max_patch = max(patch_size[0], patch_size[1])
        else:
            max_patch = int(patch_size)

        num_upscale = max(1, int(math.log2(max_patch)) - 2)  # same as author
        self.upscale = nn.Sequential(*[ScaleBlock(embed_dim) for _ in range(num_upscale)])

        # Optional input normalization (author wrapper has encoder.pixel_mean/std)
        if pixel_mean is not None:
            self.register_buffer("pixel_mean", pixel_mean.view(1, -1, 1, 1))
        else:
            self.pixel_mean = getattr(encoder, "pixel_mean", None)

        if pixel_std is not None:
            self.register_buffer("pixel_std", pixel_std.view(1, -1, 1, 1))
        else:
            self.pixel_std = getattr(encoder, "pixel_std", None)

    # -----------------------
    # helpers (prefix/patch)
    # -----------------------
    def _num_prefix_tokens(self) -> int:
        """
        For your DinoVisionTransformer:
          prefix = CLS (1) + storage tokens (n_storage_tokens)
        If backbone has explicit num_prefix_tokens, use it.
        """
        if hasattr(self.backbone, "num_prefix_tokens"):
            return int(self.backbone.num_prefix_tokens)
        n_storage = int(getattr(self.backbone, "n_storage_tokens", 0))
        return 1 + n_storage

    def _prepare_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Prefer backbone.prepare_tokens_with_masks if present (matches your DINO code).
        Otherwise assume patch_embed already returns token sequence.
        """
        if hasattr(self.backbone, "prepare_tokens_with_masks"):
            tokens, (Hp, Wp) = self.backbone.prepare_tokens_with_masks(x, masks=None)
            return tokens, (Hp, Wp)

        # Fallback: treat patch_embed output as (B, Hp, Wp, C) and add prefix if exists
        t = self.backbone.patch_embed(x)
        if t.dim() == 4:
            B, Hp, Wp, C = t.shape
            t = t.flatten(1, 2)
        else:
            # already (B, N, C)
            B, N, C = t.shape
            Hp = Wp = int(math.sqrt(N))

        return t, (Hp, Wp)

    def _rope(self, Hp: int, Wp: int):
        # Your DINO: rope_embed(H, W)
        if hasattr(self.backbone, "rope_embed") and self.backbone.rope_embed is not None:
            return self.backbone.rope_embed(H=Hp, W=Wp)
        # Author: rope_embeddings(x) (not available in your posted DINO)
        if hasattr(self.backbone, "rope_embeddings"):
            return self.backbone.rope_embeddings
        return None

    def _predict(self, x_norm: torch.Tensor, Hp: int, Wp: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mirror author _predict:
          q = first num_q tokens
          patches = after (num_q + num_prefix_tokens)
          reshape patches -> (B, C, Hp, Wp)
          mask_logits = einsum(mask_head(q), upscale(patches))
        """
        q = x_norm[:, : self.num_q, :]  # (B, Q, C)
        class_logits = self.class_head(q)

        prefix = self._num_prefix_tokens()
        patch_tokens = x_norm[:, self.num_q + prefix :, :]  # (B, N, C)

        # reshape to (B, C, Hp, Wp)
        x_img = patch_tokens.transpose(1, 2).reshape(patch_tokens.shape[0], -1, Hp, Wp)

        mask_logits = torch.einsum("bqc,bchw->bqhw", self.mask_head(q), self.upscale(x_img))
        return mask_logits, class_logits

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask: torch.Tensor, prob: float) -> torch.Tensor:
        """
        Mirror author: for a random subset of queries, disable restriction (set query->patch to True).
        """
        if prob < 1:
            random_queries = (torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device) > prob)
            prefix = self._num_prefix_tokens()
            attn_mask[:, : self.num_q, self.num_q + prefix :][random_queries] = True
        return attn_mask

    def _attn_mask(self, x: torch.Tensor, mask_logits: torch.Tensor, Hp: int, Wp: int, i: int) -> torch.Tensor:
        """
        Mirror author: allow-all mask, then restrict only query->patch region based on interpolated masks.
        """
        B, N, _ = x.shape
        attn_mask = torch.ones(B, N, N, dtype=torch.bool, device=x.device)

        prefix = self._num_prefix_tokens()

        # interpolate predicted masks to patch grid
        interpolated = F.interpolate(mask_logits, size=(Hp, Wp), mode="bilinear", align_corners=False)
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)  # (B, Q, Hp*Wp)

        # restrict query -> patch tokens
        attn_mask[:, : self.num_q, self.num_q + prefix :] = (interpolated > 0)

        # anneal for this masked block
        # author index: i - len(blocks) + num_blocks
        masked_block_idx = i - len(self.backbone.blocks) + self.num_blocks
        prob = float(self.attn_mask_probs[masked_block_idx].item())
        attn_mask = self._disable_attn_mask(attn_mask, prob)
        return attn_mask

    def _block_step(
        self,
        block: Union[nn.Module, DinoSelfAttentionBlock],
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        rope: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Executes a transformer block exactly like author does.
        If your block doesn't expose these attributes, this function will raise and we fallback.
        """
        # attn module
        attn = getattr(block, "attn", None) or getattr(block, "attention", None)
        if attn is None:
            raise AttributeError("Block has no .attn or .attention")

        # If attn supports (x, mask, rope) like author for RoPE backbones
        if rope is not None and callable(attn):
            # best-effort: if attn signature matches author
            try:
                attn_out = attn(block.norm1(x), attn_mask, rope)[0]
            except TypeError:
                # no rope-aware signature; ignore rope (not ideal)
                attn_out = attn(block.norm1(x))
        else:
            attn_out = attn(block.norm1(x), attn_mask) if attn_mask is not None else attn(block.norm1(x))

        # residual + layerscale
        if hasattr(block, "ls1"):
            x = x + block.ls1(attn_out)
        elif hasattr(block, "layer_scale1"):
            x = x + block.layer_scale1(attn_out)
        else:
            x = x + attn_out

        mlp_out = block.mlp(block.norm2(x))
        if hasattr(block, "ls2"):
            x = x + block.ls2(mlp_out)
        elif hasattr(block, "layer_scale2"):
            x = x + block.layer_scale2(mlp_out)
        else:
            x = x + mlp_out

        return x

    # -----------------------
    # forward (author style)
    # -----------------------
    def forward(self, x: torch.Tensor, per_layer=False
        ) -> Tuple[Union[torch.Tensor,List[torch.Tensor]], Union[torch.Tensor,List[torch.Tensor]]]:
        # input normalize (author wrapper)
        # if self.pixel_mean is not None and self.pixel_std is not None:
        #     x = (x - self.pixel_mean) / self.pixel_std

        # tokens include prefix for your posted DinoVisionTransformer
        tokens, (Hp, Wp) = self._prepare_tokens(x)

        rope = None
        if hasattr(self.backbone, "rope_embed") and self.backbone.rope_embed is not None:
            rope = self.backbone.rope_embed(H=Hp, W=Wp)

        attn_mask = None
        mask_logits_per_layer: List[torch.Tensor] = []
        class_logits_per_layer: List[torch.Tensor] = []

        blocks = self.backbone.blocks
        insert_at = len(blocks) - self.num_blocks

        for i, block in enumerate(blocks):
            # insert queries before last num_blocks
            if i == insert_at:
                q = self.q.weight[None, :, :].expand(tokens.shape[0], -1, -1)
                tokens = torch.cat((q, tokens), dim=1)  # [Q | prefix | patches]

            # masked attn for last num_blocks
            if self.masked_attn_enabled and i >= insert_at:
                x_norm = self.backbone.norm(tokens) if hasattr(self.backbone, "norm") else tokens
                mask_logits, class_logits = self._predict(x_norm, Hp, Wp)
                if per_layer:
                    mask_logits_per_layer.append(mask_logits)
                    class_logits_per_layer.append(class_logits)
                attn_mask = self._attn_mask(tokens, mask_logits, Hp, Wp, i)

            # step block (exact author-style if possible)
            try:
                tokens = self._block_step(block, tokens, attn_mask, rope)
            except Exception:
                # Fallback: call block directly (will NOT match masked-attn behavior)
                if rope is not None:
                    tokens = block(tokens, rope)
                else:
                    tokens = block(tokens)

        # final prediction
        x_norm = self.backbone.norm(tokens) if hasattr(self.backbone, "norm") else tokens
        mask_logits, class_logits = self._predict(x_norm, Hp, Wp)
        if not per_layer:
            return mask_logits, class_logits

        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return mask_logits_per_layer, class_logits_per_layer
    
encoder = vit_base()
encoder.init_weights()
model = DinoEoMT(encoder=encoder, num_queries=200, num_classes=1000)
convert_to_ternary(model)
summ(model)
mask_logits, class_logits = model.forward(torch.randn(1, 3, 640, 640))
print(mask_logits.shape)
print(class_logits.shape)





