# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import contextlib
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from bitlayers.dinov3.layers.bitlayers import Linear
from bitlayers.dinov3.utils import cat_keep_shapes, uncat_with_shapes

from .attention import CausalSelfAttention, SelfAttention
from .ffn_layers import Mlp
from .layer_scale import LayerScale  # , DropPath

torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.accumulated_cache_size_limit = 1024


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values).to(device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values).to(device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            # No batch dimension, do not index
            return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        """
        This is the reference implementation for a single tensor, matching what is done below for a list.
        We call the list op on [x] instead of this function.
        """
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(self, x_or_x_list, rope_or_rope_list=None) -> List[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            # return [self._forward(x, rope=rope) for x, rope in zip(x_or_x_list, rope_or_rope_list)]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError

class SelfAttentionTRMStage(nn.Module):
    """
    Transformer block stack + optional iterative refinement (AnytimeTRM-like).

    Core (no refinement):
      forward(x, rope) == apply the SelfAttentionBlock stack to x (passing rope through).

    Refinement (if used via forward_with_refinement):
      latent <- core( combine(x, solution, latent) + role_latent ) repeated num_latent_steps
      solution <- core( combine(solution, latent) + role_solution ) once

    Notes:
      - SelfAttentionBlock only accepts (x_or_x_list, rope_or_rope_list=None).
        Therefore, this stage does NOT support passing attn_mask/key_padding_mask
        unless your SelfAttention implementation encodes masking via rope or internal state.
    """

    def __init__(
        self,
        depth: int = 0,
        dim: int = 0,
        num_heads: int = 0,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Optional[Callable[..., nn.Module]] = None,
        ffn_layer: Optional[Callable[..., nn.Module]] = None,
        mask_k_bias: bool = False,
        init_std: float = 0.02,
        device=None,
        blocks_list=None,
    ):
        super().__init__()
        if blocks_list is None:
            blocks_list = [
                SelfAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    init_values=init_values,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    attn_class=attn_class,
                    ffn_layer=ffn_layer,
                    mask_k_bias=mask_k_bias,
                    device=device,
                )
                for _ in range(depth)
            ]
        else:
            dim = blocks_list[0].dim

        self.blocks = nn.ModuleList(blocks_list)


        # Default to the same classes SelfAttentionBlock expects by default
        if attn_class is None:
            attn_class = SelfAttention
        if ffn_layer is None:
            ffn_layer = Mlp

        self.d_model = dim

        # Combine projections (kept as in your original)
        self.combine_x_solution_latent = Linear(3 * dim, dim)
        self.combine_solution_latent = Linear(2 * dim, dim)

        # role embeddings: [2,1,1,D] => 0: update latent, 1: update solution
        self.role_embeddings = nn.Parameter(torch.empty(2, 1, 1, dim))
        nn.init.normal_(self.role_embeddings, std=init_std)

        # learned initial states (broadcast over seq_len): [1,1,D]
        self.init_solution_state = nn.Parameter(torch.empty(1, 1, dim))
        self.init_latent_state = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.init_solution_state, std=init_std)
        nn.init.normal_(self.init_latent_state, std=init_std)

    # -------- refinement helpers --------
    def init_refinement_states(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        solution = self.init_solution_state.expand(batch_size, seq_len, -1)
        latent = self.init_latent_state.expand(batch_size, seq_len, -1)

        if dtype is None:
            dtype = solution.dtype

        solution = solution.to(device=device, dtype=dtype)
        latent = latent.to(device=device, dtype=dtype)
        return solution, latent

    # -------- core forward (no refinement) --------
    def forward(
        self,
        x_or_x_list: Union[torch.Tensor, List[torch.Tensor]],
        rope_or_rope_list=None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = x_or_x_list        
        for i,block in enumerate(self.blocks):
            block:SelfAttentionBlock = block
            rope = None
            if rope_or_rope_list:
                rope = rope_or_rope_list[i]
            x = block(x, rope)
        return x

    def _refine_latent(
        self,
        x: torch.Tensor,
        solution: torch.Tensor,
        latent: torch.Tensor,
        rope=None,
    ) -> torch.Tensor:
        if type(x) is list and len(x)==1:x = x[0]
        fused = self.combine_x_solution_latent(torch.cat([x, solution, latent], dim=-1))
        fused = fused + self.role_embeddings[0]  # broadcasts [1,1,D] to [B,L,D]
        return self.forward([fused], rope)

    def _refine_solution(
        self,
        solution: torch.Tensor,
        latent: torch.Tensor,
        rope=None,
    ) -> torch.Tensor:
        fused = self.combine_solution_latent(torch.cat([solution, latent], dim=-1))
        fused = fused + self.role_embeddings[1]
        return self.forward([fused], rope)

    def refine_states(
        self,
        x: torch.Tensor,
        solution: torch.Tensor,
        latent: torch.Tensor,
        num_latent_steps: int,
        damping: float,
        rope=None,
        track_latent_grads: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx = contextlib.nullcontext() if track_latent_grads else torch.no_grad()
        with ctx:
            for _ in range(num_latent_steps):
                latent_next = self._refine_latent(x, solution, latent, rope=rope)
                if type(latent_next) is list and len(latent_next)==1:latent_next = latent_next[0]
                latent = latent + damping * (latent_next - latent)

        solution_next = self._refine_solution(solution, latent, rope=rope)
        if type(solution_next) is list and len(solution_next)==1:solution_next = solution_next[0]
        solution = solution + damping * (solution_next - solution)
        return solution, latent

    def forward_with_refinement(
        self,
        x: torch.Tensor,
        rope=None,
        num_latent_steps: int = 4,
        damping: float = 0.5,
        solution: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        *,
        return_latent: bool = False,
        track_latent_grads: bool = False,
    ):
        if type(x) is list:
            bsz, seq_len, dim = x[0].shape
            device = x[0].device
            dtype = x[0].dtype
        else:
            bsz, seq_len, dim = x.shape
            device = x.device
            dtype = x.dtype

        if solution is None or latent is None:
            solution, latent = self.init_refinement_states(bsz, seq_len, device, dtype=dtype)

        solution, latent = self.refine_states(
            x,
            solution,
            latent,
            num_latent_steps=num_latent_steps,
            damping=damping,
            rope=rope,
            track_latent_grads=track_latent_grads,
        )
        if type(x) is list and len(x)==1:solution = [solution]
        return (solution, latent) if return_latent else solution
    
class CausalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        self.attention = CausalSelfAttention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5
        self.attention.init_weights(init_attn_std, init_proj_std)
        self.attention_norm.reset_parameters()
        nn.init.normal_(self.feed_forward.fc1.weight, std=init_fc_std)
        nn.init.normal_(self.feed_forward.fc2.weight, std=init_proj_std)
        self.ffn_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
    ):

        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn









