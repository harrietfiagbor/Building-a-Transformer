import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from utils import Config
import einops

from transformer_lens.utils import gelu_new, tokenize_and_concatenate

class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch sequence d_model"]) -> Float[Tensor, "batch sequence d_model"]:
            residual_mean = residual.mean(dim=-1, keepdim=True)
            residual_std = (residual.var(dim=-1, keepdim=True, correction=1) + self.cfg.layer_norm_eps).sqrt()
            residual = residual - residual_mean/residual_std
            return residual * self.w + self.b
    
    

class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty(cfg.d_vocab, cfg.d_model))
        nn.init_normal(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch sequence"]) -> Float[Tensor, "batch sequence d_model"]:
        return self.W_E[tokens]
    

class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((self.cfg.n_ctx, self.cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch seqence"]) -> Float[Tensor, "batch sequence d_model"]:
         batch, seq_len =  tokens.shape
         return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)
          
     
     
class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=self.cfg.device))

    def forward(self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        ## obtain q, k, v vectors through linear transformation y = x @ W + b
        q  = (einops.einsum([normalized_resid_pre, self.W_Q], "batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head")) + self.b_Q
        k = (einops.einsum([normalized_resid_pre, self.W_K], "batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head")) + self.b_K
        v = (einops.einsum([normalized_resid_pre, self.W_V], "batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head")) + self.b_V
        # get attention scores by applying dot product over head dim
        attn_scores = einops.einsum([q, k], "batch query_pos n_head d_head, batch key_pos n_head d_head -> batch n_head query_pos key_pos")
        # scale to counter numerical instability
        attn_scores_scaled = attn_scores/t.sqrt(self.cfg.d_head)
        # mask future key positions
        attn_scores_masked = self.apply_causal_mask(attn_scores=attn_scores_scaled)
        # apply softmax to give probs
        attn_pattern = F.softmax(attn_scores_masked, dim=-1)
        # weigted sum of value vectors
        z = einops.einsum([attn_pattern, v], "batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head")

        attn_out = (einops.einsum([z, self.W_O], "batch query_pos n_head d_head, n_heads d_head d_model -> batch, query_pos d_model")) + self.b_O

        return attn_out

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        _, _, query_pos, key_pos = attn_scores.shape()
        pre_mask = t.ones((query_pos, key_pos), device=self.cfg.device)
        mask = t.triu(pre_mask, diagonal=0).bool()
        return attn_scores.masked_fill_(mask, self.IGNORE)
    

class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        pre = (
            einops.einsum([normalized_resid_mid, self.W_in], "batch posn d_model, d_model d_mlp -> batch posn d_model")
            ) + self.b_in
        post = gelu_new(pre)
        mlp_out = (
            einops.einsum([post, self.W_out], "batch posn d_model, d_mlp d_model -> batch posn d_model")
            ) + self.b_out
        return mlp_out
     
        
    


    

