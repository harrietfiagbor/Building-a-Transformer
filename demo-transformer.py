import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from utils import Config
import einops

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
    
class apply_causal_mask():
     def __init__(self, cfg: Config):
          super().__init__()
          self.cfg = cfg
          self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=self.cfg.device))

     def forward(self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
          _, _, query_pos, key_pos = attn_scores.shape()
          pre_mask = ((query_pos, key_pos), device= self.cfg.device)
          mask = t.triu(pre_mask, diagonal=0).bool()
          return attn_scores.masked_fill_(mask, self.IGNORE)
     
        
    


    

