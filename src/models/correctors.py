"""Sequence models that learn the KF/RTS position residual.

    LSTMCorrector  (B4)  two-layer LSTM, trained from scratch
    GPT4TS         (B6)  patch embedding -> frozen GPT-2 backbone + LoRA -> regression head

Both consume ``[B, in_len, in_ch]`` and return ``[B, out_len, 3]`` of delta-P.
Heavy dependencies (transformers / peft) are imported lazily so the rest of the
project stays importable on a CPU-only machine without them installed.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# B4: LSTM
# --------------------------------------------------------------------------- #
class LSTMCorrector(nn.Module):
    def __init__(self, in_ch: int = 10, hidden: int = 128, layers: int = 2,
                 out_len: int = 64, out_ch: int = 3, dropout: float = 0.1):
        super().__init__()
        self.out_len, self.out_ch = out_len, out_ch
        self.encoder = nn.LSTM(in_ch, hidden, num_layers=layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_len * out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(x)
        y = self.head(h[:, -1])                       # last time step
        return y.view(-1, self.out_len, self.out_ch)


# --------------------------------------------------------------------------- #
# B6: GPT4TS-style patch transformer with LoRA
# --------------------------------------------------------------------------- #
class PatchEmbed(nn.Module):
    """[B, in_ch, in_len] -> [B, n_patch, d_model] with a per-patch linear map."""

    def __init__(self, in_ch: int, patch_len: int, d_model: int, in_len: int):
        super().__init__()
        assert in_len % patch_len == 0, "in_len must be divisible by patch_len"
        self.patch_len = patch_len
        self.n_patch = in_len // patch_len
        self.proj = nn.Linear(in_ch * patch_len, d_model)

    def forward(self, x):
        b = x.shape[0]
        p = x.unfold(2, self.patch_len, self.patch_len)          # [B,C,n_patch,L]
        p = p.permute(0, 2, 1, 3).reshape(b, self.n_patch, -1)   # [B,n_patch,C*L]
        return self.proj(p)


class GPT4TS(nn.Module):
    """Patch embedding + (LoRA-adapted) GPT-2 + flatten regression head.

    ``inputs_embeds`` bypasses only GPT-2's *token* embedding (``wte``); the
    model still adds its positional embedding (``wpe``) internally, so nothing
    is added by hand here -- adding it again would double the position signal.
    """

    def __init__(self, in_ch: int = 10, in_len: int = 256, patch_len: int = 16,
                 d_model: int = 768, out_len: int = 64, out_ch: int = 3,
                 model_name: str = "distilgpt2", lora: bool = True,
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1,
                 head: str = "flatten"):
        super().__init__()
        from transformers import GPT2Config, GPT2Model

        self.embed = PatchEmbed(in_ch, patch_len, d_model, in_len)
        self.n_patch = self.embed.n_patch
        backbone = GPT2Model.from_pretrained(model_name)
        # GPT-2 without inputs_embeds: we supply embeddings, so mask the token table
        self.backbone = backbone
        self.head_kind = head
        self.out_len, self.out_ch = out_len, out_ch

        for p in backbone.parameters():
            p.requires_grad = False
        if lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["c_attn", "c_proj"], bias="none",
            )
            self.backbone = get_peft_model(backbone, cfg)
        if head == "flatten":
            self.head = nn.Sequential(
                nn.GELU(),
                nn.Flatten(),
                nn.Linear(self.n_patch * d_model, out_len * out_ch),
            )
        else:  # mean-pool head, ~100x fewer head parameters
            self.head = nn.Sequential(
                nn.GELU(),
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, out_len * out_ch),
            )

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_len, in_ch] -> channels-first for patching.  The backbone
        # adds wpe itself, so this is the only positional signal.
        h = self.embed(x.transpose(1, 2))                     # [B,n_patch,d]
        out = self.backbone(inputs_embeds=h).last_hidden_state
        if self.head_kind == "flatten":
            y = self.head(out)
        else:
            y = self.head(out.mean(dim=1))
        return y.view(-1, self.out_len, self.out_ch)


def build_model(name: str, cfg: dict, **kw):
    ds = cfg.get("model", {})
    if name == "lstm":
        return LSTMCorrector(in_ch=ds.get("in_ch", 20), hidden=ds.get("hidden", 128),
                             out_len=kw.get("out_len", 64), out_ch=3)
    if name == "gpt4ts":
        return GPT4TS(in_ch=ds.get("in_ch", 20), in_len=kw.get("in_len", 256),
                      patch_len=ds.get("patch_len", 16), d_model=ds.get("d_model", 768),
                      out_len=kw.get("out_len", 64), out_ch=3,
                      model_name=ds.get("backbone", "distilgpt2"),
                      lora=ds.get("lora", True), head=ds.get("head", "flatten"))
    raise ValueError(f"unknown model {name!r}")
