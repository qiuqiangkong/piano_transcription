from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

from piano_transcription.models2.attention import Block
from piano_transcription.models2.rope import RoPE
from vector_quantize_pytorch import FSQ


@dataclass
class Config:
    name: str
    sr: int = 16000
    n_fft: int = 2048
    hop_length: int = 160
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 384


class FlextokFSQ(nn.Module):
    def __init__(self, config): 
        
        super().__init__()

        self.patch_frames = 16
        self.register_size = 4
        self.pitches_num = 128
        self.config = config
        # self.head_dim = config.n_embd // config.n_head
        n_embd = config.n_embd
        n_head = config.n_head
        head_dim = config.n_embd // config.n_head

        self.pos_embedder = nn.Embedding(self.patch_frames + self.register_size, n_embd)

        self.rope = RoPE(head_dim, max_len=2000)

        self.mel_extractor = MelSpectrogram(
            sample_rate=config.sr,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            f_min=0.,
            f_max=config.sr / 2,
            n_mels=256,
            power=2.0,
            normalized=True,
        )

        self.enc_register = nn.Parameter(0.1 * torch.randn(self.register_size, n_embd))
        self.dec_register = nn.Parameter(0.1 * torch.randn(self.patch_frames, n_embd))

        self.enc_proj = nn.Linear(256, n_embd)
        self.enc_transformer = nn.ModuleList(Block(n_embd, n_head) for _ in range(3))
        self.enc_flex = nn.ModuleList(Block(n_embd, n_head) for _ in range(3))
        self.enc_proj2 = nn.Linear(n_embd, 6)

        levels = [4, 4, 4, 4, 4, 4]
        self.num_codes = np.prod(levels)
        self.fsq = FSQ(levels)

        self.dec_proj2 = nn.Linear(6, n_embd)
        self.dec_flex = nn.ModuleList(Block(n_embd, n_head) for _ in range(3))        
        self.dec_transformer = nn.ModuleList(Block(n_embd, n_head) for _ in range(3))
        self.dec_proj = nn.Linear(n_embd, 3 * self.pitches_num)

    def forward(
        self, 
        x: Tensor, 
    ) -> Tensor:
        """Model

        Args:
            t: (b,), random time steps between 0. and 1.
            x: (b, c, t, f)
            cond_dict: dict

        Outputs:
            output: (b, c, t, f)
        """
        


        # Feature
        x = self.mel_extractor(x)  # (b, c, f, t)
        x = rearrange(x, 'b c f t -> b c t f')  # (b, c, t, f)
        B, C, T = x.shape[0 : 3]
        
        x, pad_t = self.pad_tensor(x)
        x = torch.log10(torch.clamp(x, 1e-10))

        # Encoder layers
        x = rearrange(x, 'b c t f -> b t (c f)')
        x = self.enc_proj(x)  # (b, t, d)

        for block in self.enc_transformer:
            x = block(x, rope=self.rope, pos=None, mask=None)

        # Encoder flex
        x = rearrange(x, 'b (t1 t2) d -> (b t1) t2 d', t2=self.patch_frames)  # (b, t, d)
        x = self.cat_register(x, self.enc_register)  # (b, t, d)
        x = self.add_pos_emb(x)  # (b, t, d)

        for block in self.enc_flex:
            x = block(x, rope=self.rope, pos=None, mask=None)

        x = x[:, -self.register_size :, :]  # (b, t, d)
        x = self.enc_proj2(x)

        # VQ
        x, code = self.fsq(x)

        # Decoder flex
        x = self.dec_proj2(x)
        x = self.cat_register(x, self.dec_register)
        x = self.add_pos_emb(x)

        for block in self.dec_flex:
            x = block(x, rope=self.rope, pos=None, mask=None)

        x = x[:, self.register_size :, :]
        x = rearrange(x, '(b t1) t2 d -> b (t1 t2) d', b=B)  # (b, t, d)

        # Decoder layers
        for block in self.dec_transformer:
            x = block(x, rope=self.rope, pos=None, mask=None)

        x = torch.sigmoid(self.dec_proj(x))  # (b, t, d)
        
        x = x[:, 0 : T, :]
        frame_roll, onset_roll, offset_roll = torch.chunk(x, chunks=3, dim=2)
        
        output = {
            "frame_roll": frame_roll,  # (b, t, p)
            "onset_roll": onset_roll,  # (b, t, p)
            "offset_roll": offset_roll,  # (b, t, p)
            "code": code
        }

        return output

    def pad_tensor(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=201, f)
        
        Outpus:
            output: E.g., (b, c, t=204, f)
        """

        # Pad last frames, e.g., 201 -> 204
        T = x.shape[2]
        pad_t = -T % self.patch_frames
        x = F.pad(x, pad=(0, 0, 0, pad_t))

        return x, pad_t

    def add_pos_emb(self, x):
        pos_emb = torch.arange(0, x.shape[1], device=x.device)[None, :]
        pos_emb = self.pos_embedder(pos_emb)
        out = x + pos_emb
        return out

    def cat_register(self, x, register):
        reg = register[None, :, :].repeat(x.shape[0], 1, 1)
        out = torch.cat((x, reg), dim=1)  # (b, t, c)
        return out