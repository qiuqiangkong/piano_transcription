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


@dataclass
class Config:
    name: str
    sr: int = 16000
    n_fft: int = 2048
    hop_length: int = 160
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 384


class Transformer(nn.Module):
    def __init__(self, config): 
        
        super().__init__()

        self.pitches_num = 128
        self.config = config
        self.head_dim = config.n_embd // config.n_head

        self.rope = RoPE(self.head_dim, max_len=2000)

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

        self.enc_proj = nn.Linear(256, config.n_embd)
        self.enc_transformer = nn.ModuleList(Block(config) for _ in range(config.n_layer))

        self.dec_transformer = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.dec_proj = nn.Linear(config.n_embd, 3 * self.pitches_num)
        # self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))

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
        x = rearrange(x, 'b c f t -> b t (c f)')  # (b, t, d)
        
        # Encoder
        x = self.enc_proj(x)  # (b, t, d)

        for block in self.enc_transformer:
            x = block(x, rope=self.rope, pos=None, mask=None)

        # VQ here
        pass

        # Decoder
        for block in self.dec_transformer:
            x = block(x, rope=self.rope, pos=None, mask=None)

        x = torch.sigmoid(self.dec_proj(x))  # (b, t, d)
        frame_roll, onset_roll, offset_roll = torch.chunk(x, 3, dim=2)
        
        output = {
            "frame_roll": frame_roll,  # (b, t, p)
            "onset_roll": onset_roll,  # (b, t, p)
            "offset_roll": offset_roll  # (b, t, p)
        }

        return output
