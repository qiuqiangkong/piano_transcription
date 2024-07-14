import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram
from einops import rearrange
import numpy as np


class CRnn(nn.Module):
    def __init__(self, classes_num):
        super().__init__()

        self.mel_extractor = MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            hop_length=160,
            f_min=0.,
            f_max=8000,
            n_mels=229,
            power=2.0,
            normalized=True,
        )

        self.conv1 = ConvBlock(in_channels=1, out_channels=48)
        self.conv2 = ConvBlock(in_channels=48, out_channels=64)
        self.conv3 = ConvBlock(in_channels=64, out_channels=96)
        self.conv4 = ConvBlock(in_channels=96, out_channels=128)

        self.gru = nn.GRU(
            input_size=1792, 
            hidden_size=512, 
            num_layers=3, 
            bias=True, 
            batch_first=True, 
            dropout=0., 
            bidirectional=True
        )

        self.onset_fc = nn.Linear(1024, classes_num)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Separation model.

        Args:
            mixture: (batch_size, channels_num, samples_num)

        Outputs:
            output: (batch_size, channels_num, samples_num)
        """
        x = self.mel_extractor(audio)
        # shape: (B, Freq, T)
        
        x = torch.log10(torch.clamp(x, 1e-10))
        
        x = rearrange(x, 'b c f t -> b c t f')
        # shape: (B, 1, T, F)

        # from IPython import embed; embed(using=False); os._exit(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # shape: (B, C, T, Freq)

        x = rearrange(x, 'b c t f -> b t (c f)')

        x, _ = self.gru(x)

        onset_roll = torch.sigmoid(self.onset_fc(x))

        output_dict = {
            "onset_emb": x,
            "onset_roll": onset_roll
        }

        return output_dict


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, F)

        Returns:
            output: (B, C, T // 2, F // 2)
        """

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x))) 

        output = F.avg_pool2d(x, kernel_size=(1, 2))
        
        return output 
