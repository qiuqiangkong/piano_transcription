import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram
from einops import rearrange
import numpy as np

# from models.fourier import Fourier


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
            x: (batch_size, in_channels, time_steps, freq_bins)

        Returns:
            output: (batch_size, out_channels, time_steps // 2, freq_bins // 2)
        """

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x))) 

        output = F.avg_pool2d(x, kernel_size=(1, 2))
        
        return output 


class CRnn3(nn.Module):
    def __init__(self):
        super(CRnn3, self).__init__()

        self.mel_extractor = MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            hop_length=160,
            f_min=30.,
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

        self.onset_fc = nn.Linear(1024, 128)

    def forward(self, audio):
        """Separation model.

        Args:
            mixture: (batch_size, channels_num, samples_num)

        Outputs:
            output: (batch_size, channels_num, samples_num)
        """
        x = self.mel_extractor(audio)
        # shape: (B, Freq, T)

        x = torch.log10(torch.clamp(x, 1e-8))

        x = rearrange(x, 'b f t -> b t f')
        x = x[:, None, :, :]
        # shape: (B, 1, T, Freq)

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

    def cut_image(self, x):
        """Cut a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (B, C, 201, 1025)
        
        Outpus:
            output: E.g., (B, C, 208, 1024)
        """

        B, C, T, Freq = x.shape

        pad_len = (
            int(np.ceil(T / self.downsample_ratio)) * self.downsample_ratio
            - T
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))

        output = x[:, :, :, 0 : Freq - 1]

        return output

    def patch_image(self, x, time_steps):
        """Patch a spectrum to the original shape. E.g.,
        
        Args:
            x: E.g., (B, C, 208, 1024)
        
        Outpus:
            output: E.g., (B, C, 201, 1025)
        """
        x = F.pad(x, pad=(0, 1))

        output = x[:, :, 0 : time_steps, :]

        return output