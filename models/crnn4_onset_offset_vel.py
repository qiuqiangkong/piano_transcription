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


class AcousticModel(nn.Module):
    def __init__(self):
        super(AcousticModel, self).__init__()

        self.conv1 = ConvBlock(in_channels=1, out_channels=48)
        self.conv2 = ConvBlock(in_channels=48, out_channels=64)
        self.conv3 = ConvBlock(in_channels=64, out_channels=96)
        self.conv4 = ConvBlock(in_channels=96, out_channels=128)

        self.gru = nn.GRU(
            input_size=1792, 
            hidden_size=512, 
            num_layers=2, 
            bias=True, 
            batch_first=True, 
            dropout=0., 
            bidirectional=True
        )

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, time_steps, classes_num)
        """

        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # shape: (B, C, T, Freq)

        x = rearrange(x, 'b c t f -> b t (c f)')

        x, _ = self.gru(x)

        return x


class RollModel(nn.Module):
    def __init__(self):
        super(RollModel, self).__init__()

        self.gru = nn.GRU(
            input_size=1024, 
            hidden_size=512, 
            num_layers=2, 
            bias=True, 
            batch_first=True, 
            dropout=0., 
            bidirectional=True
        )

        self.fc = nn.Linear(1024, 128)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, time_steps, classes_num)
        """

        emb, _ = self.gru(input)
        output = torch.sigmoid(self.fc(emb))

        return output, emb


class CRnn4_onset_offset_vel(nn.Module):
    def __init__(self):
        super(CRnn4_onset_offset_vel, self).__init__()

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

        self.acoustic_model = AcousticModel()
        self.onset_model = RollModel()
        self.offset_model = RollModel()
        self.frame_model = RollModel()
        self.vel_model = RollModel()

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

        x = self.acoustic_model(x)

        onset_roll, onset_emb = self.onset_model(x)
        offset_roll, offset_emb = self.offset_model(x)
        frame_roll, frame_emb = self.frame_model(x)
        vel_roll, vel_emb = self.vel_model(x)

        emb = torch.cat((onset_emb, offset_emb, frame_emb, vel_emb), dim=-1)

        output_dict = {
            "onset_roll": onset_roll,
            "offset_roll": offset_roll,
            "frame_roll": frame_roll,
            "velocity_roll": vel_roll,
            "onset_emb": onset_emb,
            "offset_emb": offset_emb,
            "frame_emb": frame_emb,
            "velocity_emb": vel_emb,
            "emb": emb,
        }
        
        return output_dict