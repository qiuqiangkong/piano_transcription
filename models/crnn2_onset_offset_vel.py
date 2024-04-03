import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlibrosa.stft import Spectrogram, LogmelFilterBank

from models.crnn2 import ConvBlock, init_layer, init_bn, init_gru


class AcousticModelCRnn8DropoutWithEmbedding(nn.Module):
    def __init__(self, classes_num, midfeat, momentum):
        super(AcousticModelCRnn8DropoutWithEmbedding, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=128, momentum=momentum)

        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)

        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)

        self.fc = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, time_steps, classes_num)
        """

        x = self.conv_block1(input, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        
        (x, _) = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        output = torch.sigmoid(self.fc(x))
        return output, x


class CRnn2_onset_offset_vel(nn.Module):
    def __init__(self):
        super(CRnn2_onset_offset_vel, self).__init__()

        frames_per_second = 100
        classes_num = 128

        sample_rate = 16000
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        fmin = 30
        fmax = sample_rate // 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        midfeat = 1792
        momentum = 0.01

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
            hop_length=hop_size, win_length=window_size, window=window, 
            center=center, pad_mode=pad_mode, freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
            n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, 
            amin=amin, top_db=top_db, freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)

        self.frame_model = AcousticModelCRnn8DropoutWithEmbedding(classes_num, midfeat, momentum)
        self.reg_onset_model = AcousticModelCRnn8DropoutWithEmbedding(classes_num, midfeat, momentum)
        self.reg_offset_model = AcousticModelCRnn8DropoutWithEmbedding(classes_num, midfeat, momentum)
        self.velocity_model = AcousticModelCRnn8DropoutWithEmbedding(classes_num, midfeat, momentum)

        self.reg_onset_gru = nn.GRU(input_size=128 * 2, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)

        self.frame_gru = nn.GRU(input_size=128 * 3, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_gru(self.reg_onset_gru)
        init_gru(self.frame_gru)
        init_layer(self.reg_onset_fc)
        init_layer(self.frame_fc)
 
    def forward(self, audio):
        """
        Args:
          input: (batch_size, data_length)

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """

        x = self.spectrogram_extractor(audio)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        frame_output, _ = self.frame_model(x)  # (batch_size, time_steps, classes_num)
        reg_onset_output, _ = self.reg_onset_model(x)  # (batch_size, time_steps, classes_num)
        reg_offset_output, offset_emb = self.reg_offset_model(x)    # (batch_size, time_steps, classes_num)
        velocity_output, vel_emb = self.velocity_model(x)    # (batch_size, time_steps, classes_num)
 
        # Use velocities to condition onset regression
        x = torch.cat((reg_onset_output, (reg_onset_output ** 0.5) * velocity_output.detach()), dim=2)
        (x, _) = self.reg_onset_gru(x)
        onset_emb = F.dropout(x, p=0.5, training=self.training, inplace=False)
        reg_onset_output = torch.sigmoid(self.reg_onset_fc(onset_emb))
        """(batch_size, time_steps, classes_num)"""

        # Use onsets and offsets to condition frame-wise classification
        x = torch.cat((frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2)
        (x, _) = self.frame_gru(x)
        frame_emb = F.dropout(x, p=0.5, training=self.training, inplace=False)
        frame_output = torch.sigmoid(self.frame_fc(frame_emb))  # (batch_size, time_steps, classes_num)
        """(batch_size, time_steps, classes_num)"""

        output_dict = {
            'reg_onset_output': reg_onset_output, 
            'reg_offset_output': reg_offset_output, 
            'frame_output': frame_output, 
            'velocity_output': velocity_output,
            'onset_emb': onset_emb,
            'offset_emb': offset_emb,
            'frame_emb': frame_emb,
            'vel_emb': vel_emb
        }

        return output_dict