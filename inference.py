from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from piano_transcription.utils import parse_yaml, write_midi
from train import get_model


def inference(args) -> None:

    # Arguments and parameters
    config_yaml = args.config_yaml
    ckpt_path = args.ckpt_path
    audio_path = args.audio_path
    midi_path = args.midi_path
    device = "cuda"

    # Default parameters
    configs = parse_yaml(config_yaml)
    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    fps = configs["fps"]
    clip_samples = round(clip_duration * sr)

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=ckpt_path
    ).to(device)

    # Load audio
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
    
    # Foward
    events = forward(model, audio, clip_samples, sr, fps)

    # Create directory
    Path(midi_path).parent.mkdir(parents=True, exist_ok=True)

    # Write out to MIDI
    write_midi(events, midi_path)


def forward(
    model: nn.Module, 
    audio: torch.Tensor, 
    clip_samples: int, 
    sr: float, 
    fps: int
):
    r"""Split audio into clips. Inference the result on each clip. Concatenate 
    the results.
    """

    device = next(model.parameters()).device
    audio_samples = audio.shape[0]

    start_sample = 0
    frame_rolls = []
    onset_rolls = []
    offset_rolls = []

    while start_sample < audio_samples:

        clip = audio[start_sample : start_sample + clip_samples]
        clip = librosa.util.fix_length(data=clip, size=clip_samples, axis=0)
        clip = torch.Tensor(clip[None, None, :]).to(device)  # shape: (b, c, t)

        with torch.no_grad():
            model.eval()
            output_dict = model(clip)

        frame_roll = output_dict["frame_roll"][0].cpu().numpy()
        onset_roll = output_dict["onset_roll"][0].cpu().numpy()
        offset_roll = output_dict["offset_roll"][0].cpu().numpy()

        frame_rolls.append(frame_roll[0 : -1, :])  # (frames_num, pitches_num)
        onset_rolls.append(onset_roll[0 : -1, :])  # (frames_num, pitches_num)
        offset_rolls.append(offset_roll[0 : -1, :])  # (frames_num, pitches_num)

        start_sample += clip_samples

        if False:
            visualize_rolls(frame_roll, onset_roll, offset_roll)

    frame_roll = np.concatenate(frame_rolls, axis=0)  # (frames_num, pitches_num)
    onset_roll = np.concatenate(onset_rolls, axis=0)  # (frames_num, pitches_num)
    offset_roll = np.concatenate(offset_rolls, axis=0)  # (frames_num, pitches_num)

    # Postprocess rolls to events
    events = rolls_to_events(onset_roll, fps)

    return events


def rolls_to_events(onset_roll: np.ndarray, fps: int) -> list[dict]:
    r"""Postprocess rolls to events."""

    T, P = onset_roll.shape
    events = []

    for t in range(1, T - 1):
        for p in range(P):
            if (onset_roll[t, p] > 0.1) and \
                (onset_roll[t, p] > onset_roll[t - 1, p]) and \
                (onset_roll[t, p] > onset_roll[t + 1, p]):

                onset = t / fps
                offset = onset + 0.5
                velocity = 100

                event = {
                    "onset": onset, 
                    "offset": offset,
                    "pitch": p,
                    "velocity": velocity
                }
                events.append(event)

    events.sort(key=lambda event: event["onset"])

    return events


def visualize_rolls(frame_roll, onset_roll, offset_roll):
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].matshow(frame_roll.T, origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(onset_roll.T, origin='lower', aspect='auto', cmap='jet')
    axs[2].matshow(offset_roll.T, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--midi_path', type=str, required=True)

    args = parser.parse_args()

    inference(args)