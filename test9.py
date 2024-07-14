from typing import Dict, List
import librosa
import torch
from torch.utils.data import DataLoader
import soundfile
import matplotlib.pyplot as plt

from data.maestro import MAESTRO
from data.tokenizers import *
from data.collate import collate_list_fn, CollateToken
from data.target_transforms import Note2Token


def add():

    audio_path = "/datasets/maestro-v3.0.0/maestro-v3.0.0/2004/MIDI-Unprocessed_XP_08_R1_2004_01-02_ORIG_MID--AUDIO_08_R1_2004_01_Track01_wav.wav"

    sample_rate = librosa.get_samplerate(path=audio_path)

    from IPython import embed; embed(using=False); os._exit(0)


def add2():

    root = "/datasets/maestro-v3.0.0"

    # Dataset
    dataset = MAESTRO(
        root=root,
        split="train",
        sr=16000,
        clip_duration=10.,
    )

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=4, 
        num_workers=0, 
    )

    for data in dataloader:

        audio = data["audio"]
        frame_roll = data["frame_roll"]
        onset_roll = data["onset_roll"]
        offset_roll = data["offset_roll"]
        velocity_roll = data["velocity_roll"]
        break

    n = 0
    soundfile.write(file="_zz.wav", data=audio[n, 0].cpu().numpy(), samplerate=16000)

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(20, 10))
    axs[0].matshow(frame_roll[n].cpu().numpy().T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[1].matshow(onset_roll[n].cpu().numpy().T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[2].matshow(offset_roll[n].cpu().numpy().T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[3].matshow(velocity_roll[n].cpu().numpy().T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


def add3():

    root = "/datasets/maestro-v3.0.0"

    sr = 16000
    clip_duration = 10.
    max_tokens = 4096

    # Tokenizer. Users may define their own tokenizer
    tokenizer = ConcatTokenizer([
        SpecialTokenizer(),
        NameTokenizer(),
        TimeTokenizer(),
        PitchTokenizer(),
        VelocityTokenizer()
    ])

    # Target transform. Users may define their own target transform
    target_transform = Note2Token(
        clip_duration=clip_duration, 
        tokenizer=tokenizer, 
        max_tokens=max_tokens
    )

    # Dataset
    dataset = MAESTRO(
        root=root,
        split="train",
        sr=sr,
        clip_duration=clip_duration,
        target_transform=target_transform
    )

    # Collate. Users may define their own collate
    collate_fn = CollateToken()

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=16, 
        num_workers=0, 
        collate_fn=collate_fn
    )

    for data in dataloader:

        from IPython import embed; embed(using=False); os._exit(0)





if __name__ == '__main__':

    add2()