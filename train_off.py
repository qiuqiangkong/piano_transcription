import torch
import torch.nn.functional as F
import time
import random
import librosa
import numpy as np
import soundfile
import matplotlib.pyplot as plt
from pathlib import Path
import torch.optim as optim
from data.maestro import Maestro
from data.collate import collate_fn
from models.crnn import CRnn
from tqdm import tqdm
import museval
import argparse
import wandb

from data.tokenizers import Tokenizer
from losses import regress_onset_offset_frame_velocity_bce, regress_onset_offset_frame_velocity_bce2


def train(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    device = "cuda"
    batch_size = 16
    num_workers = 32
    save_step_frequency = 10000
    training_steps = 200000
    debug = False
    filename = Path(__file__).stem
    segment_seconds = 4.
    wandb_log = False

    if wandb_log:
        wandb.init(project="mini_piano_transcription")

    checkpoints_dir = Path("./checkpoints", filename, model_name)
    
    # root = "/datasets/maestro-v2.0.0/maestro-v2.0.0"
    root = "/datasets/maestro-v3.0.0/maestro-v3.0.0"

    tokenizer = Tokenizer()

    # Dataset
    dataset = Maestro(
        root=root,
        split="train",
        segment_seconds=segment_seconds,
        tokenizer=tokenizer,
        max_token_len=1024,
    )

    # Sampler
    sampler = Sampler(dataset_size=len(dataset))

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers, 
        pin_memory=True
    )

    # Model
    model = get_model(model_name)
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for step, data in enumerate(tqdm(dataloader)):

        audio = data["audio"].to(device)
        offset_roll = data["offset_roll"].to(device)

        # soundfile.write(file="_zz.wav", data=audio.cpu().numpy()[0], samplerate=16000)

        # fig, axs = plt.subplots(2,1, sharex=True)
        # axs[0].matshow(onsets_roll.cpu().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
        # axs[1].matshow(data["frames_roll"].cpu().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
        # plt.savefig("_zz.pdf")
        # asdf
        

        # Play the audio.
        if debug:
            play_audio(mixture, target)

        

        model.train()
        output_dict = model(audio=audio)

        # fig, axs = plt.subplots(2,1, sharex=True)
        # axs[0].matshow(onsets_roll.cpu().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
        # axs[1].matshow(output_dict["offset_roll"].data.cpu().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
        # plt.savefig("_zz.pdf")

        # from IPython import embed; embed(using=False); os._exit(0)

        loss = bce_loss(output_dict["offset_roll"], offset_roll)

        optimizer.zero_grad()    
        loss.backward()

        optimizer.step()

        if step % 100 == 0:
            print("step: {}, loss: {:.3f}".format(step, loss.item()))

            if wandb_log:
                wandb.log({"train loss": loss.item()})

        # Save model
        if step % save_step_frequency == 0:
            checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            checkpoint_path = Path(checkpoints_dir, "latest.pth")
            torch.save(model.state_dict(), Path(checkpoint_path))
            print("Save model to {}".format(checkpoint_path))

        if step == training_steps:
            break


def get_model(model_name):

    if model_name == "CRnn3_off":
        from models.crnn3 import CRnn3_off
        return CRnn3_off()
    
    else:
        raise NotImplementedError


class Sampler:
    def __init__(self, dataset_size):
        self.indexes = list(range(dataset_size))
        random.shuffle(self.indexes)
        
    def __iter__(self):

        pointer = 0

        while True:

            if pointer == len(self.indexes):
                random.shuffle(self.indexes)
                pointer = 0
                
            index = self.indexes[pointer]
            pointer += 1

            yield index


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


def play_audio(mixture, target):
    soundfile.write(file="tmp_mixture.wav", data=mixture[0].cpu().numpy().T, samplerate=44100)
    soundfile.write(file="tmp_target.wav", data=target[0].cpu().numpy().T, samplerate=44100)
    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="CRnn3_off")
    args = parser.parse_args()

    train(args)