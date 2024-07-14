import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SequentialSampler
import numpy as np
import soundfile
from pathlib import Path
import torch.optim as optim
from tqdm import tqdm
import argparse
import random
from accelerate import Accelerator
import wandb
wandb.require("core")

from data.maestro import MAESTRO
from models.crnn import CRnn


def train(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    sr = 16000
    clip_duration = 10.
    extend_pedal = True
    batch_size = 16
    num_workers = 16
    pin_memory = True
    learning_rate = 1e-4
    test_step_frequency = 1000
    save_step_frequency = 1000
    training_steps = 10000
    debug = False
    wandb_log = True
    device = "cuda"

    filename = Path(__file__).stem
    classes_num = MAESTRO.pitches_num

    checkpoints_dir = Path("./checkpoints", filename, model_name)

    root = "/datasets/maestro-v3.0.0"

    if wandb_log:
        wandb.init(project="mini_piano_transcription") 

    # Dataset
    train_dataset = MAESTRO(
        root=root,
        split="train",
        sr=sr,
        clip_duration=clip_duration,
    )

    test_dataset = MAESTRO(
        root=root,
        split="test",
        sr=sr,
        clip_duration=clip_duration,
    )

    # Sampler
    train_sampler = InfiniteSampler(train_dataset)
    
    test_sampler = SequentialSampler(test_dataset)
    
    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        sampler=test_sampler,
        num_workers=1, 
        pin_memory=pin_memory
    )

    # Model
    model = get_model(model_name, classes_num)
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Prepare for multiprocessing
    accelerator = Accelerator()

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # Move data to device
        audio = data["audio"].to(device)
        target_onset_roll = data["onset_roll"].to(device)

        # Play the audio
        if debug:
            play_audio(mixture, target)

        # Forward
        model.train()
        output_dict = model(audio=audio)

        # Loss
        loss = bce_loss(output_dict["onset_roll"], target_onset_roll)

        # Optimize
        optimizer.zero_grad()   # Reset all parameter.grad to 0
        accelerator.backward(loss)     # Update all parameter.grad
        optimizer.step()    # Update all parameters based on all parameter.grad

        if step % test_step_frequency == 0:

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:

                if accelerator.num_processes == 1:
                    val_model = model
                else:
                    val_model = model.module

                print("step: {}, loss: {:.3f}".format(step, loss.item()))
                test_loss = validate(val_model, test_dataloader)
                print("Test loss: {}".format(test_loss))

                if wandb_log:
                    wandb.log(
                        data={"test_loss": test_loss},
                        step=step
                    )

        # Save model
        if step % save_step_frequency == 0:

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:

                checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
                torch.save(model.state_dict(), checkpoint_path)
                print("Save model to {}".format(checkpoint_path))

                checkpoint_path = Path(checkpoints_dir, "latest.pth")
                torch.save(model.state_dict(), Path(checkpoint_path))
                print("Save model to {}".format(checkpoint_path))

        if step == training_steps:
            break


def get_model(model_name, classes_num):
    if model_name == "CRnn":
        return CRnn(classes_num)
    else:
        raise NotImplementedError


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


def play_audio(mixture, target):
    soundfile.write(file="tmp_mixture.wav", data=mixture[0].cpu().numpy().T, samplerate=44100)
    soundfile.write(file="tmp_target.wav", data=target[0].cpu().numpy().T, samplerate=44100)
    from IPython import embed; embed(using=False); os._exit(0)


class InfiniteSampler:
    def __init__(self, dataset):

        self.indexes = list(range(len(dataset)))
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


def validate(model, dataloader):

    device = next(model.parameters()).device

    losses = []

    for step, data in tqdm(enumerate(dataloader)):
        
        audio = torch.Tensor(data["audio"]).to(device)
        target_onset_roll = torch.Tensor(data["onset_roll"]).to(device)

        with torch.no_grad():
            model.eval()
            output_dict = model(audio=audio)

        loss = bce_loss(output_dict["onset_roll"], target_onset_roll)
        losses.append(loss.item())

    loss = np.mean(losses)

    return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="CRnn")
    args = parser.parse_args()

    train(args)