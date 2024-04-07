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
from data.io import events_to_notes
from models.crnn import CRnn
from tqdm import tqdm
import museval
import argparse
import wandb
from losses import regress_onset_offset_frame_velocity_bce2

from data.tokenizers import Tokenizer
from models.audiollama import LLaMAConfig, AudioLlama


def train(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    device = "cuda"
    # batch_size = 16
    batch_size = 9
    num_workers = 32
    evaluate_step_frequency = 2000
    save_step_frequency = 10000
    training_steps = 300001
    debug = False
    filename = Path(__file__).stem
    segment_seconds = 4.
    lr = 2e-4
    # max_token_len = 256
    max_token_len = 1024
    wandb_log = True

    checkpoints_dir = Path("./checkpoints", filename, model_name)
    
    root = "/datasets/maestro-v3.0.0/maestro-v3.0.0"

    if wandb_log:
        wandb.init(
            project="mini_piano_transcription",
        )

    tokenizer = Tokenizer()

    # Dataset
    train_dataset = Maestro(
        root=root,
        split="train",
        segment_seconds=segment_seconds,
        tokenizer=tokenizer,
        max_token_len=max_token_len + 1,
    )

    test_dataset = Maestro(
        root=root,
        split="train",
        segment_seconds=segment_seconds,
        tokenizer=tokenizer,
        max_token_len=max_token_len + 1,
    )

    # Sampler
    train_sampler = Sampler(dataset_size=len(train_dataset))
    eval_train_sampler = Sampler(dataset_size=len(train_dataset))
    eval_test_sampler = Sampler(dataset_size=len(test_dataset))

    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers, 
        pin_memory=True
    )

    eval_train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=eval_train_sampler,
        collate_fn=collate_fn,
        num_workers=0, 
        pin_memory=True
    )

    eval_test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        sampler=eval_test_sampler,
        collate_fn=collate_fn,
        num_workers=0, 
        pin_memory=True
    )

    # Load checkpoint
    enc_model_name = "CRnn3_onset_offset_vel"
    checkpoint_path = Path("checkpoints/train/{}/step=100000.pth".format(enc_model_name))
    # enc_model_name = "CRnn3"
    # checkpoint_path = Path("checkpoints/train/{}/step=90000.pth".format(enc_model_name))
    enc_model = get_model(enc_model_name)
    enc_model.load_state_dict(torch.load(checkpoint_path))
    enc_model.to(device)

    # for param in enc_model.parameters():
    #     param.requires_grad = False
    
    # Model
    # model = get_model(model_name)
    '''
    config = LLaMAConfig(
        block_size=401 + max_token_len, 
        vocab_size=tokenizer.vocab_size, 
        padded_vocab_size=tokenizer.vocab_size, 
        n_layer=6, 
        n_head=6, 
        n_embd=384, 
        audio_n_embd=1024
    )
    '''
    
    config = LLaMAConfig(
        block_size=401 + max_token_len, 
        vocab_size=tokenizer.vocab_size, 
        padded_vocab_size=tokenizer.vocab_size, 
        n_layer=12, 
        n_head=16, 
        n_embd=768, 
        audio_n_embd=4096
    )
    '''
    config = LLaMAConfig(
        block_size=401 + max_token_len, 
        vocab_size=tokenizer.vocab_size, 
        padded_vocab_size=tokenizer.vocab_size, 
        n_layer=8, 
        n_head=16, 
        n_embd=768, 
        audio_n_embd=4096
    )
    '''
    model = AudioLlama(config)
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(list(enc_model.parameters()) + list(model.parameters()), lr=lr)

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    tmp = []

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        audio = data["audio"].to(device)
        onset_roll = data["onset_roll"].to(device)

        # tokens_num = min(max(data["tokens_num"]), max_token_len)
        tokens_num = max_token_len
        input_token = data["token"][:, 0 : tokens_num - 1].to(device)
        target_token = data["token"][:, 1 : tokens_num].to(device)
        # print(tokens_num, input_token.shape, target_token.shape)

        # from IPython import embed; embed(using=False); os._exit(0)
        # strings = tokenizer.tokens_to_strings(input_token[0].data.cpu().numpy())
        # events = data["string_processor"][0].strings_to_events(strings)
        # notes = events_to_notes(events)
        # print(notes)

        # Play the audio.
        if debug:
            play_audio(mixture, target)

        optimizer.zero_grad()

        enc_model.train()
        model.train()
        # audio_emb = enc_model(audio)["emb"]
        enc_output_dict = enc_model(audio)
        audio_emb = enc_output_dict["emb"]
        logits, dec_loss = model(audio_emb=audio_emb, idx=input_token, target=target_token)
        
        target_dict = {
            "onset_roll": data["onset_roll"].to(device),
            "offset_roll": data["offset_roll"].to(device),
            "frame_roll": data["frame_roll"].to(device),
            "velocity_roll": data["velocity_roll"].to(device),
        }
        enc_loss = regress_onset_offset_frame_velocity_bce2(enc_output_dict, target_dict)

        loss = dec_loss + enc_loss

        loss.backward()

        optimizer.step()

        if step % evaluate_step_frequency == 0:
            print("Evaluating ...")
            train_loss = validate(enc_model, model, eval_train_dataloader, max_token_len)
            test_loss = validate(enc_model, model, eval_test_dataloader, max_token_len)
            print("--- step: {} ---".format(step))
            print("Train loss: {:.4f}".format(train_loss))
            print("Test loss: {:.4f}".format(test_loss))

            if wandb_log:
                wandb.log({
                    "train loss": train_loss,
                    "test loss": test_loss
                })
        
        # Save model
        if step % save_step_frequency == 0:
            checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            checkpoint_path = Path(checkpoints_dir, "latest.pth")
            torch.save(model.state_dict(), Path(checkpoint_path))
            print("Save model to {}".format(checkpoint_path))

            #
            checkpoint_path = Path(checkpoints_dir, "step={}_encoder.pth".format(step))
            torch.save(enc_model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

        if step == training_steps:
            break

    from IPython import embed; embed(using=False); os._exit(0)


def get_model(model_name):
    if model_name == "CRnn":
        return CRnn()
    elif model_name == "CRnn2":
        from models.crnn2 import CRnn2
        return CRnn2()
    elif model_name == "CRnn3":
        from models.crnn3 import CRnn3
        return CRnn3()
    elif model_name == "CRnn3_onset_offset_vel":
        from models.crnn3_onset_offset_vel import CRnn3_onset_offset_vel
        return CRnn3_onset_offset_vel()
    elif model_name == "CRnn4_onset_offset_vel":
        from models.crnn4_onset_offset_vel import CRnn4_onset_offset_vel
        return CRnn4_onset_offset_vel()
    elif model_name == "AudioLlama":
        from models.audiollama import AudioLlama
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


def validate(enc_model, model, dataloader, max_token_len):

    pred_ids = []
    target_ids = []
    device = next(model.parameters()).device
    losses = []

    for step, data in tqdm(enumerate(dataloader)):

        if step == 5:
            break

        audio = data["audio"].to(device)
        onset_roll = data["onset_roll"].to(device)
        tokens_num = max_token_len
        input_token = data["token"][:, 0 : tokens_num - 1].to(device)
        target_token = data["token"][:, 1 : tokens_num].to(device)

        with torch.no_grad():
            enc_model.eval()
            audio_emb = enc_model(audio)["emb"]

        with torch.no_grad():
            model.eval()
            logits, loss = model(audio_emb=audio_emb, idx=input_token, target=target_token)

        losses.append(loss.item())

    return np.mean(losses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="AudioLlama")
    args = parser.parse_args()

    train(args)