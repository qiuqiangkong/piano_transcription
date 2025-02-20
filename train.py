from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from audidata.collate.default import collate_fn
from audidata.samplers import InfiniteSampler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from piano_transcription.utils import LinearWarmUp, parse_yaml


def train(args) -> None:

    # Arguments
    wandb_log = not args.no_log
    config_path = args.config
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]

    # Checkpoints directory
    config_name = Path(config_path).stem
    ckpts_dir = Path("./checkpoints", filename, config_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = get_dataset(configs, split="train")

    # Sampler
    train_sampler = InfiniteSampler(train_dataset)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=configs["train"]["batch_size_per_device"], 
        sampler=train_sampler,
        num_workers=configs["train"]["num_workers"], 
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=configs["train"]["resume_ckpt_path"]
    ).to(device)

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=model.parameters()
    )

    # Logger
    if wandb_log:
        wandb.init(project="piano_transcription", name="{}".format(config_name))

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # 1.1 Data preparation
        audio = data["audio"].to(device)  # (b, c, t)
        
        target_dict = {
            "frame_roll": data["frame_roll"].to(device),  # (b, t, k)
            "onset_roll": data["onset_roll"].to(device),  # (b, t, k)
            "offset_roll": data["offset_roll"].to(device)  # (b, t, k)
        }

        # 1.2 Forward
        model.train()
        output_dict = model(audio)

        # 1.3 Loss
        loss = bce_loss(output_dict, target_dict)
        
        # 1.4 Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        loss.backward()  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad

        # 1.5 Learning rate scheduler
        if scheduler:
            scheduler.step()

        if step % 100 == 0:
            print(loss)
        
        # ------ 2. Evaluation ------
        # 2.1 Evaluate
        if step % configs["train"]["test_every_n_steps"] == 0:
            # TODO. No validation now.
            pass

        # 2.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0:
            
            ckpt_path = Path(ckpts_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

        if step == configs["train"]["training_steps"]:
            break
        
        
def get_dataset(
    configs: dict, 
    split: str
) -> Dataset:
    r"""Get datasets."""

    from audidata.io.crops import RandomCrop
    from audidata.transforms import Mono

    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    fps = configs["fps"]
    datasets_split = "{}_datasets".format(split)

    for name in configs[datasets_split].keys():
    
        if name == "MAESTRO":

            from audidata.datasets import MAESTRO
            from audidata.transforms.midi import PianoRoll

            # Dataset
            dataset = MAESTRO(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=RandomCrop(clip_duration=clip_duration, end_pad=clip_duration - 0.1),
                transform=Mono(),
                load_target=True,
                extend_pedal=True,
                target_transform=PianoRoll(fps=fps, pitches_num=128),
            )
            return dataset

        else:
            raise ValueError(name)


def get_model(
    configs: dict, 
    ckpt_path: str
) -> nn.Module:
    r"""Initialize LLM decoder."""

    name = configs["model"]["name"]

    if name == "Conformer2D":

        from piano_transcription.models.conformer2d import (Conformer2D,
                                                            Conformer2DConfig)

        config = Conformer2DConfig(
            sr=configs["sample_rate"],
            n_fft=configs["model"]["n_fft"],
            hop_length=configs["model"]["hop_length"],
        )

        model = Conformer2D(config)

    else:
        raise ValueError(name)    

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)

    return model


def bce_loss(output_dict: dict, target_dict: dict) -> torch.float:

    frame_loss = F.binary_cross_entropy(output_dict["frame_roll"], target_dict["frame_roll"])
    onset_loss = F.binary_cross_entropy(output_dict["onset_roll"], target_dict["onset_roll"])
    offset_loss = F.binary_cross_entropy(output_dict["offset_roll"], target_dict["offset_roll"])
    loss = frame_loss + onset_loss + offset_loss

    return loss


def get_optimizer_and_scheduler(
    configs: dict, 
    params: list[torch.Tensor]
) -> tuple[optim.Optimizer, None | optim.lr_scheduler.LambdaLR]:
    r"""Get optimizer and scheduler."""

    lr = float(configs["train"]["lr"])
    warm_up_steps = configs["train"]["warm_up_steps"]
    optimizer_name = configs["train"]["optimizer"]

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(params=params, lr=lr)

    if warm_up_steps:
        lr_lambda = LinearWarmUp(warm_up_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler

'''
def validate(
    configs: dict,
    model: nn.Module
    valid_steps=50
) -> float:
    r"""Validate the model on part of data."""

    device = next(audio_encoder.parameters()).device
    losses = []

    batch_size = configs["train"]["batch_size_per_device"]
    skip_n = max(1, len(dataset) // valid_steps)

    for idx in range(0, len(dataset), skip_n):
        print("{}/{}".format(idx, len(dataset)))

        # ------ 1. Data preparation ------
        # 1.0 Collate data to batch
        data = [dataset[i] for i in range(idx, min(idx + batch_size, len(dataset)))]
        data = collate_fn(data)

        # 1.1 Prepare audio, question, and answering
        audio, question, answering = get_audio_question_answering(data)
        # audio: (b, c, t), question: (b, t), answering: (b, t)

        # 1.3 Tokenize question text to IDs
        audio = audio.to(device)
        audio_latent = audio_encoder.encode(audio=audio, train_mode=False)  # shape: (b, t, d)

        # 1.4 Tokenize answering text to IDs
        question_ids = tokenizer.texts_to_ids(
            texts=question, 
            fix_length=configs["max_question_len"]
        ).to(device)  # shape: (b, t)

        # 1.5 Remove padded columns to speed up training
        answering_ids = tokenizer.texts_to_ids(
            texts=answering, 
            fix_length=configs["max_answering_len"]
        ).to(device)  # shape: (b, t)

        # 1.6 Prepare inputs
        if configs["train"]["remove_padded_columns"]:
            answering_ids = remove_padded_columns(
                ids=answering_ids, 
                pad_token_id=tokenizer.pad_token_id
            )

        # Prepare inputs
        seqs = [audio_latent, question_ids, answering_ids]
        seq_types = ["audio", "id", "id"]
        loss_types = [None, None, "ce"]

        # ------ 2. Training ------
        # 2.1 Forward
        with torch.no_grad():
            llm.eval()
            output_seqs = llm(
                seqs=seqs,
                seq_types=seq_types,
                mask=None
            )  # list

        # 2.2 Prepare data for next ID prediction
        output_seqs = [seq[:, 0 : -1] for seq in output_seqs]
        target_seqs = [seq[:, 1 :] for seq in seqs]
        
        # 2.3 Loss
        loss = ce_loss(
            output_seqs=output_seqs, 
            target_seqs=target_seqs, 
            loss_types=loss_types,
            ignore_index=tokenizer.pad_token_id
        )

        losses.append(loss.item())
        
    return np.mean(losses)
'''

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)