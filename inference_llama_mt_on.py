import torch
import time
import pickle
import librosa
import numpy as np
import pandas as pd
import soundfile
import pretty_midi
from pathlib import Path
import torch.optim as optim
from data.maestro import Maestro
from data.collate import collate_fn
from models.crnn import CRnn
from tqdm import tqdm
import museval
import argparse
import matplotlib.pyplot as plt
from train import get_model
import mir_eval
import re

from data.tokenizers import Tokenizer2
from models.audiollama_qa import LLaMAConfig, AudioLlamaQA
from data.maestro import MaestroStringProcessor
from data.io import events_to_notes, notes_to_midi, read_single_track_midi, write_notes_to_midi, fix_length


def inference_in_batch(args):

    # Arguments
    # model_name = args.model_name
    filename = Path(__file__).stem

    # Default parameters
    segment_seconds = 4.
    device = "cuda"
    sample_rate = 16000
    top_k = 1
    batch_size = 15
    question_token_len = 20
    answer_token_len = 512
    segment_samples = int(segment_seconds * sample_rate)

    tokenizer = Tokenizer2()

    # Load checkpoint
    enc_model_name = "CRnn3"
    checkpoint_path = Path("checkpoints/train_llama_mt_on/AudioLlama/step=60000_encoder.pth")
    enc_model = get_model(enc_model_name)
    enc_model.load_state_dict(torch.load(checkpoint_path))
    enc_model.to(device)

    for param in enc_model.parameters():
        param.requires_grad = False

    # Load checkpoint
    checkpoint_path = Path("checkpoints/train_llama_mt_on/AudioLlama/step=60000.pth")
    config = LLaMAConfig(
        block_size=401 + question_token_len + answer_token_len + 1, 
        vocab_size=tokenizer.vocab_size, 
        padded_vocab_size=tokenizer.vocab_size, 
        n_layer=6, 
        n_head=16, 
        n_embd=1024, 
        audio_n_embd=1024
    )
    model = AudioLlamaQA(config)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Data
    root = "/datasets/maestro-v3.0.0/maestro-v3.0.0"
    meta_csv = Path(root, "maestro-v3.0.0.csv")
    meta_data = load_meta(meta_csv, split="test")
    audio_paths = [Path(root, name) for name in meta_data["audio_filename"]]
    midi_paths = [Path(root, name) for name in meta_data["midi_filename"]]

    onset_midis_dir = Path("pred_midis", filename)
    Path(onset_midis_dir).mkdir(parents=True, exist_ok=True)

    string_processor = MaestroStringProcessor(
        label=False,
        onset=True,
        offset=False,
        sustain=False,
        velocity=False,
        pedal_onset=False,
        pedal_offset=False,
        pedal_sustain=False,
    )

    precs = []
    recalls = []
    f1s = []

    precs = []
    recalls = []
    f1s = []

    # idx = tokenizer.stoi("<sos>")
    # idx = torch.LongTensor(idx * np.ones((batch_size, 1))).to(device)

    for audio_idx in range(len(audio_paths)):
    # for audio_idx in range(110, len(audio_paths)):

        print(audio_idx)
        t1 = time.time()

        audio_path = audio_paths[audio_idx]

        # from IPython import embed; embed(using=False); os._exit(0)

        audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)
        # (channels_num, audio_samples)

        audio_samples = audio.shape[-1]
        bgn = 0
        segment_samples = int(segment_seconds * sample_rate)
        clip_samples = segment_samples * batch_size

        ##
        question_strings = [
            "<sos>",
            "task=onset",
            "<eos>",
        ]
        question_tokens = tokenizer.strings_to_tokens(question_strings)
        question_tokens = np.array(fix_length(
            x=question_tokens, 
            max_len=question_token_len, 
            constant_value=tokenizer.stoi("<pad>")
        ))
        question_tokens = np.repeat(question_tokens[None, :], repeats=batch_size, axis=0)
        question_tokens = torch.LongTensor(question_tokens).to(device)

        answer_tokens = np.array([tokenizer.stoi("<sos>")])
        answer_tokens = np.repeat(answer_tokens[None, :], repeats=batch_size, axis=0)
        answer_tokens = torch.LongTensor(answer_tokens).to(device)
        
        idx = torch.cat((question_tokens, answer_tokens), dim=1)

        all_notes = []

        while bgn < audio_samples:

            clip = audio[bgn : bgn + clip_samples]
            clip = librosa.util.fix_length(data=clip, size=clip_samples, axis=-1)

            segments = librosa.util.frame(clip, frame_length=segment_samples, hop_length=segment_samples).T

            bgn_sec = bgn / sample_rate
            print("Processing: {:.1f} s".format(bgn_sec))

            segments = torch.Tensor(segments).to(device)

            with torch.no_grad():
                enc_model.eval()
                audio_emb = enc_model(segments)["onset_emb"]

            # 
            with torch.no_grad():
                model.eval()
                pred_tokens = model.generate_in_batch(
                    audio_emb=audio_emb, 
                    idx=idx, 
                    max_new_tokens=answer_token_len,
                    end_token=tokenizer.stoi("<eos>")
                ).data.cpu().numpy()

                for k in range(pred_tokens.shape[0]):
                    for i, token in enumerate(pred_tokens[k]):
                        if token == tokenizer.stoi("<eos>"):
                            break                    

                    new_pred_tokens = pred_tokens[k, 0 : i + 1]

                    strings = tokenizer.tokens_to_strings(new_pred_tokens)
                    events = string_processor.strings_to_events(strings)
                    notes = events_to_notes(events)

                    for note in notes:
                        note.start += bgn_sec + k * segment_seconds
                        note.end += bgn_sec + k * segment_seconds

                    all_notes.extend(notes)

            bgn += clip_samples

        notes_to_midi(all_notes, "_zz.mid")
        # soundfile.write(file="_zz.wav", data=audio, samplerate=16000)
        
        est_midi_path = Path(onset_midis_dir, "{}.mid".format(Path(audio_path).stem))
        notes_to_midi(all_notes, str(est_midi_path))
        
        ref_midi_path = midi_paths[audio_idx]
        ref_intervals, ref_pitches, ref_vels = parse_midi(ref_midi_path)
        est_intervals, est_pitches, est_vels = parse_midi(est_midi_path)

        note_precision, note_recall, note_f1, _ = \
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals, 
            ref_pitches=ref_pitches, 
            est_intervals=est_intervals, 
            est_pitches=est_pitches, 
            onset_tolerance=0.05, 
            offset_ratio=None,)

        print("P: {:.3f}, R: {:.3f}, F1: {:.3f}, time: {:.3f} s".format(note_precision, note_recall, note_f1, time.time() - t1))
        precs.append(note_precision)
        recalls.append(note_recall)
        f1s.append(note_f1)
        # from IPython import embed; embed(using=False); os._exit(0)

    print("----------")
    print("Avg Prec: {:.3f}".format(np.mean(precs)))
    print("Avg Recall: {:.3f}".format(np.mean(recalls)))
    print("Avg F1: {:.3f}".format(np.mean(f1s)))


def load_meta(meta_csv, split):

    df = pd.read_csv(meta_csv, sep=',')

    indexes = df["split"].values == split

    midi_filenames = df["midi_filename"].values[indexes]
    audio_filenames = df["audio_filename"].values[indexes]

    meta_data = {
        "midi_filename": midi_filenames,
        "audio_filename": audio_filenames
    }

    return meta_data


def parse_midi(midi_path):

    midi_data = pretty_midi.PrettyMIDI(str(midi_path))

    notes = midi_data.instruments[0].notes

    intervals = []
    pitches = []
    velocities = []

    for note in notes:
        intervals.append([note.start, note.end])
        pitches.append(note.pitch)
        velocities.append(note.velocity)

    return np.array(intervals), np.array(pitches), np.array(velocities)


def deduplicate_array(array):

    new_array = []

    for pair in array:
        time = pair[0]
        pitch = pair[1]
        if (time - 1, pitch) not in new_array:
            new_array.append((time, pitch))

    return np.array(new_array)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="AudioLlama")
    args = parser.parse_args()

    # inference(args)
    inference_in_batch(args)
