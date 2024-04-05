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

from data.tokenizers import Tokenizer
from models.audiollama import LLaMAConfig, AudioLlama
from data.maestro import MaestroStringProcessor
from data.io import events_to_notes, notes_to_midi


def inference_in_batch(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    segment_seconds = 4.
    device = "cuda"
    sample_rate = 16000
    max_token_len = 1024
    top_k = 1
    batch_size = 15

    tokenizer = Tokenizer()

    # Load checkpoint
    enc_model_name = "CRnn3_onset_offset_vel"
    checkpoint_path = Path("checkpoints/train_llama_ft2/AudioLlama/step=20000_encoder.pth")
    enc_model = get_model(enc_model_name)
    enc_model.load_state_dict(torch.load(checkpoint_path))
    enc_model.to(device)

    for param in enc_model.parameters():
        param.requires_grad = False

    # Load checkpoint
    checkpoint_path = Path("checkpoints/train_llama_ft2/AudioLlama/step=20000.pth")
    config = LLaMAConfig(
        block_size=401 + max_token_len, 
        vocab_size=tokenizer.vocab_size, 
        padded_vocab_size=tokenizer.vocab_size, 
        n_layer=6, 
        n_head=16, 
        n_embd=1024, 
        audio_n_embd=4096
    )
    model = AudioLlama(config)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Data
    root = "/datasets/maestro-v3.0.0/maestro-v3.0.0"
    meta_csv = Path(root, "maestro-v3.0.0.csv")
    meta_data = load_meta(meta_csv, split="test")
    audio_paths = [Path(root, name) for name in meta_data["audio_filename"]]
    midi_paths = [Path(root, name) for name in meta_data["midi_filename"]]

    # audio_paths = ["/home/qiuqiangkong/datasets/maestro-v2.0.0/2009/MIDI-Unprocessed_11_R1_2009_01-05_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_04_WAV.wav"]
    # midi_paths = ["/home/qiuqiangkong/datasets/maestro-v2.0.0/2009/MIDI-Unprocessed_11_R1_2009_01-05_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_04_WAV.midi"]

    # Load audio. Change this path to your favorite song.
    # audio_paths = ["/home/qiuqiangkong/datasets/maestro-v2.0.0/2014/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--2.wav"]

    output_dir = Path("pred_midis", model_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    string_processor = MaestroStringProcessor(
        label=False,
        onset=True,
        offset=True,
        sustain=False,
        velocity=True,
        pedal_onset=False,
        pedal_offset=False,
        pedal_sustain=False,
    )

    precs = []
    recalls = []
    f1s = []

    idx = tokenizer.stoi("<sos>")
    idx = torch.LongTensor(idx * np.ones((batch_size, 1))).to(device)

    for audio_idx in range(len(audio_paths)):

        print(audio_idx)
        t1 = time.time()

        audio_path = audio_paths[audio_idx]

        audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)
        # (channels_num, audio_samples)

        audio_samples = audio.shape[-1]
        bgn = 0
        clip_idx = 0
        clip_seconds = batch_size * segment_seconds
        segment_samples = int(segment_seconds * sample_rate)
        clip_samples = segment_samples * batch_size

        # onset_rolls = []
        all_notes = []

        while bgn < audio_samples:

            # clip is 60s (4s * 15 batches)
            clip = audio[bgn : bgn + clip_samples]
            clip = librosa.util.fix_length(data=clip, size=clip_samples, axis=-1)

            segments = librosa.util.frame(clip, frame_length=segment_samples, hop_length=segment_samples).T

            bgn_sec = bgn / sample_rate
            print("Processing: {:.1f} s".format(bgn_sec))

            segments = torch.Tensor(segments).to(device)

            with torch.no_grad():
                enc_model.eval()
                audio_emb = enc_model(segments)["emb"]

            clip_strings = []

            # Generate in batch
            with torch.no_grad():
                model.eval()
                pred_tokens = model.generate_in_batch(
                    audio_emb=audio_emb, 
                    idx=idx, 
                    max_new_tokens=max_token_len,
                    end_token=tokenizer.stoi("<eos>")
                ).data.cpu().numpy()

                # Convert tokens to strings
                for k in range(pred_tokens.shape[0]):
                    for i, token in enumerate(pred_tokens[k]):
                        if token == tokenizer.stoi("<eos>"):
                            break                    

                    print(i)
                    new_pred_tokens = pred_tokens[k, 0 : i + 1]

                    strings = tokenizer.tokens_to_strings(new_pred_tokens)

                    for string in strings[1 : -1]:

                        if "time=" in string:
                            ti = float(re.search('time=(.*)', string).group(1))
                            ti = clip_idx * clip_seconds + k * segment_seconds + ti
                            clip_strings.append("time={:.2f}".format(ti))

                        else:
                            clip_strings.append(string)

            # 60s audio's string
            clip_strings = ["<sos>"] + clip_strings + ["<eos>"]
            clip_events = string_processor.strings_to_events(clip_strings)
            clip_notes = events_to_notes(clip_events)
            
            bgn += clip_samples
            clip_idx += 1

            all_notes.extend(clip_notes)

        notes_to_midi(all_notes, "_zz.mid")
        soundfile.write(file="_zz.wav", data=audio, samplerate=16000)
        
        est_midi_path = Path(output_dir, "{}.mid".format(Path(audio_path).stem))
        notes_to_midi(all_notes, str(est_midi_path))
        
        ref_midi_path = midi_paths[audio_idx]
        ref_intervals, ref_pitches = parse_midi(ref_midi_path)
        est_intervals, est_pitches = parse_midi(est_midi_path)

        note_precision, note_recall, note_f1, _ = \
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals, 
            ref_pitches=ref_pitches, 
            est_intervals=est_intervals, 
            est_pitches=est_pitches, 
            onset_tolerance=0.05, 
            offset_ratio=None,
        )

        print("P: {:.3f}, R: {:.3f}, F1: {:.3f}, time: {:.3f} s".format(note_precision, note_recall, note_f1, time.time() - t1))
        precs.append(note_precision)
        recalls.append(note_recall)
        f1s.append(note_f1)
        

        a1 = list(np.concatenate((ref_intervals, ref_pitches[:, None]),axis=-1))
        a1.sort(key=lambda x: x[0])  # sort by keys
        from IPython import embed; embed(using=False); os._exit(0)

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

    for note in notes:
        intervals.append([note.start, note.end])
        pitches.append(note.pitch)

    return np.array(intervals), np.array(pitches)


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
