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

from data.tokenizers import Tokenizer
from models.audiollama import LLaMAConfig, AudioLlama
from data.maestro import MaestroStringProcessor
from data.io import events_to_notes, notes_to_midi


def inference(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    segment_seconds = 4.
    device = "cuda"
    sample_rate = 16000
    max_token_len = 768
    top_k = 1

    tokenizer = Tokenizer()

    # Load checkpoint
    enc_model_name = "CRnn3"
    checkpoint_path = Path("checkpoints/train_llama_ft/AudioLlama/step=100000_encoder.pth")
    enc_model = get_model(enc_model_name)
    enc_model.load_state_dict(torch.load(checkpoint_path))
    enc_model.to(device)

    for param in enc_model.parameters():
        param.requires_grad = False

    # Load checkpoint
    checkpoint_path = Path("checkpoints/train_llama_ft/AudioLlama/step=100000.pth")
    config = LLaMAConfig(
        block_size=401 + max_token_len, 
        vocab_size=tokenizer.vocab_size, 
        padded_vocab_size=tokenizer.vocab_size, 
        n_layer=6, 
        n_head=16, 
        n_embd=1024, 
        audio_n_embd=1024
    )
    model = AudioLlama(config)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Data
    root = "/datasets/maestro-v2.0.0/maestro-v2.0.0"
    meta_csv = Path(root, "maestro-v2.0.0.csv")
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
    
    idx = tokenizer.stoi("<sos>")
    idx = torch.LongTensor([[idx]]).to(device)

    # for audio_idx, audio_path in enumerate(audio_paths):
    for audio_idx in range(len(audio_paths)):
    # for audio_idx in range(2, len(audio_paths)):

        print(audio_idx)
        audio_path = audio_paths[audio_idx]

        audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)
        # (channels_num, audio_samples)

        audio_samples = audio.shape[-1]
        bgn = 0
        segment_samples = int(segment_seconds * sample_rate)

        # onset_rolls = []
        all_notes = []

        # Do separation
        while bgn + segment_samples < audio_samples:

            # bgn = 136 * sample_rate
            bgn_sec = bgn / sample_rate
            print("Processing: {:.1f} s".format(bgn_sec))

            # Cut segments
            segment = audio[bgn : bgn + segment_samples]
            segment = librosa.util.fix_length(data=segment, size=segment_samples, axis=-1)
            segment = torch.Tensor(segment).to(device)[None, :]

            with torch.no_grad():
                enc_model.eval()
                audio_emb = enc_model(segment)["onset_emb"]

            # Separate a segment
            with torch.no_grad():
                model.eval()
                pred_tokens = model.generate(
                    audio_emb=audio_emb, 
                    idx=idx, 
                    max_new_tokens=max_token_len,
                    top_k=top_k
                )[0].data.cpu().numpy()

                for i, token in enumerate(pred_tokens):
                    if token == tokenizer.stoi("<eos>"):
                        break

                pred_tokens = pred_tokens[0 : i + 1]
                    
                # for ix in new_outputs:
                #     print(tokenizer.itos(ix))

                # from IPython import embed; embed(using=False); os._exit(0)
                strings = tokenizer.tokens_to_strings(pred_tokens)
                events = string_processor.strings_to_events(strings)
                notes = events_to_notes(events)

                for note in notes:
                    note.start += bgn_sec
                    note.end += bgn_sec

                all_notes.extend(notes)

                # notes_to_midi(notes, "_zz.mid")
                # soundfile.write(file="_zz.wav", data=audio[bgn : bgn + segment_samples], samplerate=16000)

                # if bgn / sample_rate > 30:
                #     break

            bgn += segment_samples

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
            offset_ratio=None,)

        print("P: {:.3f}, R: {:.3f}, F1: {:.3f}".format(note_precision, note_recall, note_f1))
        precs.append(note_precision)
        recalls.append(note_recall)
        f1s.append(note_f1)

    print("----------")
    print("Avg Prec: {:.3f}".format(np.mean(precs)))
    print("Avg Recall: {:.3f}".format(np.mean(recalls)))
    print("Avg F1: {:.3f}".format(np.mean(f1s)))


def inference_in_batch(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    segment_seconds = 4.
    device = "cuda"
    sample_rate = 16000
    max_token_len = 768
    top_k = 1
    batch_size = 15

    tokenizer = Tokenizer()

    # Load checkpoint
    enc_model_name = "CRnn3"
    checkpoint_path = Path("checkpoints/train_llama_ft/AudioLlama/step=50000_encoder.pth")
    enc_model = get_model(enc_model_name)
    enc_model.load_state_dict(torch.load(checkpoint_path))
    enc_model.to(device)

    for param in enc_model.parameters():
        param.requires_grad = False

    # Load checkpoint
    checkpoint_path = Path("checkpoints/train_llama_ft/AudioLlama/step=50000.pth")
    config = LLaMAConfig(
        block_size=401 + max_token_len, 
        vocab_size=tokenizer.vocab_size, 
        padded_vocab_size=tokenizer.vocab_size, 
        n_layer=6, 
        n_head=16, 
        n_embd=1024, 
        audio_n_embd=1024
    )
    model = AudioLlama(config)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Data
    root = "/datasets/maestro-v2.0.0/maestro-v2.0.0"
    meta_csv = Path(root, "maestro-v2.0.0.csv")
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

    idx = tokenizer.stoi("<sos>")
    idx = torch.LongTensor(idx * np.ones((batch_size, 1))).to(device)

    # for audio_idx, audio_path in enumerate(audio_paths):
    for audio_idx in range(len(audio_paths)):
    # for audio_idx in range(2, len(audio_paths)):

        print(audio_idx)
        t1 = time.time()

        audio_path = audio_paths[audio_idx]

        audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)
        # (channels_num, audio_samples)

        audio_samples = audio.shape[-1]
        bgn = 0
        segment_samples = int(segment_seconds * sample_rate)
        clip_samples = segment_samples * batch_size

        # onset_rolls = []
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
                    max_new_tokens=max_token_len,
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
            offset_ratio=None,)

        print("P: {:.3f}, R: {:.3f}, F1: {:.3f}, time: {:.3f} s".format(note_precision, note_recall, note_f1, time.time() - t1))
        precs.append(note_precision)
        recalls.append(note_recall)
        f1s.append(note_f1)
        asdf

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
