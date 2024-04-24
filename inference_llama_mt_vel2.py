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
from models_bd.models import Note_pedal, Regress_onset_offset_frame_velocity_CRNN

from data.tokenizers import Tokenizer2
from models.audiollama_qa import LLaMAConfig, AudioLlamaQA
from data.maestro import MaestroStringProcessor
from data.io import events_to_notes, notes_to_midi, read_single_track_midi, write_notes_to_midi, fix_length


def inference_in_batch(args):

    # Arguments
    # model_name = args.model_name
    filename = Path(__file__).stem

    # Default parameters
    segment_seconds = 10.
    device = "cuda"
    sample_rate = 16000
    # max_token_len = 20
    top_k = 1
    batch_size = 15
    frames_num = 1001
    segment_samples = int(segment_seconds * sample_rate)
    question_token_len = 1024
    answer_token_len = 512

    tokenizer = Tokenizer2()

    # Load checkpoint
    # enc_model_name = "Note_pedal"
    checkpoint_path = Path("checkpoints/train_llama_mt_vel2/AudioLlama/step=10000_encoder.pth")
    enc_model = Note_pedal()
    enc_model.load_state_dict(torch.load(checkpoint_path))
    enc_model.to(device)

    for param in enc_model.parameters():
        param.requires_grad = False

    # Load checkpoint
    checkpoint_path = Path("checkpoints/train_llama_mt_vel2/AudioLlama/step=10000.pth")
    config = LLaMAConfig(
        block_size=frames_num + question_token_len + answer_token_len + 1, 
        vocab_size=tokenizer.vocab_size, 
        padded_vocab_size=tokenizer.vocab_size, 
        n_layer=6, 
        n_head=16, 
        n_embd=1024, 
        audio_n_embd=264
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

    onset_midis_dir = Path("pred_midis", "inference_llama_mt_on4")
    vel_midis_dir = Path("pred_midis_vel", filename)
    Path(vel_midis_dir).mkdir(parents=True, exist_ok=True)

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

    # idx = tokenizer.stoi("<sos>")
    # idx = torch.LongTensor(idx * np.ones((batch_size, 1))).to(device)

    for audio_idx in range(len(audio_paths)):
    # for audio_idx in range(110, len(audio_paths)):

        print(audio_idx)
        t1 = time.time()

        audio_path = audio_paths[audio_idx]

        audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)
        # (channels_num, audio_samples)

        audio_samples = audio.shape[-1]
        audio_seconds = audio_samples / sample_rate
        
        onset_midi_path = Path(onset_midis_dir, "{}.mid".format(Path(audio_path).stem))

        onset_midi_data = pretty_midi.PrettyMIDI(str(onset_midi_path))
        pred_onset_notes = onset_midi_data.instruments[0].notes

        bgn_sec = 0
        all_notes = []

        while bgn_sec < audio_seconds:

            print(bgn_sec)

            end_sec = bgn_sec + segment_seconds
            bgn_sample = int(bgn_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)

            segment = audio[bgn_sample : end_sample]
            segment = librosa.util.fix_length(data=segment, size=segment_samples, axis=-1)

            candidate_notes = []
            for note in pred_onset_notes:
                if bgn_sec <= note.start < end_sec:
                    candidate_notes.append(note)

            pointer = 0
            while pointer < len(candidate_notes):

                _N = min(batch_size, len(candidate_notes) - pointer)

                segments = np.repeat(segment[None, :], repeats=_N, axis=0)
                segments = torch.Tensor(segments).to(device)

                with torch.no_grad():
                    enc_model.eval()
                    audio_emb = enc_model(segments)["onoffvel_emb"]

                q_tokens = []
                a_tokens = []
                
                for note in candidate_notes[pointer : pointer + batch_size]:
                    question_strings = [
                        "<sos>",
                        "task=velocity",
                        "time={}".format(note.start - bgn_sec),
                        "pitch={}".format(note.pitch),
                        "<eos>",
                    ]

                    question_tokens = tokenizer.strings_to_tokens(question_strings)
                    question_tokens_num = len(question_tokens)

                    question_tokens = np.array(fix_length(
                        x=question_tokens, 
                        max_len=question_token_len,
                        constant_value=tokenizer.stoi("<pad>")
                    ))
                    q_tokens.append(question_tokens)
                    a_tokens.append([tokenizer.stoi("<sos>")])

                q_tokens = np.array(q_tokens)
                a_tokens = np.array(a_tokens)

                idx = np.concatenate((q_tokens, a_tokens), axis=-1)
                idx = torch.LongTensor(idx).to(device)
                # from IPython import embed; embed(using=False); os._exit(0)
                
                with torch.no_grad():
                    model.eval()
                    pred_tokens = model.generate_in_batch(
                        audio_emb=audio_emb, 
                        idx=idx, 
                        max_new_tokens=answer_token_len,
                        end_token=tokenizer.stoi("<eos>")
                    ).data.cpu().numpy()

                from IPython import embed; embed(using=False); os._exit(0)

                for k in range(len(pred_tokens)):
                    vel = tokenizer.itos(pred_tokens[k][4])
                    try:
                        vel = int(re.search('velocity=(.*)', vel).group(1))
                    except:
                        print(vel)
                        vel = 100
                        # from IPython import embed; embed(using=False); os._exit(0)

                    note = pretty_midi.Note(
                        pitch=candidate_notes[pointer + k].pitch, 
                        start=candidate_notes[pointer + k].start, 
                        end=candidate_notes[pointer + k].start + 0.1, 
                        velocity=vel
                    )
                    all_notes.append(note)
                    # from IPython import embed; embed(using=False); os._exit(0)


                pointer += batch_size
                # from IPython import embed; embed(using=False); os._exit(0)
                # notes_to_midi(all_notes, "_zz.mid")

            bgn_sec += segment_seconds

        #
        notes_to_midi(all_notes, "_zz.mid")
        soundfile.write(file="_zz.wav", data=audio, samplerate=16000)

        est_midi_path = Path(vel_midis_dir, "{}.mid".format(Path(audio_path).stem))
        notes_to_midi(all_notes, str(est_midi_path))
        
        # Load with pedals GT
        ref_midi_path = midi_paths[audio_idx]
        notes, _ = read_single_track_midi(ref_midi_path, extend_pedal=True)
        write_notes_to_midi(notes, "_zz_gt.mid")

        ref_intervals, ref_pitches, ref_vels = parse_midi("_zz_gt.mid")
        est_intervals, est_pitches, est_vels = parse_midi(est_midi_path)


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

        # eval with offset
        note_precision, note_recall, note_f1, _ = \
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals, 
            ref_pitches=ref_pitches, 
            est_intervals=est_intervals, 
            est_pitches=est_pitches, 
            onset_tolerance=0.05, 
            offset_ratio=0.2,

        )
        print("    P: {:.3f}, R: {:.3f}, F1: {:.3f}, time: {:.3f} s".format(note_precision, note_recall, note_f1, time.time() - t1))
        
        # eval with vel
        note_precision, note_recall, note_f1, _ = \
           mir_eval.transcription_velocity.precision_recall_f1_overlap(
               ref_intervals=ref_intervals,
               ref_pitches=ref_pitches,
               ref_velocities=ref_vels,
               est_intervals=est_intervals,
               est_pitches=est_pitches,
               est_velocities=est_vels,
               offset_ratio=None,
               # onset_tolerance=0.05, 
               # offset_ratio=0.2,
               )

        print("        P: {:.3f}, R: {:.3f}, F1: {:.3f}, time: {:.3f} s".format(note_precision, note_recall, note_f1, time.time() - t1))

        # a1 = list(np.concatenate((ref_intervals, ref_pitches[:, None]),axis=-1))
        # a1.sort(key=lambda x: x[0])  # sort by keys
        from IPython import embed; embed(using=False); os._exit(0) 
        # asdf

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
