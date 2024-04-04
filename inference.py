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

# from evalaute import parse_midi


def inference(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    segment_seconds = 4.
    device = "cuda"
    sample_rate = 16000

    # Load checkpoint
    # checkpoint_path = Path("checkpoints", model_name, "latest.pth")
    # checkpoint_path = Path("checkpoints", model_name, "epoch=100.pth")
    # checkpoint_path = Path("checkpoints/train/CRnn3/step=90000.pth")
    # checkpoint_path = Path("checkpoints/train/CRnn3/step=60000.pth")
    # checkpoint_path = Path("checkpoints/train/CRnn2_onset_offset_vel/step=100000.pth")
    # checkpoint_path = Path("checkpoints/train/CRnn3_onset_offset_vel/step=100000.pth")
    # checkpoint_path = Path("/public/qq_to_dylan/CRnn3_onset_offset_vel/step=100000.pth")
    checkpoint_path = Path("checkpoints/train/CRnn3_onset_offset_vel/step=30000.pth")

    model = get_model(model_name)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # root = "/datasets/maestro-v2.0.0/maestro-v2.0.0"
    root = "/datasets/maestro-v3.0.0/maestro-v3.0.0"
    meta_csv = Path(root, "maestro-v3.0.0.csv")
    meta_data = load_meta(meta_csv, split="test")
    # meta_data = load_meta(meta_csv, split="train")
    audio_paths = [Path(root, name) for name in meta_data["audio_filename"]]
    midi_paths = [Path(root, name) for name in meta_data["midi_filename"]]

    # audio_paths = ["/home/qiuqiangkong/datasets/maestro-v2.0.0/2009/MIDI-Unprocessed_11_R1_2009_01-05_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_04_WAV.wav"]
    # midi_paths = ["/home/qiuqiangkong/datasets/maestro-v2.0.0/2009/MIDI-Unprocessed_11_R1_2009_01-05_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_04_WAV.midi"]

    # Load audio. Change this path to your favorite song.
    # audio_paths = ["/home/qiuqiangkong/datasets/maestro-v2.0.0/2014/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--2.wav"]

    output_dir = Path("pred_midis", model_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    precs = []
    recalls = []
    f1s = []

    # for audio_idx, audio_path in enumerate(audio_paths):
    for audio_idx in range(len(audio_paths)):

        # audio_idx = 3
        # print(audio_idx)
        audio_path = audio_paths[audio_idx]


        audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)
        # (channels_num, audio_samples)

        audio_samples = audio.shape[-1]
        bgn = 0
        segment_samples = int(segment_seconds * sample_rate)

        onset_rolls = []

        # Do separation
        while bgn < audio_samples:
            
            # print("Processing: {:.1f} s".format(bgn / sample_rate))

            # Cut segments
            segment = audio[bgn : bgn + segment_samples]
            segment = librosa.util.fix_length(data=segment, size=segment_samples, axis=-1)
            segment = torch.Tensor(segment).to(device)

            # Separate a segment
            with torch.no_grad():
                model.eval()
                output_dict = model(audio=segment[None, :])

                if model_name in ["CRnn2_onset_offset_vel"]:
                    onset_roll = output_dict["reg_onset_output"].cpu().numpy()[0]

                elif model_name in ["CRnn3_onset_offset_vel"]:
                    onset_roll = output_dict["onset_roll"].cpu().numpy()[0]
                    emb = output_dict["emb"].cpu().numpy()[0]

                else:
                    onset_roll = output_dict["onset_roll"].cpu().numpy()[0]
                
                onset_rolls.append(onset_roll[0:-1, :])
                # sep_wavs.append(sep_wav.cpu().numpy())

            bgn += segment_samples

            '''
            soundfile.write(file="_zz.wav", data=segment.cpu().numpy(), samplerate=sample_rate)

            fig, axs = plt.subplots(2, 2, sharex=True)
            axs[0, 0].matshow(output_dict["onset_roll"].cpu().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
            axs[0, 1].matshow(output_dict["offset_roll"].cpu().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
            axs[1, 0].matshow(output_dict["frame_roll"].cpu().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
            axs[1, 1].matshow(output_dict["velocity_roll"].cpu().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
            plt.savefig("_zz.pdf")
            from IPython import embed; embed(using=False); os._exit(0)
            '''

        onset_rolls = np.concatenate(onset_rolls, axis=0)
        # pickle.dump(onset_rolls, open("_zz.pkl", "wb"))

        est_midi_path = Path(output_dir, "{}.mid".format(Path(audio_path).stem))
        post_process(onset_rolls, est_midi_path)
        
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

        # if audio_idx == 10:
        #     break

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


def post_process(onset_roll, output_path):
    # onset_roll = pickle.load(open("_zz.pkl", "rb"))

    # import torch
    # a1 = torch.Tensor(onset_roll[None, None, :, :])
    # w = torch.Tensor(np.zeros((1, 1, 5, 5)))
    # y = torch.nn.functional.conv2d(input=a1, weight=w, padding=2)
    # y = y.cpu().numpy()[0, 0]
    # plt.matshow(onset_roll[0:1000].T, origin='lower', aspect='auto', cmap='jet')
    # plt.savefig("_zz.pdf")
    # from IPython import embed; embed(using=False); os._exit(0)
    
    frames_num, pitches_num = onset_roll.shape

    notes = []

    array = np.stack(np.where(onset_roll > 0.3), axis=-1)

    array = deduplicate_array(array)

    onsets = array[:, 0] / 100
    pitches = array[:, 1]

    for onset, pitch in zip(onsets, pitches):
        offset = onset + 0.2
        note = {
            "onset": onset,
            "offset": offset,
            "pitch": pitch
        }
        notes.append(note)

    # Write to MIDI
    new_midi_data = pretty_midi.PrettyMIDI()
    new_track = pretty_midi.Instrument(program=0)
    for note in notes:
        midi_note = pretty_midi.Note(pitch=note["pitch"], start=note["onset"], end=note["offset"], velocity=100)
        new_track.notes.append(midi_note)
    new_midi_data.instruments.append(new_track)
    new_midi_data.write(str(output_path))
    print("Write out to {}".format(output_path))


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
    parser.add_argument('--model_name', type=str, default="CRnn3")
    args = parser.parse_args()

    inference(args) 
