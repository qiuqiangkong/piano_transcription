from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import pretty_midi

from inference import forward, write_midi
from piano_transcription.utils import (load_maestro_meta, note_to_freq,
                                       parse_yaml)
from train import get_model


def evaluate(args) -> None:

    # Arguments and parameters
    config_yaml = args.config_yaml
    ckpt_path = args.ckpt_path
    results_dir = args.results_dir
    device = "cuda"
    eval_audios = 5  # Only evaluate partial data for speeding up 

    # Default parameters
    configs = parse_yaml(config_yaml)
    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    fps = configs["fps"]
    clip_samples = round(clip_duration * sr)

    root = "/datasets/maestro-v3.0.0"
    meta_csv = Path(root, "maestro-v3.0.0.csv")
    split = "test"

    # Load meta
    meta_dict = load_maestro_meta(root, meta_csv, split)
    audios_num = len(meta_dict["audio_name"])

    # LLM decoder
    model = get_model(
        configs=configs, 
        ckpt_path=ckpt_path
    ).to(device)

    # Prepare stat buffer
    precs = []
    recalls = []
    f1s = []
    skip_n = max(1, audios_num // eval_audios)

    # Create directory to write MIDI
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    for idx in range(0, audios_num, skip_n):

        audio_path = meta_dict["audio_path"][idx]
        ref_midi_path = meta_dict["midi_path"][idx]
        print("{}/{}, {}".format(idx, audios_num, audio_path))

        # Load audio
        audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
        
        # Foward
        events = forward(model, audio, clip_samples, sr, fps)
        
        # Write out to MIDI
        est_midi_path = str(Path(results_dir, Path(ref_midi_path).name))
        write_midi(events, est_midi_path)

        # Calculate score
        prec, recall, f1 = calculate_score(est_midi_path=est_midi_path, ref_midi_path=ref_midi_path)
        precs.append(prec)
        recalls.append(recall)
        f1s.append(f1)
        print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(prec, recall, f1))
    
    print("------ Average metric ------")
    print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(np.mean(precs), np.mean(recalls), np.mean(f1s)))
    

def calculate_score(est_midi_path: str, ref_midi_path: str):
    r"""Calculate precision, recall, F1 using mir_eval toolbox.
    """
    est_intervals, est_pitches, _ = load_midi_data_for_evaluation(est_midi_path)
    ref_intervals, ref_pitches, _ = load_midi_data_for_evaluation(ref_midi_path)

    note_precision, note_recall, note_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals=ref_intervals, 
        ref_pitches=ref_pitches, 
        est_intervals=est_intervals, 
        est_pitches=est_pitches, 
        onset_tolerance=0.05, 
        offset_ratio=None,
    )

    return note_precision, note_recall, note_f1


def load_midi_data_for_evaluation(midi_path: str) -> tuple[np.array, np.array, np.array]:
    r"""Load intervals, pitches, and velocities from MIDI.
    """
    ref_midi_data = pretty_midi.PrettyMIDI(midi_path)
    notes = ref_midi_data.instruments[0].notes
    ref_intervals = np.array([(note.start, note.end) for note in notes])  # shape: (n, 2)
    ref_pitches = np.array([note_to_freq(note.pitch) for note in notes])  # shape: (n,)
    ref_velocities = np.array([note.velocity for note in notes])  # shape: (n,)

    return ref_intervals, ref_pitches, ref_velocities


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)

    args = parser.parse_args()

    evaluate(args)