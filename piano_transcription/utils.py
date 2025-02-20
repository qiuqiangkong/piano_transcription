from __future__ import annotations

from pathlib import Path

import pandas as pd
import pretty_midi
import yaml


def parse_yaml(config_yaml: str) -> dict:
    r"""Parse yaml file."""
    
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


class LinearWarmUp:
    r"""Linear learning rate warm up scheduler.
    """
    def __init__(self, warm_up_steps: int) -> None:
        self.warm_up_steps = warm_up_steps

    def __call__(self, step: int) -> float:
        if step <= self.warm_up_steps:
            return step / self.warm_up_steps
        else:
            return 1.


def load_maestro_meta(root: str, meta_csv: str, split: str) -> dict:
    r"""Load meta dict.
    """

    df = pd.read_csv(meta_csv, sep=',')

    indexes = df["split"].values == split

    audio_names = df["audio_filename"].values[indexes]
    midi_names = df["midi_filename"].values[indexes]
    durations = df["duration"].values[indexes]

    midi_paths = [str(Path(root, name)) for name in midi_names]
    audio_paths = [str(Path(root, name)) for name in audio_names]

    meta_dict = {
        "audio_name": audio_names,
        "audio_path": audio_paths,
        "midi_name": midi_names,
        "midi_path": midi_paths,
        "duration": durations
    }

    return meta_dict


def write_midi(events: dict, midi_path: str):
    r"""Write note events to MIDI file.
    """
    track = pretty_midi.Instrument(program=0)
    track.is_drum = False
    for event in events:
        note = pretty_midi.Note(
            pitch=event["pitch"], 
            start=event["onset"], 
            end = event["offset"],
            velocity=event["velocity"]
        )
        track.notes.append(note)

    midi_data = pretty_midi.PrettyMIDI()
    midi_data.instruments.append(track)
    midi_data.write(str(midi_path))
    print("Write out to {}".format(midi_path))


def note_to_freq(piano_note: int) -> float:
    return 2 ** ((piano_note - 39) / 12) * 440