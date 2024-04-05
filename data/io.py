import time
import copy
import math
import numpy as np

import pretty_midi


class Pedal:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __repr__(self):
        return "Pedal(start={}, end={})".format(self.start, self.end)


def read_single_track_midi(midi_path, extend_pedal):

    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    
    assert len(midi_data.instruments) == 1

    notes = midi_data.instruments[0].notes
    control_changes = midi_data.instruments[0].control_changes
    
    # Get pedals.
    pedals = get_pedals(control_changes)
    
    # Extend note offsets by pedal information.
    if extend_pedal:
        notes = extend_offset_by_pedal(notes, pedals)

    return notes, pedals


def get_pedals(control_changes):

    onset = None
    offset = None
    pairs = []

    control_changes.sort(key=lambda cc: cc.time)

    for cc in control_changes:

        if cc.number == 64:

            if cc.value >= 64 and onset is None:
                onset = cc.time

            elif cc.value < 64 and onset is not None:
                offset = cc.time
                pedal = Pedal(start=onset, end=offset)
                pairs.append(pedal)
                onset = None
                offset = None

    if onset is not None and offset is None:
        offset = control_changes[-1].time
        pedal = Pedal(start=onset, end=offset)
        pairs.append(pedal)

    return pairs


def extend_offset_by_pedal(notes, pedals):

    notes = copy.deepcopy(notes)
    pedals = copy.deepcopy(pedals)
    pitches_num = 128

    notes.sort(key=lambda note: note.end)

    notes_dict = {pitch: [] for pitch in range(pitches_num)}
    
    while len(pedals) > 0 and len(notes) > 0:

        pedal = pedals[0]  # Get the first pedal.

        while notes:

            note = notes[0]  # Get the first note.
            pitch = note.pitch
            velocity = note.velocity

            if 0 <= note.end < pedal.start:
                notes_dict[pitch].append(note)
                notes.pop(0)

            elif pedal.start <= note.end < pedal.end:

                new_note = pretty_midi.Note(
                    pitch=pitch, 
                    start=note.start, 
                    end=pedal.end, 
                    velocity=velocity
                )

                notes_dict[pitch].append(new_note)
                notes.pop(0)

                # if new_note.start > 19.4 and new_note.pitch == 75:
                #     from IPython import embed; embed(using=False); os._exit(0)

            elif pedal.end <= note.end < math.inf:
                pedals.pop(0)
                break

            else:
                raise NotImplementedError 

    # 
    for note in notes:
        notes_dict[note.pitch].append(note)

    # 
    for pitch in notes_dict.keys():
        len_notes = len(notes_dict[pitch])
        if len_notes >= 2:
            for i in range(len_notes - 1):
                if notes_dict[pitch][i].end > notes_dict[pitch][i + 1].start:
                    notes_dict[pitch][i].end = notes_dict[pitch][i + 1].start

    new_notes = []
    for pitch in notes_dict.keys():
        new_notes.extend(notes_dict[pitch])
    
    new_notes.sort(key=lambda note: note.start)


    return new_notes


def write_notes_to_midi(notes, midi_path):

    midi_data = pretty_midi.PrettyMIDI()
    track = pretty_midi.Instrument(program=0)
    for note in notes:
        track.notes.append(note)
    midi_data.instruments.append(track)
    midi_data.write(midi_path)
    print("Write MIDI to {}".format(midi_path))


def time_to_grid(time, fps):
    return round(time * fps) / fps


def notes_to_rolls_and_events(notes, segment_frames, segment_start, segment_end, fps, label):

    #
    seg_start = segment_start
    seg_end = segment_end
    seg_len = seg_end - seg_start
    pitches_num = 128

    # Covert notes information to words.
    frame_roll = np.zeros((segment_frames, pitches_num))
    onset_roll = np.zeros((segment_frames, pitches_num))
    offset_roll = np.zeros((segment_frames, pitches_num))
    velocity_roll = np.zeros((segment_frames, pitches_num))

    events = []

    for note in notes:

        if 0 <= note.end < seg_start or seg_end < note.start < math.inf:
            continue

        onset_time = note.start - seg_start
        offset_time = note.end - seg_start
        pitch = note.pitch
        velocity = note.velocity

        onset_time = time_to_grid(onset_time, fps)
        offset_time = time_to_grid(offset_time, fps)

        if offset_time == onset_time:
            offset_time = onset_time + 0.01

        if onset_time < 0 and 0 <= offset_time <= seg_len:

            offset_idx = round(offset_time * fps)
            offset_roll[offset_idx, pitch] = 1
            frame_roll[0 : offset_idx + 1, pitch] = 1

            events.append({
                "name": "note_sustain", 
                "time": 0, 
                "label": label,
                "pitch": pitch, 
                "velocity": velocity
            })
            events.append({
                "name": "note_off",
                "time": offset_time, 
                "label": label,
                "pitch": pitch,
            })

        elif onset_time < 0 and seg_len < offset_time < math.inf:

            frame_roll[:, pitch] = 1

            events.append({
                "name": "note_sustain", 
                "time": 0, 
                "label": label,
                "pitch": pitch, 
                "velocity": velocity
            })

        elif 0 <= onset_time <= seg_len and 0 <= offset_time <= seg_len:

            onset_idx = round(onset_time * fps)
            offset_idx = round(offset_time * fps)
            onset_roll[onset_idx, pitch] = 1
            velocity_roll[onset_idx, pitch] = velocity / 128.0
            offset_roll[offset_idx, pitch] = 1
            frame_roll[onset_idx : offset_idx + 1, pitch] = 1

            events.append({
                "name": "note_on",
                "time": onset_time, 
                "label": label,
                "pitch": pitch, 
                "velocity": velocity
            })
            events.append({
                "name": "note_off",
                "time": offset_time, 
                "label": label,
                "pitch": pitch, 
            })

        elif 0 <= onset_time <= seg_len and seg_len < offset_time < math.inf:

            onset_idx = round(onset_time * fps)
            onset_roll[onset_idx, pitch] = 1
            velocity_roll[onset_idx, pitch] = velocity / 128.0
            frame_roll[onset_idx : , pitch] = 1

            events.append({
                "name": "note_on",
                "time": onset_time, 
                "label": label,
                "pitch": pitch, 
                "velocity": velocity
            })

    # if label == "slakh2100-Drums":
    #     from IPython import embed; embed(using=False); os._exit(0)

    events.sort(key=lambda event: (event["time"], event["name"], event["label"], event["pitch"]))
    
    data = {
        "frame_roll": frame_roll,
        "onset_roll": onset_roll,
        "offset_roll": offset_roll,
        "velocity_roll": velocity_roll,
        "events": events,
    }

    return data


def pedals_to_rolls_and_events(pedals, segment_frames, segment_start, segment_end, fps, label):

    #
    seg_start = segment_start
    seg_end = segment_end
    seg_len = seg_end - seg_start

    # Covert pedals information to words.
    frame_roll = np.zeros(segment_frames)
    onset_roll = np.zeros(segment_frames)
    offset_roll = np.zeros(segment_frames)

    events = []

    for pedal in pedals:

        if 0 <= pedal.end < seg_start or seg_end < pedal.start < math.inf:
            continue

        onset_time = pedal.start - seg_start
        offset_time = pedal.end - seg_start

        onset_time = time_to_grid(onset_time, fps)
        offset_time = time_to_grid(offset_time, fps)

        if offset_time == onset_time:
            offset_time = onset_time + 0.01

        if onset_time < 0 and 0 <= offset_time <= seg_len:

            offset_idx = round(offset_time * fps)
            offset_roll[offset_idx] = 1
            frame_roll[0 : offset_idx + 1] = 1

            events.append({
                "name": "pedal_sustain", 
                "time": 0, 
                "label": label,
            })
            events.append({
                "name": "pedal_off",
                "time": offset_time, 
                "label": label,
            })

        elif onset_time < 0 and seg_len < offset_time < math.inf:

            frame_roll[:] = 1

            events.append({
                "name": "pedal_sustain", 
                "time": 0, 
                "label": label,
            })

        elif 0 <= onset_time <= seg_len and 0 <= offset_time <= seg_len:

            onset_idx = round(onset_time * fps)
            offset_idx = round(offset_time * fps)
            onset_roll[onset_idx] = 1
            offset_roll[offset_idx] = 1
            frame_roll[onset_idx : offset_idx + 1] = 1

            events.append({
                "name": "pedal_on",
                "time": onset_time, 
                "label": label,
            })
            events.append({
                "name": "pedal_off",
                "time": offset_time,
                "label": label,
            })

        elif 0 <= onset_time <= seg_len and seg_len < offset_time < math.inf:

            onset_idx = round(onset_time * fps)
            onset_roll[onset_idx] = 1
            frame_roll[onset_idx :] = 1

            events.append({
                "name": "pedal_on",
                "time": onset_time, 
                "label": label,
            })

    events.sort(key=lambda event: (event["time"], event["name"], event["label"]))

    data = {
        "frame_roll": frame_roll,
        "onset_roll": onset_roll,
        "offset_roll": offset_roll,
        "events": events,
    }

    return data


# def beats_to_rolls_and_events(
#     beats,
#     segment_start, 
#     segment_end, 
#     fps
# ):
#     seg_start = segment_start
#     seg_end = segment_end
#     seg_frames = round((seg_end - seg_start) * fps) + 1
    
#     beat_roll = np.zeros(seg_frames)
#     downbeat_roll = np.zeros(seg_frames)
#     events = []

#     for beat in beats:

#         beat_time = beat.start - seg_start
#         beat_time = time_to_grid(beat_time, fps)

#         if seg_start <= beat.start <= seg_end:

#             frame_idx = round(beat_time * fps)
#             beat_roll[frame_idx] += 1

#             events.append({
#                 "name": "beat",
#                 "time": beat_time,
#                 "index": beat.index
#             })

#             if beat.index == 0:
#                 downbeat_roll[frame_idx] += 1

#                 events.append({
#                     "name": "downbeat",
#                     "time": beat_time,
#                 })

#     # No need to sort events because they are already sorted.

#     data = {
#         "beat_roll": beat_roll,
#         "downbeat_roll": downbeat_roll,
#         "events": events,
#     }

#     return data


def read_beats(midi_path):

    midi_data = pretty_midi.PrettyMIDI(str(midi_path))

    beats = midi_data.get_beats()
    downbeats = midi_data.get_downbeats()

    return beats, downbeats


def beats_to_targets(beats, downbeats, segment_frames, segment_start, segment_end, fps):

    #
    seg_start = segment_start
    seg_end = segment_end
    
    # Covert pedals information to words.
    beat_roll = np.zeros(segment_frames)
    downbeat_roll = np.zeros(segment_frames)

    events = []
    from IPython import embed; embed(using=False); os._exit(0)
    for beat in beats:

        if seg_start <= beat <= seg_end:

            beat_time = beat - seg_start
            beat_idx = round(beat_time * fps)
            beat_roll[beat_idx] += 1

            events.append({
                "name": "beat",
                "time": beat_time, 
            })

    for beat in downbeats:

        if seg_start <= beat <= seg_end:

            beat_time = beat - seg_start
            beat_idx = round(beat_time * fps)
            downbeat_roll[beat_idx] += 1

            events.append({
                "name": "downbeat",
                "time": beat_time, 
            })

    events.sort(key=lambda event: event["time"])

    data = {
        "beat_roll": beat_roll,
        "downbeat_roll": downbeat_roll,
        "events": events,
    }

    return data


def events_to_notes(events):

    pitches_num = 128

    note_on_buffer = {pitch: [] for pitch in range(pitches_num)}
    notes = []

    for e in events:
        
        if e["name"] == "note_on":
            pitch = e["pitch"]
            note_on_buffer[pitch].append(e)

        elif e["name"] == "note_off":
            pitch = e["pitch"]
            if len(note_on_buffer[pitch]) > 0:
                onset_event = note_on_buffer[pitch].pop(0)
                note = pretty_midi.Note(
                    pitch=pitch, 
                    start=onset_event["time"], 
                    end=e["time"], 
                    velocity=onset_event["velocity"],
                )
                notes.append(note)
    
    notes.sort(key=lambda note: (note.start, note.pitch, note.end, note.velocity))

    return notes


def fix_length(x, max_len, constant_value):
    if len(x) >= max_len:
        return x[0 : max_len]
    else:
        return x + [constant_value] * (max_len - len(x))
    

def notes_to_midi(notes, midi_path):

    track = pretty_midi.Instrument(program=0)
    track.is_drum = False

    for note in notes:
        track.notes.append(note)

    midi_data = pretty_midi.PrettyMIDI()
    midi_data.instruments.append(track)
    midi_data.write(midi_path)
    print("Write out to {}".format(midi_path))


def mt_notes_to_midi(mt_notes, inst_map, midi_path):

    midi_data = pretty_midi.PrettyMIDI()
    
    for key, notes in mt_notes.items():

        if inst_map[key] == "Drums":
            program = 10
            track = pretty_midi.Instrument(program=program)
            track.is_drum = True
        else:
            program = pretty_midi.instrument_name_to_program(inst_map[key])
            track = pretty_midi.Instrument(program=program)
            track.is_drum = False
        
        for note in notes:
            track.notes.append(note)

        midi_data.instruments.append(track)
    
    midi_data.write(midi_path)
    print("Write out to {}".format(midi_path))
    # from IPython import embed; embed(using=False); os._exit(0)


def test():

    midi_path = "/home/qiuqiangkong/datasets/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi"

    notes, pedals = read_single_track_midi(midi_path)
    
    write_notes_to_midi(notes, "_zz.mid")

    segment_frames = 1001
    seg_start = 25.
    seg_end = 35.
    fps = 100

    t1 = time.time()
    note_data = notes_to_targets(
        notes=notes,
        segment_frames=segment_frames, 
        segment_start=seg_start,
        segment_end=seg_end,
        fps=fps
    )

    pedal_data = pedals_to_targets(
        pedals=pedals,
        segment_frames=segment_frames, 
        segment_start=seg_start,
        segment_end=seg_end,
        fps=fps
    )

    frame_roll = note_data["frame_roll"]
    onset_roll = note_data["onset_roll"]
    offset_roll = note_data["offset_roll"]
    velocity_roll = note_data["velocity_roll"]

    ped_frame_roll = pedal_data["frame_roll"]
    ped_onset_roll = pedal_data["onset_roll"]
    ped_offset_roll = pedal_data["offset_roll"]

    all_events = note_data["events"] + pedal_data["events"]
    all_events.sort(key=lambda event: event["time"])

    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':

    test()