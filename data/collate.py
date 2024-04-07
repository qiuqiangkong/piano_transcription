import numpy as np
import torch
import librosa


def collate_fn(list_data_dict):

    data_dict = {}

    for key in list_data_dict[0].keys():

        try:
            if key in ["token", "question_token", "answer_token"]:
                data_dict[key] = torch.LongTensor(np.stack([dd[key] for dd in list_data_dict], axis=0))

            elif key in ["audio", "frame_roll", "onset_roll", "offset_roll", "velocity_roll", "ped_frame_roll", "ped_onset_roll", "ped_offset_roll"]:

                data_dict[key] = torch.Tensor(np.stack([dd[key] for dd in list_data_dict], axis=0))

            else:
                data_dict[key] = [dd[key] for dd in list_data_dict]
        except:
            from IPython import embed; embed(using=False); os._exit(0)

    return data_dict