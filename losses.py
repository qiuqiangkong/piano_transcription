import torch
import torch.nn as nn
import torch.nn.functional as F


def bce(output, target, mask):
    """Binary crossentropy (BCE) with mask. The positions where mask=0 will be 
    deactivated when calculation BCE.
    """
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    return torch.sum(matrix * mask) / torch.sum(mask)


def regress_onset_offset_frame_velocity_bce(output_dict, target_dict):
    """High-resolution piano note regression loss, including onset regression, 
    offset regression, velocity regression and frame-wise classification losses.
    """
    try:
        onset_loss = F.binary_cross_entropy(output_dict["reg_onset_output"], target_dict["onset_roll"])
        offset_loss = F.binary_cross_entropy(output_dict["reg_offset_output"], target_dict["offset_roll"])
        frame_loss = F.binary_cross_entropy(output_dict["frame_output"], target_dict["frame_roll"])
        velocity_loss = F.binary_cross_entropy(output_dict["velocity_output"], target_dict["velocity_roll"])

        total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    except:
        from IPython import embed; embed(using=False); os._exit(0)

    return total_loss


def regress_onset_offset_frame_velocity_bce2(output_dict, target_dict):
    """High-resolution piano note regression loss, including onset regression, 
    offset regression, velocity regression and frame-wise classification losses.
    """
    try:
        onset_loss = F.binary_cross_entropy(output_dict["onset_roll"], target_dict["onset_roll"])
        offset_loss = F.binary_cross_entropy(output_dict["offset_roll"], target_dict["offset_roll"])
        frame_loss = F.binary_cross_entropy(output_dict["frame_roll"], target_dict["frame_roll"])
        velocity_loss = F.binary_cross_entropy(output_dict["velocity_roll"], target_dict["velocity_roll"])

        total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    except:
        from IPython import embed; embed(using=False); os._exit(0)

    return total_loss