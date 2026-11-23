"""Opt-in tensor snapshots for training-side Kimi DSA boundaries."""
import torch

def freeze(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {str(k):freeze(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [freeze(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return {'type':type(value).__name__}


def training_stage(owner,**values):
    if getattr(owner,'_alignment_capture_dsa',False):
        owner._alignment_training_dsa_trace.update({k:freeze(v) for k,v in values.items()})
