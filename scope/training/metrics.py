import torch

def accuracy(logits, labels, mask):
    pred=logits.argmax(-1); return ((pred==labels)&mask.bool()).float().sum()/mask.float().sum().clamp_min(1)

def brier(prob, target, mask):
    return (((prob-target.float())**2)*mask.float()).sum()/mask.float().sum().clamp_min(1)
