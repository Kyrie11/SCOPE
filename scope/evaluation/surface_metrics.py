import torch
from scope.training.metrics import accuracy, brier

def compute_surface_metrics(outputs,batch):
    m={}
    mask=batch['masks']['outcome'];
    if mask.any(): m['outcome_accuracy']=float(accuracy(outputs['outcome_logits'], batch['labels']['outcome'], mask))
    pm=batch['masks']['pressure'];
    if pm.any(): m['pressure_mae']=float(((outputs['pressure_mean']-batch['labels']['pressure_ordinal'].float()).abs()*pm.float()).sum()/pm.float().sum())
    sm=batch['masks']['safety'];
    if sm.any(): m['collision_brier']=float(brier(outputs['collision_prob'], batch['labels']['collision'], sm))
    return m
