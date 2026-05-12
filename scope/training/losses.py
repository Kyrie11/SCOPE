import torch
import torch.nn.functional as F

def masked_mean(loss: torch.Tensor, mask: torch.Tensor, eps: float=1e-6):
    return (loss * mask.float()).sum()/(mask.float().sum()+eps)

def ordinal_targets(y):
    return torch.stack([(y>=1),(y>=2),(y>=3)], dim=-1).float()

def pressure_distribution_from_exceedance(p_ge):
    p0=1-p_ge[...,0]; p1=p_ge[...,0]-p_ge[...,1]; p2=p_ge[...,1]-p_ge[...,2]; p3=p_ge[...,2]
    return torch.stack([p0,p1,p2,p3],-1).clamp_min(0)

def js_divergence(p,q,eps=1e-6):
    p=p.clamp_min(eps); q=q.clamp_min(eps); m=.5*(p+q)
    return .5*(p*(p/m).log()).sum(-1)+.5*(q*(q/m).log()).sum(-1)

def label_losses(outputs, batch, prefix=''):
    labels=batch['labels']; masks=batch['masks']; losses={}; counts={}
    ce=F.cross_entropy(outputs['outcome_logits'].transpose(1,2), labels['outcome'], reduction='none')
    losses['outcome_nll']=masked_mean(ce, masks['outcome'] & masks.get('query', torch.ones_like(masks['outcome']))); counts['outcome']=masks['outcome'].sum().item()
    targ=ordinal_targets(labels['pressure_ordinal']); bce=F.binary_cross_entropy_with_logits(outputs['pressure_exceedance_logits'], targ, reduction='none').mean(-1)
    losses['pressure_bce']=masked_mean(bce, masks['pressure'] & masks.get('query', torch.ones_like(masks['pressure']))); counts['pressure']=masks['pressure'].sum().item()
    traj=((outputs['traj_params']['mean']-labels['trajectory'])**2).mean(-1).mean(-1)
    losses['traj_mse']=masked_mean(traj, masks['trajectory'] & masks.get('query', torch.ones_like(masks['trajectory']))); counts['trajectory']=masks['trajectory'].sum().item()
    safety=0
    for name,out_key in [('collision','collision_prob'),('near_collision','near_collision_prob'),('hard_brake','hard_brake_prob'),('high_pressure','high_pressure_prob')]:
        b=F.binary_cross_entropy(outputs[out_key].clamp(1e-4,1-1e-4), labels[name].float(), reduction='none')
        safety=safety+masked_mean(b,masks['safety'])
    losses['safety_bce']=safety/4
    fd=F.binary_cross_entropy(outputs['forced_dependence_prob'].clamp(1e-4,1-1e-4), labels['forced_dependence'].float(), reduction='none')
    losses['forced_dependence_bce']=masked_mean(fd,masks['forced_dependence']); counts['forced_dependence']=masks['forced_dependence'].sum().item()
    return losses, counts

def boundary_loss(outputs,batch,lambda_pressure=.25,lambda_risk=.25,margin=.5):
    pairs=batch['neighbor_pairs']; m=batch['masks']['boundary']
    if pairs.numel()==0: return outputs['outcome_logits'].sum()*0
    B,E,_=pairs.shape; idxa=pairs[...,0]; idxb=pairs[...,1]
    def gather(x):
        shape=x.shape[2:]; ia=idxa.reshape(B,E,*([1]*len(shape))).expand(B,E,*shape); ib=idxb.reshape(B,E,*([1]*len(shape))).expand(B,E,*shape); return torch.gather(x,1,ia), torch.gather(x,1,ib)
    pa,pb=gather(outputs['outcome_prob']); pra,prb=gather(outputs['pressure_exceedance_prob']); ca,cb=gather(outputs['collision_prob'].unsqueeze(-1));
    d=js_divergence(pa,pb)+lambda_pressure*(pra-prb).abs().mean(-1)+lambda_risk*(ca.squeeze(-1)-cb.squeeze(-1)).abs()
    y=batch['labels']['boundary'].float(); loss=torch.where(y>0.5, torch.relu(margin-d), d)
    return masked_mean(loss,m)

def total_loss(outputs,batch,weights):
    losses,counts=label_losses(outputs,batch); losses['boundary']=boundary_loss(outputs,batch)
    total=0
    mapping={'w_sim_outcome':'outcome_nll','w_sim_pressure':'pressure_bce','w_sim_traj':'traj_mse','w_sim_safety':'safety_bce','w_fd':'forced_dependence_bce','w_boundary':'boundary'}
    for wk,lk in mapping.items(): total=total+float(weights.get(wk,1.0))*losses[lk]
    losses['total']=total; return losses, counts
