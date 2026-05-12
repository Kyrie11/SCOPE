from pathlib import Path
import torch
torch.set_num_threads(1)
from torch.utils.data import DataLoader
from scope.data.dataset import SCOPEDataset
from scope.data.collate import collate_scope
from scope.models.baselines import build_model
from scope.training.losses import total_loss
from scope.training.checkpoints import save_checkpoint

class Trainer:
    def __init__(self, train_cfg, model_cfg, train_root, val_root, output_dir, device=None, max_train_groups=None, max_val_groups=None):
        self.train_cfg=train_cfg; self.model_cfg=model_cfg; self.output_dir=Path(output_dir); self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=build_model(model_cfg).to(self.device); self.train_ds=SCOPEDataset(train_root, max_groups=max_train_groups); self.val_ds=SCOPEDataset(val_root, max_groups=max_val_groups)
        self.opt=torch.optim.AdamW(self.model.parameters(), lr=train_cfg.get('training',{}).get('lr',2e-4), weight_decay=train_cfg.get('training',{}).get('weight_decay',.01), foreach=False)
    def _move(self,b):
        for k,v in list(b.items()):
            if torch.is_tensor(v): b[k]=v.to(self.device)
            elif isinstance(v,dict):
                for kk,vv in v.items():
                    if torch.is_tensor(vv): v[kk]=vv.to(self.device)
        return b
    def train(self, overfit_debug=False):
        bs=self.train_cfg.get('training',{}).get('batch_size',4); epochs=1 if overfit_debug else self.train_cfg.get('training',{}).get('epochs',50)
        loader=DataLoader(self.train_ds,batch_size=min(bs,max(1,len(self.train_ds))),shuffle=True,collate_fn=collate_scope)
        weights=self.train_cfg.get('loss_weights',{})
        best=1e9
        for ep in range(epochs):
            self.model.train(); vals=[]
            for batch in loader:
                batch=self._move(batch); out=self.model(batch,'support_adapted'); losses,_=total_loss(out,batch,weights); loss=losses['total']; self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.get('training',{}).get('grad_clip_norm',5.0)); self.opt.step(); vals.append(float(loss.detach()))
            avg=sum(vals)/max(1,len(vals)); print(f'epoch={ep} train_loss={avg:.4f}')
            if avg<best: best=avg; save_checkpoint(self.output_dir/'best.ckpt', self.model, self.opt, {'epoch':ep,'loss':best,'model_config':self.model_cfg})
        save_checkpoint(self.output_dir/'last.ckpt', self.model, self.opt, {'loss':best,'model_config':self.model_cfg})
