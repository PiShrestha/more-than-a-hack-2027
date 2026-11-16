"""Fixed other-match stress benchmark; reuse fold models, save cheap-to-reanalyse probabilities."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import lightgbm as lgb
from possession import model,idsse,tcn
from research.corruptions import CASES,SEEDS,mask,gap_summary
from research.aggregate import baseline

@torch.inference_mode()
def neural(net,stats,X,device):
    Z=torch.from_numpy(tcn._prep(X,stats)).T[None].to(device)
    return torch.softmax(net(Z)[0],0).T.cpu().numpy()

def metrics(gt,pred,p=None):
    r=model.pm.evaluate(gt.astype(np.int8),pred.astype(np.int8))
    out={k:r[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells','n_gt_spells','accuracy')}
    out['dead_ball']=r['per_class']['D']
    if p is not None:
        ok=gt<3
        yy=gt[ok]
        pp=p[ok]
        out['brier_multiclass']=float(np.mean(np.sum((pp-np.eye(3)[yy])**2,axis=1)))
        confidence=pp.max(1)
        correct=pp.argmax(1)==yy
        ece=0.
        for i in range(15):
            sel=(confidence>=i/15)&((confidence<(i+1)/15) if i<14 else (confidence<=1))
            if sel.any():
                ece+=sel.mean()*abs(float(correct[sel].mean())-float(confidence[sel].mean()))
        out['ece15']=float(ece)
    return out

def main():
    task=int(os.environ['SLURM_ARRAY_TASK_ID'])
    match=idsse.MATCHES[task%len(idsse.MATCHES)]
    seed=SEEDS[task//len(idsse.MATCHES)]
    fi=(0,0,1,1,2,2,2)[task%len(idsse.MATCHES)]
    work=Path(os.environ['RESEARCH_RESULTS'])
    out=work/'stress'/f'{match}-seed{seed}'
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    assert torch.cuda.is_available()
    torch.backends.cudnn.allow_tf32=False
    params=json.loads((ROOT/'possession/decoder.json').read_text())
    booster=lgb.Booster(model_file=str(work/'folds/lgb'/f'fold-{fi}.txt'))
    fold_meta=json.loads((work/'folds/lgb'/f'fold-{fi}.json').read_text())
    assert match in fold_meta['validation'] and match not in fold_meta['train']
    nets=[]
    for tag in params['tcn_tags']:
        ck=torch.load(work/'folds'/('tcn'+(tag or '_base'))/f'fold-{fi}.pt',weights_only=False,map_location='cpu')
        assert match in ck['validation'] and match not in ck['train']
        net=tcn.TCN(len(ck['mu']),ck['state']['inp.weight'].shape[0])
        net.load_state_dict(ck['state'])
        nets.append((tag,net.to('cuda').eval(),(ck['mu'],ck['sd'])))
    template=model.bar_ath_ball_mask()
    # Generate player dropout and slow dead-ball carries ONCE, paired across all masks.
    # Labels are used only by the pre-existing other-match augmentation and scoring.
    original=idsse.load(match)
    base=model.add_noise(original,np.zeros_like(template),np.random.default_rng(seed+task%7))
    gt=np.where(base['y']<0,3,base['y'])
    rows=[]
    cpu_gpu_check=None
    for case in CASES:
        started=time.time()
        dropped=mask(case,len(gt),template,seed+task%7)
        d=dict(base,bx=np.where(dropped,np.nan,base['bx']),by=np.where(dropped,np.nan,base['by']))
        X,_=tcn.features(d)
        Xf,_=tcn.features(d,flip=True)
        lp=.5*(booster.predict(X,num_threads=int(os.environ['SLURM_CPUS_PER_TASK']))+
               booster.predict(Xf,num_threads=int(os.environ['SLURM_CPUS_PER_TASK']))[:,[1,0,2]])
        saved={'lightgbm':lp.astype(np.float32)}
        ps=[]
        for tag,net,stats in nets:
            p=.5*(neural(net,stats,X,'cuda')+neural(net,stats,Xf,'cuda')[:,[1,0,2]])
            ps.append(p)
            saved['tcn'+(tag or '_base')]=p
            if cpu_gpu_check is None and not tag:
                # One full-match check per task, with the same global normalization scope.
                net.cpu()
                cp=.5*(neural(net,stats,X,'cpu')+neural(net,stats,Xf,'cpu')[:,[1,0,2]])
                net.cuda()
                cpu_gpu_check=dict(max_probability_error=float(np.max(abs(cp-p))),
                                   argmax_agreement=float((cp.argmax(1)==p.argmax(1)).mean()))
                assert cpu_gpu_check['max_probability_error']<.002,cpu_gpu_check
        ens=(1-params['tcn_weight'])*lp+params['tcn_weight']*np.mean(ps,axis=0)
        pred=model.decode(ens,d['period'],params)
        bp=baseline(d)
        saved.update(ensemble=ens.astype(np.float32),y=gt.astype(np.int8),period=d['period'],bvis=~dropped)
        dest=out/f'{case}.npz'
        tmp=dest.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,**saved)
        tmp.replace(dest)
        record=dict(case=case,match=match,seed=seed,fold=fi,corruption=gap_summary(dropped),
                    ensemble=metrics(gt,pred,ens),smooth50=metrics(gt,bp),seconds=time.time()-started)
        rows.append(record)
        print(json.dumps(record),flush=True)
    report=dict(source=(ROOT/'research/SOURCE_COMMIT').read_text().strip(),match=match,seed=seed,
                decoder=params,cpu_gpu_check=cpu_gpu_check,gpu=torch.cuda.get_device_name(),rows=rows,
                note='Additional corruption seeds, not new independent matches. Frozen model/decoder; prior CV match selection still applies.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    main()
