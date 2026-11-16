"""Controlled CNN context probe using fixed external-match features and held-out models.

Optional context uses automatic BAR-ATH feature rows at ungraded timestamps.
Frozen preprocessing windows can reach observations across half boundaries.
No BAR-ATH labels are loaded. Context feature vectors are not exported.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from possession import model,idsse,tcn
from research.cache_tools import separate_periods,insert_context

CASES=('compact','halves_independent','neutral_ungraded','observed_ungraded')

@torch.inference_mode()
def neural(net,z):
    x=torch.from_numpy(z.T[None]).to('cuda')
    return torch.softmax(net(x)[0],0).T.cpu().numpy()

def main():
    start=time.time()
    task=int(os.environ['SLURM_ARRAY_TASK_ID'])
    match=idsse.MATCHES[task]
    fi=(0,0,1,1,2,2,2)[task]
    work=Path(os.environ['RESEARCH_RESULTS'])
    out=work/'context-probe'/match
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    params=json.loads((ROOT/'possession/decoder.json').read_text())
    # Reconstruct exactly the original held-out corruption sequence, including RNG order.
    val=((0,1),(2,3),(4,5,6))[fi]
    rng=np.random.default_rng(model.SEED+1)
    template=model.bar_ath_ball_mask()
    for i in val:
        d=model.add_noise(idsse.load(idsse.MATCHES[i]),template,rng)
        if i==task:
            break
    base=np.load(Path(idsse.SRC)/f'cv_{match}.npz')
    assert np.array_equal(base['y'],d['y']) and np.array_equal(base['bvis'],~np.isnan(d['bx']))
    X,_=tcn.features(d)
    Xf,_=tcn.features(d,flip=True)
    bar=model.bar_ath()
    BX,_=tcn.features(bar)
    BXf,_=tcn.features(bar,flip=True)
    h1=np.flatnonzero(bar['period']==1)
    h2=np.flatnonzero(bar['period']==2)
    ranges=[np.arange(0,h1[0]),np.arange(h1[-1]+1,h2[0]),np.arange(h2[-1]+1,len(bar['period']))]
    assert all((bar['period'][r]==0).all() for r in ranges)
    context_lengths=[len(r) for r in ranges]
    interior=np.zeros(len(d['y']),dtype=bool)
    for half in (1,2):
        idx=np.flatnonzero(d['period']==half)
        interior[idx[1500:-1500]]=True  # >60s from boundaries, beyond the nominal convolution radius
    assert interior.any()
    member_prob={case:[] for case in CASES}
    checks=[]
    for tag in params['tcn_tags']:
        ck=torch.load(work/'folds'/('tcn'+(tag or '_base'))/f'fold-{fi}.pt',weights_only=False,map_location='cpu')
        assert match in ck['validation'] and match not in ck['train']
        net=tcn.TCN(len(ck['mu']),ck['state']['inp.weight'].shape[0])
        net.load_state_dict(ck['state'])
        net.to('cuda').eval()
        stats=(ck['mu'],ck['sd'])
        oriented={case:[] for case in CASES}
        for features,context_features in ((X,BX),(Xf,BXf)):
            z=tcn._prep(features,stats)
            oriented['compact'].append(neural(net,z))
            p=np.empty((len(z),3),dtype=np.float32)
            for half in (1,2):
                idx=np.flatnonzero(d['period']==half)
                p[idx]=neural(net,z[idx])
            oriented['halves_independent'].append(p)
            neutral=[np.zeros((n,z.shape[1]),dtype=np.float32) for n in context_lengths]
            observed=[tcn._prep(context_features[r],stats) for r in ranges]
            for case,context in (('neutral_ungraded',neutral),('observed_ungraded',observed)):
                expanded,index=insert_context(z,d['period'],context)
                oriented[case].append(neural(net,expanded)[index])
        for case in CASES:
            p=.5*(oriented[case][0]+oriented[case][1][:,[1,0,2]])
            assert np.isfinite(p).all()
            member_prob[case].append(p)
        ref=np.load(Path(idsse.SRC)/f'cvtcn{tag}_{match}.npz')['p']
        difference=float(np.max(abs(ref-member_prob['compact'][-1])))
        assert difference<.002,(tag,difference)
        checks.append(dict(member=tag or 'base',cpu_gpu_max_probability_difference=difference))
        print('completed member',tag or 'base','seconds',time.time()-start,flush=True)
    gt=np.where(d['y']<0,3,d['y']).astype(np.int8)
    ensembles={case:(1-params['tcn_weight'])*base['p']+params['tcn_weight']*np.mean(member_prob[case],axis=0) for case in CASES}
    rows=[]
    ref=ensembles['compact']
    for case in CASES:
        p=ensembles[case]
        pred=model.decode(p,d['period'],params)
        a,b=separate_periods(gt,pred,d['period'])
        score=model.pm.evaluate(a,b)
        row=dict(case=case,match=match,spell_f1=score['spell_f1_mean'],macro_f1=score['macro_f1'],
                 dead_f1=score['per_class']['D']['f1'],
                 interior_argmax_agreement_with_compact=float((p.argmax(1)[interior]==ref.argmax(1)[interior]).mean()),
                 interior_mean_abs_probability_change=float(abs(p-ref)[interior].mean()),
                 interior_max_abs_probability_change=float(abs(p-ref)[interior].max()))
        rows.append(row)
        saved={('tcn'+(tag or '_base')):member_prob[case][i] for i,tag in enumerate(params['tcn_tags'])}
        saved.update(ensemble=p,decoded=pred,y=gt,period=d['period'],interior=interior)
        dest=out/f'{case}.npz'
        tmp=dest.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,**saved)
        tmp.replace(dest)
    report=dict(match=match,fold=fi,source=(ROOT/'research/SOURCE_COMMIT').read_text().strip(),
                decoder=params,context_lengths=context_lengths,context_feature_vectors_exported=False,
                play_features_identical_across_conditions=True,interior_margin_seconds=60,
                cpu_gpu_checks=checks,rows=rows,seconds=time.time()-start,
                caveat='Controlled context sensitivity, not accuracy on BAR-ATH. Same seven previously selected validation matches; no new model selection.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    main()
