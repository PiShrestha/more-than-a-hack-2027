"""Compare one fixed per-frame-normalized CNN against the same-capacity stock base CNN."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from possession import model,tcn,idsse
from research.frame_norm import FrameTCN
from research.cache_tools import separate_periods,insert_context
from research.context_probe import neural,CASES

def main():
    started=time.time()
    task=int(os.environ['SLURM_ARRAY_TASK_ID'])
    fi=(0,0,1,1,2,2,2)[task]
    match=idsse.MATCHES[task]
    work=Path(os.environ['RESEARCH_RESULTS'])
    out=work/'frame-norm'/match
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    torch.backends.cudnn.allow_tf32=False
    ck=torch.load(work/'frame-norm'/f'fold-{fi}.pt',weights_only=False,map_location='cpu')
    assert match in ck['validation'] and match not in ck['train']
    net=FrameTCN(len(ck['mu']),ck['state']['inp.weight'].shape[0])
    net.load_state_dict(ck['state'])
    net.cuda().eval()
    stats=(ck['mu'],ck['sd'])
    params=json.loads((ROOT/'possession/decoder.json').read_text())
    rng=np.random.default_rng(model.SEED+1)
    template=model.bar_ath_ball_mask()
    for i in ((0,1),(2,3),(4,5,6))[fi]:
        d=model.add_noise(idsse.load(idsse.MATCHES[i]),template,rng)
        if i==task:break
    base=np.load(Path(idsse.SRC)/f'cv_{match}.npz')
    assert np.array_equal(base['y'],d['y']) and np.array_equal(base['bvis'],~np.isnan(d['bx']))
    X,_=tcn.features(d);Xf,_=tcn.features(d,flip=True)
    bar=model.bar_ath()
    BX,_=tcn.features(bar);BXf,_=tcn.features(bar,flip=True)
    h1=np.flatnonzero(bar['period']==1);h2=np.flatnonzero(bar['period']==2)
    ranges=[np.arange(h1[0]),np.arange(h1[-1]+1,h2[0]),np.arange(h2[-1]+1,len(bar['period']))]
    outputs={case:[] for case in CASES}
    for features,context_features in ((X,BX),(Xf,BXf)):
        z=tcn._prep(features,stats)
        outputs['compact'].append(neural(net,z))
        p=np.empty((len(z),3),dtype=np.float32)
        for half in (1,2):
            idx=np.flatnonzero(d['period']==half)
            p[idx]=neural(net,z[idx])
        outputs['halves_independent'].append(p)
        for case,context in [('neutral_ungraded',[np.zeros((len(r),z.shape[1]),np.float32) for r in ranges]),
                             ('observed_ungraded',[tcn._prep(context_features[r],stats) for r in ranges])]:
            expanded,index=insert_context(z,d['period'],context)
            outputs[case].append(neural(net,expanded)[index])
    ps={case:.5*(outputs[case][0]+outputs[case][1][:,[1,0,2]]) for case in CASES}
    gt=np.where(d['y']<0,3,d['y']).astype(np.int8)
    interior=np.zeros(len(gt),dtype=bool)
    for half in (1,2):
        idx=np.flatnonzero(d['period']==half)
        interior[idx[1500:-1500]]=True
    rows=[]
    for case in CASES:
        with np.load(work/'context-probe'/match/f'{case}.npz') as old:
            stock=old['tcn_base']
            assert np.array_equal(old['y'],gt) and np.array_equal(old['period'],d['period'])
        error=float(abs(ps[case]-ps['compact'])[interior].max())
        if error>.002:
            raise RuntimeError(f'Unexpected remote-context dependence for per-frame normalization: {case}, {error}')
        for name,prob in [('stock_base',stock),('frame_norm',ps[case])]:
            mixed=(1-params['tcn_weight'])*base['p']+params['tcn_weight']*prob
            pred=model.decode(mixed,d['period'],params)
            a,b=separate_periods(gt,pred,d['period'])
            score=model.pm.evaluate(a,b)
            rows.append(dict(case=case,model=name,match=match,spell_f1=score['spell_f1_mean'],
                             macro_f1=score['macro_f1'],dead_f1=score['per_class']['D']['f1']))
        tmp=(out/f'{case}.npz').with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,frame_norm=ps[case],stock_base=stock,lightgbm=base['p'],y=gt,period=d['period'],interior=interior)
        tmp.replace(out/f'{case}.npz')
    report=dict(match=match,fold=fi,rows=rows,seconds=time.time()-started,
                max_interior_context_error=max(float(abs(p-ps['compact'])[interior].max()) for p in ps.values()),
                note='One training seed, fixed base capacity and fixed mixed-model decoder. Research ablation only; not a replacement submission and not separately tuned optimal decoders.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
