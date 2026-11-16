"""Resumable, single-fold reconstruction without re-running decoder grid search."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from possession import model,idsse

def main():
    model.LGB['num_threads']=int(os.environ['SLURM_CPUS_PER_TASK'])
    fi=int(os.environ['SLURM_ARRAY_TASK_ID'])
    fold=((0,1),(2,3),(4,5,6))[fi]
    val=[idsse.MATCHES[i] for i in fold]
    tr=[m for m in idsse.MATCHES if m not in val]
    out=Path(os.environ['RESEARCH_RESULTS'])/'folds/lgb'
    out.mkdir(parents=True,exist_ok=True)
    for m in val:
        old=Path(idsse.SRC)/f'cv_audit_{m}.npz'
        new=Path(idsse.SRC)/f'cv_{m}.npz'
        if old.exists() and not new.exists():
            new.write_bytes(old.read_bytes())
    if (all((Path(idsse.SRC)/f'cv_{m}.npz').exists() for m in val)
            and not os.environ.get('RETAIN_MODEL')):
        print('reusing complete fold',fi,val,flush=True)
        return
    started=time.time()
    mask=model.bar_ath_ball_mask()
    print('building features',fi,tr,flush=True)
    X,y,names=model.dataset(tr,mask)
    print('features ready',X.shape,'seconds',time.time()-started,flush=True)
    booster=model.train(X,y)
    booster.save_model(str(out/f'fold-{fi}.txt'))
    (out/f'fold-{fi}.json').write_text(json.dumps(dict(train=tr,validation=val,
        features=names,lgb=model.LGB,rounds=model.N_ROUNDS),indent=2)+'\n')
    print('training completed','seconds',time.time()-started,flush=True)
    rng=np.random.default_rng(model.SEED+1)
    for m in val:
        d=model.add_noise(idsse.load(m),mask,rng)
        path=Path(idsse.SRC)/f'cv_{m}.npz'
        if path.exists():
            old=np.load(path)
            assert np.array_equal(old['y'],d['y']) and np.array_equal(old['bvis'],~np.isnan(d['bx']))
            print('reusing',m,flush=True)
            continue
        p=model.predict_proba(booster,d)
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez(f,p=p,y=d['y'],period=d['period'],bvis=~np.isnan(d['bx']))
        tmp.replace(path)
        print('saved',m,'seconds',time.time()-started,flush=True)

if __name__=='__main__':
    main()
