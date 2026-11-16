"""Retain the final tree, verify the original output, and isolate a research candidate.

Reads permitted inputs and committed weights; never reads BAR-ATH ground truth.
The official submission path is never written.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import lightgbm as lgb
from possession import model,idsse,tcn

EXPECTED='557ea3435993e72a83839000e98799be75d268f91a8e1fc0432686ab9a27c19c'

def write_csv(path,labels):
    letters=np.array(list('HADX'))[labels]
    with path.open('w',encoding='utf-8',newline='') as f:
        f.write('frame,label\n')
        f.writelines('{},{}\n'.format(i,label) for i,label in enumerate(letters))
    checked=subprocess.run([sys.executable,str(ROOT/'evaluation/validate_submission.py'),str(path)],
                           cwd=ROOT,capture_output=True,text=True,check=True)
    path.with_suffix('.validation.txt').write_text(checked.stdout)
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    started=time.time()
    out=Path(os.environ['RESEARCH_RESULTS'])/'final-tree'
    out.mkdir(parents=True,exist_ok=True)
    assert int(os.environ['SLURM_CPUS_PER_TASK'])==8 and model.LGB['num_threads']==8
    assert (model.N_ROUNDS,model.AUG,model.SUBSAMPLE)==(1600,1,3)
    params=json.loads((ROOT/'possession/decoder.json').read_text())
    dest=out/'model.txt'
    if dest.exists():
        tree=lgb.Booster(model_file=str(dest))
        print('reusing saved final tree',flush=True)
    else:
        X,y,names=model.dataset(list(idsse.MATCHES),model.bar_ath_ball_mask())
        tree=model.train(X,y)
        tmp=dest.with_suffix('.tmp')
        tree.save_model(str(tmp));tmp.replace(dest)
        (out/'training.json').write_text(json.dumps(dict(matches=list(idsse.MATCHES),features=names,
            lgb=model.LGB,rounds=model.N_ROUNDS,seed=model.SEED,source=(ROOT/'research/SOURCE_COMMIT').read_text().strip()),indent=2)+'\n')
        del X,y
    bar=model.bar_ath()
    p=model.predict_proba(tree,bar)
    reloaded=lgb.Booster(model_file=str(dest))
    pp=model.predict_proba(reloaded,bar)
    reload_error=float(np.max(abs(p-pp)))
    assert reload_error<1e-12
    np.savez_compressed(out/'bar-tree-probabilities.npz',p=pp,period=bar['period'])
    full=tcn.predict(params['tcn_tags'])
    original=(1-params['tcn_weight'])*pp+params['tcn_weight']*full
    original_labels=model.decode(original,bar['period'],params)
    original_path=out/'frozen-replay.csv'
    original_hash=write_csv(original_path,original_labels)
    if original_hash!=EXPECTED:
        (out/'reproduction-mismatch.json').write_text(json.dumps(dict(expected=EXPECTED,observed=original_hash),indent=2)+'\n')
        raise RuntimeError('Saved final tree did not reproduce the frozen submission; candidate generation aborted')
    print('Frozen CSV reproduced exactly from retained tree',flush=True)
    # Preserve the frozen full-video feature extraction; isolate only CNN sequence scope.
    X,_=tcn.features(bar);Xf,_=tcn.features(bar,flip=True)
    halves=[]
    for tag in params['tcn_tags']:
        net,stats=tcn.load_net(X.shape[1],tag)
        member=np.full_like(full,1/3)
        for half in (1,2):
            idx=np.flatnonzero(bar['period']==half)
            a=tcn.proba(net,stats,X[idx])
            b=tcn.proba(net,stats,Xf[idx])[:,[1,0,2]]
            member[idx]=.5*(a+b)
        halves.append(member)
        print('isolated-half member',tag or 'base','seconds',time.time()-started,flush=True)
    half_cnn=np.mean(halves,axis=0)
    candidate=(1-params['tcn_weight'])*pp+params['tcn_weight']*half_cnn
    labels=model.decode(candidate,bar['period'],params)
    candidate_path=out/'half-isolated-research-candidate.csv'
    candidate_hash=write_csv(candidate_path,labels)
    np.savez_compressed(out/'cnn-scope-probabilities.npz',full=full,halves=half_cnn,period=bar['period'])
    active=bar['period']>0
    report=dict(source=(ROOT/'research/SOURCE_COMMIT').read_text().strip(),decoder=params,
        saved_tree_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        tree_reload_max_probability_error=reload_error,frozen_csv_sha256=original_hash,
        candidate_csv_sha256=candidate_hash,candidate_status='research_only_not_submitted',
        active_frame_disagreements=int((labels[active]!=original_labels[active]).sum()),
        candidate_label_counts={k:int((labels==i).sum()) for i,k in enumerate('HADX')},
        seconds=time.time()-started,
        note='Candidate isolates CNN halves with unchanged weights, play features, tree and decoder. No BAR-ATH labels or accuracy evaluation; original submission files unchanged.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
