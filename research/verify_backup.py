"""Verify the private local backup and write a non-reconstructive checksum inventory."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from research.corruptions import CASES,SEEDS

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    root=ROOT/'research/artifacts'
    results=root/'results'
    cohort=json.loads((results/'dataset-summary.json').read_text())
    params=json.loads((ROOT/'possession/decoder.json').read_text())
    members=['lightgbm','ensemble']+['tcn'+(tag or '_base') for tag in params['tcn_tags']]
    total_frames=0
    files=0
    for item in cohort:
        match=item['match']
        with np.load(root/'data/external/idsse'/f'{match}.npz') as raw:
            gt=np.where(raw['y']<0,3,raw['y'])
            period=raw['period']
        for seed in SEEDS:
            folder=results/'stress'/f'{match}-seed{seed}'
            report=json.loads((folder/'report.json').read_text())
            assert report['decoder']==params
            assert (report['match'],report['seed'])==(match,seed)
            assert report['cpu_gpu_check']['max_probability_error']<.002
            records={r['case']:r for r in report['rows']}
            assert set(records)==set(CASES)
            for case in CASES:
                with np.load(folder/f'{case}.npz') as d:
                    assert np.array_equal(d['y'],gt) and np.array_equal(d['period'],period)
                    assert d['bvis'].shape==gt.shape and d['bvis'].dtype==np.dtype(bool)
                    assert abs((1-d['bvis'].mean())-records[case]['corruption']['missing_fraction'])<1e-12
                    for key in members:
                        p=d[key]
                        assert p.shape==(len(gt),3) and np.isfinite(p).all()
                        assert p.min()>=0 and p.max()<=1+1e-7
                        assert np.allclose(p.sum(1),1,rtol=0,atol=1e-6)
                files+=1
                total_frames+=len(gt)
    assert files==105
    assert len(list((results/'folds').rglob('*.pt')))==18
    assert len(list((results/'folds').rglob('*.txt')))==3
    replay=json.loads((results/'cpu-replay.json').read_text())
    assert len(replay['rows'])==105
    max_macro=max(abs(r['macro_f1']-json.loads((results/'stress'/f"{r['match']}-seed{r['seed']}"/'report.json').read_text())['rows'][list(CASES).index(r['case'])]['ensemble']['macro_f1']) for r in replay['rows'])
    max_spell=max(abs(r['spell_f1_mean']-json.loads((results/'stress'/f"{r['match']}-seed{r['seed']}"/'report.json').read_text())['rows'][list(CASES).index(r['case'])]['ensemble']['spell_f1_mean']) for r in replay['rows'])
    inventory=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file() or 'partial-backups' in path.parts:
            continue
        if path.suffix in ('.tmp','.part'):
            continue
        inventory.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=digest(path)))
    result=dict(status='passed',probability_files=files,member_probability_matrices=files*len(members),
        frame_evaluations=total_frames,independent_matches=7,noise_seeds=list(SEEDS),
        cpu_replay_max_macro_difference=max_macro,cpu_replay_max_spell_difference=max_spell,
        cnn_fold_checkpoints=18,tree_fold_models=3,
        research_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        verifier_sha256=digest(Path(__file__)),files=inventory)
    (ROOT/'research/backup-inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))

if __name__=='__main__':
    main()
