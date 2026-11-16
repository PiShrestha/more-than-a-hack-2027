"""Re-score cached probabilities on CPU without Torch, training or raw tracking.

Default replay preserves cached ensemble probabilities/labels. Explicit alternative
weights use float32 cached member probabilities and may differ at tiny rounding levels.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from possession import model,idsse

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    default=Path(os.environ.get('RESEARCH_RESULTS',ROOT/'research/artifacts/results'))
    parser.add_argument('--cache',type=Path,default=default/'stress')
    parser.add_argument('--decoder',type=Path)
    parser.add_argument('--tcn-weight',type=float)
    parser.add_argument('--out',type=Path,default=default/'cpu-replay.json')
    args=parser.parse_args()
    params=json.loads((ROOT/'possession/decoder.json').read_text())
    if args.decoder is not None:
        params.update(json.loads(args.decoder.read_text()))
    if args.tcn_weight is not None:
        if not 0<=args.tcn_weight<=1:
            parser.error('--tcn-weight must be between 0 and 1')
        params['tcn_weight']=args.tcn_weight
    rows=[]
    for path in sorted(args.cache.glob('*-seed*/*.npz')):
        match,seed=path.parent.name.split('-seed')
        assert match in idsse.MATCHES,'Only the declared other-match cohort is allowed'
        d=np.load(path)
        if args.tcn_weight is None and args.decoder is None:
            p=d['ensemble']
        else:
            mean=np.mean([d['tcn'+(tag or '_base')] for tag in params['tcn_tags']],axis=0)
            p=(1-params['tcn_weight'])*d['lightgbm']+params['tcn_weight']*mean
        if args.decoder is None and args.tcn_weight is None and 'decoded' in d:
            pred=d['decoded']
        else:
            pred=model.decode(p,d['period'],params)
        score=model.pm.evaluate(d['y'],pred)
        rows.append(dict(match=match,seed=int(seed),case=path.stem,
                         spell_f1_mean=score['spell_f1_mean'],macro_f1=score['macro_f1']))
    if not rows:
        raise RuntimeError('No cached probability files found')
    assert 'torch' not in sys.modules,'CPU replay unexpectedly imported Torch'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(decoder=params,rows=rows,
        note='CPU replay only; explicit alternative weights do not constitute an independent test.'),indent=2)+'\n')
    print('CPU replay completed:',len(rows),'evaluations ->',args.out)

if __name__=='__main__':
    main()
