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
from research.cache_tools import configuration,probability,separate_periods

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    default=Path(os.environ.get('RESEARCH_RESULTS',ROOT/'research/artifacts/results'))
    parser.add_argument('--cache',type=Path,default=default/'stress')
    parser.add_argument('--decoder',type=Path)
    parser.add_argument('--tcn-weight',type=float)
    parser.add_argument('--period-safe',action='store_true',help='Separate compact other-match halves before scoring')
    parser.add_argument('--out',type=Path,default=default/'cpu-replay.json')
    args=parser.parse_args()
    frozen=json.loads((ROOT/'possession/decoder.json').read_text())
    overrides={}
    if args.decoder is not None:
        overrides=json.loads(args.decoder.read_text())
        if not isinstance(overrides,dict):
            parser.error('Decoder overrides must be a JSON object')
    if args.tcn_weight is not None:
        overrides['tcn_weight']=args.tcn_weight
    try:
        params=configuration(frozen,overrides)
    except ValueError as error:
        parser.error(str(error))
    rows=[]
    for path in sorted(args.cache.glob('*-seed*/*.npz')):
        match,seed=path.parent.name.split('-seed')
        assert match in idsse.MATCHES,'Only the declared other-match cohort is allowed'
        with np.load(path) as d:
            p=probability(d,params,frozen)
            if params==frozen and 'decoded' in d:
                pred=d['decoded']
            else:
                pred=model.decode(p,d['period'],params)
            gt=d['y']
            if args.period_safe:
                gt,pred=separate_periods(gt,pred,d['period'])
            score=model.pm.evaluate(gt,pred)
        rows.append(dict(match=match,seed=int(seed),case=path.stem,
                         spell_f1_mean=score['spell_f1_mean'],macro_f1=score['macro_f1']))
    if not rows:
        raise RuntimeError('No cached probability files found')
    assert 'torch' not in sys.modules,'CPU replay unexpectedly imported Torch'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(decoder=params,period_safe=args.period_safe,rows=rows,
        note='CPU replay only; explicit alternative weights do not constitute an independent test.'),indent=2)+'\n')
    print('CPU replay completed:',len(rows),'evaluations ->',args.out)

if __name__=='__main__':
    main()
