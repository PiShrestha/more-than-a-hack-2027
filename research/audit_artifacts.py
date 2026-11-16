"""Audit retained training/validation separation and probability integrity on CPU."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from possession import idsse

def main():
    work=Path(os.environ['RESEARCH_RESULTS'])
    params=json.loads((ROOT/'possession/decoder.json').read_text())
    folds=((0,1),(2,3),(4,5,6))
    weights=[]
    for fi,fold in enumerate(folds):
        val=[idsse.MATCHES[i] for i in fold]
        train=[m for m in idsse.MATCHES if m not in val]
        meta=json.loads((work/'folds/lgb'/f'fold-{fi}.json').read_text())
        assert meta['train']==train and meta['validation']==val
        assert len(meta['features'])==111 and meta['rounds']==1600
        assert (work/'folds/lgb'/f'fold-{fi}.txt').stat().st_size>0
        for tag in params['tcn_tags']:
            path=work/'folds'/('tcn'+(tag or '_base'))/f'fold-{fi}.pt'
            ck=torch.load(path,weights_only=False,map_location='cpu')
            assert ck['train']==train and ck['validation']==val
            assert len(ck['mu'])==111 and np.isfinite(ck['mu']).all()
            assert np.isfinite(ck['sd']).all() and (ck['sd']>0).all()
            assert ck['state']['inp.weight'].shape[1]==111
            assert ck['settings']['tag']==tag
            weights.append(dict(variant=tag or 'base',fold=fi,train=train,validation=val))
    probabilities=[]
    for match in idsse.MATCHES:
        with np.load(Path(idsse.SRC)/f'{match}.npz') as raw:
            yy,period=raw['y'],raw['period']
        visibility=None
        for prefix in ['cv']+['cvtcn'+t for t in params['tcn_tags']]:
            path=Path(idsse.SRC)/f'{prefix}_{match}.npz'
            with np.load(path) as d:
                p=d['p']
                assert p.shape==(len(yy),3) and np.isfinite(p).all()
                assert (p>=0).all() and (p<=1+1e-7).all()
                assert np.allclose(p.sum(1),1,atol=1e-6)
                assert np.array_equal(d['y'],yy) and np.array_equal(d['period'],period)
                if visibility is None:
                    visibility=d['bvis']
                assert np.array_equal(d['bvis'],visibility)
            probabilities.append(dict(match=match,method=prefix,frames=len(yy)))
    report=dict(status='passed',cnn_fold_checkpoints=len(weights),tree_fold_models=3,
                probability_files=len(probabilities),folds=weights,probabilities=probabilities,
                note='Confirms separation/integrity of retained artifacts; does not remove earlier model-selection optimism.')
    (work/'artifact-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Artifact audit passed:',len(weights),'CNN folds, 3 tree folds,',len(probabilities),'probability files')

if __name__=='__main__':
    main()
