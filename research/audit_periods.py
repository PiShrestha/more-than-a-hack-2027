"""CPU audit of compact IDSSE halftime boundaries; reconstruct the paired baseline too."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from possession import model,idsse
from research.aggregate import baseline
from research.corruptions import CASES,SEEDS,mask

def separated(gt,pred,period):
    aa,bb=[],[]
    for half in np.unique(period[period>0]):
        sel=period==half
        aa.extend([gt[sel],np.array([3])])
        bb.extend([pred[sel],np.array([3])])
    return model.pm.evaluate(np.concatenate(aa).astype(np.int8),np.concatenate(bb).astype(np.int8))

def main():
    work=Path(os.environ['RESEARCH_RESULTS'])
    params=json.loads((ROOT/'possession/decoder.json').read_text())
    template=model.bar_ath_ball_mask()
    rows=[]
    for i,match in enumerate(idsse.MATCHES):
        for seed in SEEDS:
            folder=work/'stress'/f'{match}-seed{seed}'
            report=json.loads((folder/'report.json').read_text())
            records={r['case']:r for r in report['rows']}
            base=model.add_noise(idsse.load(match),np.zeros_like(template),np.random.default_rng(seed+i))
            for case in CASES:
                with np.load(folder/f'{case}.npz') as cache:
                    dropped=mask(case,len(base['y']),template,seed+i)
                    d=dict(base,bx=np.where(dropped,np.nan,base['bx']),by=np.where(dropped,np.nan,base['by']))
                    assert np.array_equal(~np.isnan(d['bx']),cache['bvis'])
                    gt=cache['y']
                    pred=cache['decoded'] if 'decoded' in cache else model.decode(cache['ensemble'],cache['period'],params)
                    bp=baseline(d)
                    legacy=model.pm.evaluate(gt,bp)
                    assert abs(legacy['spell_f1_mean']-records[case]['smooth50']['spell_f1_mean'])<1e-12
                    er=separated(gt,pred,cache['period'])
                    br=separated(gt,bp,cache['period'])
                    rows.append(dict(match=match,seed=seed,case=case,
                        ensemble_spell=er['spell_f1_mean'],baseline_spell=br['spell_f1_mean'],
                        difference_from_legacy=er['spell_f1_mean']-records[case]['ensemble']['spell_f1_mean']))
    summary={case:dict(ensemble_mean=float(np.mean([r['ensemble_spell'] for r in rows if r['case']==case])),
                            baseline_mean=float(np.mean([r['baseline_spell'] for r in rows if r['case']==case])))
             for case in CASES}
    out=dict(evaluations=len(rows),max_abs_ensemble_difference=max(abs(r['difference_from_legacy']) for r in rows),
             period_safe_means=summary,rows=rows,
             note='Diagnostic alongside preregistered legacy scorer; does not select models or parameters. Baseline masks and scores independently reconstructed.')
    (work/'period-audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Period audit:',len(rows),'cases; maximum spellF1 difference',out['max_abs_ensemble_difference'])

if __name__=='__main__':
    main()
