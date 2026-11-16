"""Evaluate frozen choices on reconstructed OOF predictions; never select on BAR-ATH."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import model, idsse
from starter.baseline import smooth
import numpy as np

def baseline(d):
    dist = np.hypot(d['px']-d['bx'][:,None], d['py']-d['by'][:,None])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    nearest = np.argmin(dist, axis=1)
    raw = np.where(np.isfinite(np.min(dist,axis=1)), d['team'][nearest], -1)
    letters = smooth(raw,50)
    codes = np.where(letters=='H',0,np.where(letters=='A',1,2))
    return np.where(d['period']>0,codes,3).astype(np.int8)

def main():
    params = json.loads((ROOT/'possession/decoder.json').read_text())
    rows = []
    for fold in ((0,1),(2,3),(4,5,6)):
        rng = np.random.default_rng(model.SEED+1)
        for i in fold:
            match = idsse.MATCHES[i]
            d = model.add_noise(idsse.load(match),model.bar_ath_ball_mask(),rng)
            a = np.load(Path(idsse.SRC)/f'cv_{match}.npz')
            ps = []
            for tag in params['tcn_tags']:
                b = np.load(Path(idsse.SRC)/f'cvtcn{tag}_{match}.npz')
                assert np.array_equal(a['y'],b['y'])
                assert np.array_equal(a['period'],b['period'])
                assert np.array_equal(a['bvis'],b['bvis'])
                ps.append(b['p'])
            assert np.array_equal(a['bvis'],~np.isnan(d['bx']))
            p = (1-params['tcn_weight'])*a['p']+params['tcn_weight']*np.mean(ps,axis=0)
            pred = model.decode(p,a['period'],params)
            bp = baseline(d)
            gt = np.where(a['y']<0,3,a['y'])
            metrics = {}
            for name, labels in [('ensemble',pred),('smooth50',bp)]:
                r = model.pm.evaluate(gt.astype(np.int8),labels.astype(np.int8))
                metrics[name] = {k:r[k] for k in ('spell_f1_mean','macro_f1','f1_D','precision_D','recall_D','n_pred_spells','n_gt_spells')}
                # Original scorer inputs may join adjacent halves; explicitly audit half separation.
                parts_gt,parts_pr = [],[]
                for half in np.unique(a['period'][a['period']>0]):
                    sel = a['period']==half
                    parts_gt.extend([gt[sel],np.array([3])])
                    parts_pr.extend([labels[sel],np.array([3])])
                rr = model.pm.evaluate(np.concatenate(parts_gt).astype(np.int8),np.concatenate(parts_pr).astype(np.int8))
                metrics[name]['period_safe_spell_f1'] = rr['spell_f1_mean']
            rows.append(dict(match=match,metrics=metrics))
            print(match,metrics,flush=True)
    x = np.array([r['metrics']['ensemble']['spell_f1_mean'] for r in rows])
    b = np.array([r['metrics']['smooth50']['spell_f1_mean'] for r in rows])
    draws = np.random.default_rng(20261116).integers(0,len(rows),(10000,len(rows)))
    report = dict(source=(ROOT/'research/SOURCE_COMMIT').read_text().strip(),
                  decoder=params, matches=rows,
                  ensemble_mean=float(x.mean()),baseline_mean=float(b.mean()),
                  match_bootstrap_95=np.percentile(x[draws].mean(1),[2.5,97.5]).tolist(),
                  paired_improvement_95=np.percentile((x-b)[draws].mean(1),[2.5,97.5]).tolist(),
                  caveat='These matches were used for model/decoder selection. Bootstrap does not remove selection optimism; this is not an untouched test.')
    out = Path(os.environ['RESEARCH_RESULTS'])
    (out/'cv-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='matches'},indent=2))

if __name__=='__main__':
    main()
