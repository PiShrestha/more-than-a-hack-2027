"""Duration-stratified possession detection recall on reconstructed other-match CV."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from possession import model,idsse

def main():
    work=Path(os.environ['RESEARCH_RESULTS'])
    params=json.loads((ROOT/'possession/decoder.json').read_text())
    bins=((0,2),(2,5),(5,10),(10,20),(20,float('inf')))
    per_match=[]
    for match in idsse.MATCHES:
        a=np.load(Path(idsse.SRC)/f'cv_{match}.npz')
        tp=np.mean([np.load(Path(idsse.SRC)/f'cvtcn{tag}_{match}.npz')['p'] for tag in params['tcn_tags']],axis=0)
        p=(1-params['tcn_weight'])*a['p']+params['tcn_weight']*tp
        pred=model.decode(p,a['period'],params)
        aa,bb=[],[]
        for half in (1,2):
            sel=a['period']==half
            aa.extend([a['y'][sel],np.array([3])])
            bb.extend([pred[sel],np.array([3])])
        gt=np.concatenate(aa).astype(np.int8)
        pred=np.concatenate(bb).astype(np.int8)
        runs=model.pm.valid_runs(gt)
        g=model.pm.possession_spells(gt,runs)
        q=model.pm.possession_spells(pred,runs)
        overlaps=model.pm.spell_overlaps(g,q)
        flags=np.vstack([model.pm.matched_gt_flags(overlaps,len(g),t) for t in model.pm.SPELL_IOUS])
        durations=np.array([(b-a)/25 for a,b,_ in g])
        rows=[]
        for lo,hi in bins:
            sel=(durations>=lo)&(durations<hi)
            rows.append(dict(bin_s=f'{lo}-{hi}',truth_spells=int(sel.sum()),
                             summed_threshold_hits=int(flags[:,sel].sum()),
                             mean_detection_recall=float(flags[:,sel].mean()) if sel.any() else None))
        per_match.append(dict(match=match,truth_spells=len(g),predicted_spells=len(q),bins=rows))
    totals=[]
    for i,(lo,hi) in enumerate(bins):
        count=sum(r['bins'][i]['truth_spells'] for r in per_match)
        hits=sum(r['bins'][i]['summed_threshold_hits'] for r in per_match)
        totals.append(dict(bin_s=f'{lo}-{hi}',truth_spells=count,
                           mean_detection_recall=hits/(len(model.pm.SPELL_IOUS)*count) if count else None))
    result=dict(bins=totals,matches=per_match,
                predicted_to_true_spell_ratio=sum(r['predicted_spells'] for r in per_match)/sum(r['truth_spells'] for r in per_match),
                note='Descriptive selected-CV analysis. Recall averaged over IoU .3-.7, not spellF1 within bins; halves separated explicitly. No BAR-ATH labels.')
    (work/'failure-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='matches'},indent=2))

if __name__=='__main__':
    main()
