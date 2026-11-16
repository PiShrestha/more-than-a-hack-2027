"""Paired match summary of the single-seed normalization-scope ablation."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from possession.idsse import MATCHES

def main():
    work=Path(os.environ['RESEARCH_RESULTS'])
    reports=[json.loads((work/'frame-norm'/m/'report.json').read_text()) for m in MATCHES]
    draws=np.random.default_rng(20261116).integers(0,7,(10000,7))
    rows=[]
    for case in ('compact','halves_independent','neutral_ungraded','observed_ungraded'):
        def values(name,key):
            return np.array([next(x[key] for x in r['rows'] if x['case']==case and x['model']==name) for r in reports])
        a,b=values('frame_norm','spell_f1'),values('stock_base','spell_f1')
        delta=a-b
        rows.append(dict(case=case,frame_norm_spell=float(a.mean()),stock_base_spell=float(b.mean()),
            frame_norm_macro=float(values('frame_norm','macro_f1').mean()),stock_base_macro=float(values('stock_base','macro_f1').mean()),
            paired_spell_delta=float(delta.mean()),paired_delta_95=np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist()))
    result=dict(cases=rows,matches=7,evaluations=56,training_seeds=1,
                max_frame_norm_interior_context_error=max(r['max_interior_context_error'] for r in reports),
                selected_configuration=None,
                caveats=['One training seed per fold; no claim to isolate training hardware randomness.',
                         'Same base network parameter shapes/capacity, training recipe and fixed decoder; no separately optimized decoders.',
                         'Same previously selected seven validation matches; no BAR-ATH labels and no submission replacement.'])
    (work/'frame-norm-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
