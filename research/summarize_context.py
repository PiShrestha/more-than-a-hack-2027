"""Paired match summary of the fixed CNN context diagnostic."""
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
    reports=[json.loads((work/'context-probe'/m/'report.json').read_text()) for m in MATCHES]
    cases=('compact','halves_independent','neutral_ungraded','observed_ungraded')
    draws=np.random.default_rng(20261116).integers(0,7,(10000,7))
    reference=np.array([r['rows'][0]['spell_f1'] for r in reports])
    output=[]
    for case in cases:
        rows=[next(x for x in r['rows'] if x['case']==case) for r in reports]
        scores=np.array([r['spell_f1'] for r in rows])
        delta=scores-reference
        output.append(dict(case=case,spell_f1=float(scores.mean()),
            macro_f1=float(np.mean([r['macro_f1'] for r in rows])),
            delta_from_compact=float(delta.mean()),
            paired_delta_95=np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist(),
            interior_argmax_agreement=float(np.mean([r['interior_argmax_agreement_with_compact'] for r in rows])),
            interior_mean_abs_probability_change=float(np.mean([r['interior_mean_abs_probability_change'] for r in rows])),
            per_match=rows))
    summary=dict(cases=output,matches=7,evaluations=28,
        max_compact_cpu_gpu_probability_error=max(x['cpu_gpu_max_probability_difference'] for r in reports for x in r['cpu_gpu_checks']),
        note='Play features, model weights, tree probabilities and decoder fixed. No BAR-ATH labels. Context vectors private and not exported; results are selected-validation diagnostics.')
    (work/'context-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
