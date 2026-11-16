"""CPU-only paired match-bootstrap summary; no training or model selection."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from possession.idsse import MATCHES
from research.corruptions import CASES,SEEDS

def main():
    work=Path(os.environ['RESEARCH_RESULTS'])
    reports=[]
    for match in MATCHES:
        for seed in SEEDS:
            p=work/'stress'/f'{match}-seed{seed}'/'report.json'
            d=json.loads(p.read_text())
            assert (d['match'],d['seed'])==(match,seed)
            assert [r['case'] for r in d['rows']]==list(CASES)
            reports.append(d)
    rows=[r for d in reports for r in d['rows']]
    assert len(rows)==105
    rng=np.random.default_rng(20261116)
    draws=rng.integers(0,len(MATCHES),(10000,len(MATCHES)))
    summaries={}
    vectors={}
    for case in CASES:
        group=[r for r in rows if r['case']==case]
        scores={}
        for method in ('ensemble','smooth50'):
            scores[method]={}
            for metric in ('spell_f1_mean','macro_f1'):
                values=np.array([[next(r[method][metric] for r in group if r['match']==m and r['seed']==s)
                                  for s in SEEDS] for m in MATCHES])
                means=values.mean(1)  # three noise seeds do not count as extra matches
                scores[method][metric]=dict(mean=float(means.mean()),
                    match_bootstrap_95=np.percentile(means[draws].mean(1),[2.5,97.5]).tolist(),
                    per_match_means=dict(zip(MATCHES,means.tolist())))
                vectors[(case,method,metric)]=means
        delta=vectors[(case,'ensemble','spell_f1_mean')]-vectors[(case,'smooth50','spell_f1_mean')]
        summaries[case]=dict(scores=scores,paired_spell_improvement=float(delta.mean()),
            paired_improvement_95=np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist(),
            missing_fraction_mean=float(np.mean([r['corruption']['missing_fraction'] for r in group])),
            realised_mean_gap_s=float(np.mean([r['corruption']['mean_gap_s'] for r in group])),
            ensemble_brier_mean=float(np.mean([r['ensemble']['brier_multiclass'] for r in group])),
            ensemble_ece15_mean=float(np.mean([r['ensemble']['ece15'] for r in group])),
            dead_ball_precision_mean=float(np.mean([r['ensemble']['dead_ball']['precision'] for r in group])),
            dead_ball_recall_mean=float(np.mean([r['ensemble']['dead_ball']['recall'] for r in group])))
    for case in CASES:
        drop=vectors[('visible','ensemble','spell_f1_mean')]-vectors[(case,'ensemble','spell_f1_mean')]
        summaries[case]['paired_spell_drop_from_visible']=float(drop.mean())
        summaries[case]['paired_drop_95']=np.percentile(drop[draws].mean(1),[2.5,97.5]).tolist()
    result=dict(cases=summaries,matches=7,seeds=list(SEEDS),evaluations=len(rows),
        max_cpu_gpu_probability_error=max(d['cpu_gpu_check']['max_probability_error'] for d in reports),
        min_cpu_gpu_argmax_agreement=min(d['cpu_gpu_check']['argmax_agreement'] for d in reports),
        caveats=['The seven matches were previously used for model selection; no untouched-test claim.',
                 'Noise repetitions are paired within matches and averaged before match bootstrapping.',
                 'Artificial masks do not reproduce game-state-dependent real occlusion.',
                 'All cases retain simulated player dropout and label-dependent other-match dead-ball carries.',
                 'Calibration metrics are descriptive, without post-hoc calibration or selection.'])
    (work/'stress-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
