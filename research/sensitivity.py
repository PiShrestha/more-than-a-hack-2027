"""Fixed CPU-only sensitivity study of cached external-match probabilities.

All settings are predefined; the runner does not select or overwrite a decoder.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from possession import model,idsse
from research.cache_tools import configuration,probability,separate_periods
from research.corruptions import CASES,SEEDS

def specifications(frozen):
    specs=[dict(name='frozen',group='reference',params=configuration(frozen))]
    for field in ('switch','die','revive'):
        for factor in (.5,.75,1.5,2.):
            specs.append(dict(name=f'{field}_x{factor:g}',group='penalty',
                              params=configuration(frozen,{field:frozen[field]*factor})))
    for weight in (0.,.25,.5,.9,1.):
        specs.append(dict(name=f'tcn_weight_{weight:g}',group='mixture',
                          params=configuration(frozen,{'tcn_weight':weight})))
    for name,tags in [('base_only',['']),('three_seeds',['','_s1','_s2'])]:
        specs.append(dict(name=name,group='members',params=configuration(frozen,{'tcn_tags':tags})))
    for tag in frozen['tcn_tags']:
        specs.append(dict(name='without_'+(tag.lstrip('_') or 'base'),group='members',
                          params=configuration(frozen,{'tcn_tags':[t for t in frozen['tcn_tags'] if t!=tag]})))
    return specs

def protocol(frozen):
    return dict(source=(ROOT/'research/SOURCE_COMMIT').read_text().strip(),
                cohort=list(idsse.MATCHES),seeds=list(SEEDS),cases=list(CASES),
                configs=specifications(frozen),period_safe=True,
                purpose='Sensitivity analysis only; no decoder selection and no target-game labels.',
                precision='Penalty-only variants reuse cached ensemble precision. Mixture variants average quantized member probabilities in float64.')

def main():
    started=time.time()
    work=Path(os.environ.get('RESEARCH_RESULTS',ROOT/'research/artifacts/results'))
    frozen=json.loads((ROOT/'possession/decoder.json').read_text())
    plan=protocol(frozen)
    plan_hash=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    out=work/'sensitivity'
    (out/'cases').mkdir(parents=True,exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(dict(protocol_hash=plan_hash,**plan),indent=2)+'\n')
    inventory=json.loads((ROOT/'research/backup-inventory.json').read_text())
    hashes={f['path']:f['sha256'] for f in inventory['files']}
    all_rows=[]
    for match in idsse.MATCHES:
        for seed in SEEDS:
            for case in CASES:
                folder=f'{match}-seed{seed}'
                src=work/'stress'/folder/f'{case}.npz'
                digest=hashlib.sha256(src.read_bytes()).hexdigest()
                assert digest==hashes[f'results/stress/{folder}/{case}.npz'],str(src)
                dest=out/'cases'/f'{folder}-{case}.json'
                if dest.exists():
                    saved=json.loads(dest.read_text())
                    assert saved['protocol_hash']==plan_hash and saved['input_sha256']==digest
                    rows=saved['rows']
                else:
                    with np.load(src) as archive:
                        keys=['ensemble','lightgbm']+['tcn'+(t or '_base') for t in frozen['tcn_tags']]
                        cache={key:archive[key] for key in keys}
                        gt,period=archive['y'],archive['period']
                        for p in cache.values():
                            assert p.shape==(len(gt),3) and np.isfinite(p).all()
                        predictions=archive['decoded'] if 'decoded' in archive else None
                    mixtures={}
                    rows=[]
                    for spec in plan['configs']:
                        params=spec['params']
                        key=(params['tcn_weight'],tuple(sorted(params['tcn_tags'])))
                        if key not in mixtures:
                            mixtures[key]=probability(cache,params,frozen)
                        p=mixtures[key]
                        pred=predictions if spec['name']=='frozen' and predictions is not None else model.decode(p,period,params)
                        a,b=separate_periods(gt,pred,period)
                        score=model.pm.evaluate(a,b)
                        rows.append(dict(config=spec['name'],match=match,seed=seed,case=case,
                            spell_f1=score['spell_f1_mean'],macro_f1=score['macro_f1'],
                            pred_spells=score['n_pred_spells'],truth_spells=score['n_gt_spells'],
                            dead_precision=score['per_class']['D']['precision'],dead_recall=score['per_class']['D']['recall']))
                    tmp=dest.with_suffix('.tmp')
                    tmp.write_text(json.dumps(dict(protocol_hash=plan_hash,input_sha256=digest,rows=rows),indent=2)+'\n')
                    tmp.replace(dest)
                assert len(rows)==len(plan['configs'])
                all_rows.extend(rows)
                print('completed',folder,case,'rows',len(all_rows),flush=True)
    assert len(all_rows)==105*len(plan['configs'])
    draws=np.random.default_rng(20261116).integers(0,7,(10000,7))
    summaries=[]
    for spec in plan['configs']:
        item=dict(name=spec['name'],group=spec['group'],params=spec['params'],cases={})
        for case in CASES:
            group=[r for r in all_rows if r['config']==spec['name'] and r['case']==case]
            frozen_group=[r for r in all_rows if r['config']=='frozen' and r['case']==case]
            def match_means(rows,key):
                return np.array([np.mean([r[key] for r in rows if r['match']==m]) for m in idsse.MATCHES])
            values=match_means(group,'spell_f1')
            delta=values-match_means(frozen_group,'spell_f1')
            item['cases'][case]=dict(spell_f1=float(values.mean()),
                macro_f1=float(np.mean([r['macro_f1'] for r in group])),
                predicted_to_true_spells=sum(r['pred_spells'] for r in group)/sum(r['truth_spells'] for r in group),
                delta_from_frozen=float(delta.mean()),
                paired_delta_95=np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist(),
                per_match=dict(zip(idsse.MATCHES,values.tolist())))
        summaries.append(item)
    # Confirm the reference reproduces the independently checked period-safe audit.
    audit=json.loads((work/'period-audit.json').read_text())
    for case in CASES:
        assert abs(summaries[0]['cases'][case]['spell_f1']-audit['period_safe_means'][case]['ensemble_mean'])<1e-12
    assert 'torch' not in sys.modules
    result=dict(protocol_hash=plan_hash,configs=summaries,case_evaluations=len(all_rows),
                seconds=time.time()-started,selected_configuration=None,
                caveats=['Seven matches were already used for model selection; no independent-test claim.',
                         'Noise repetitions are averaged within each match before paired match bootstrapping.',
                         'Intervals are descriptive; no multiplicity adjustment across the sensitivity comparisons.',
                         'The frozen submission and its configuration are not changed.'])
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Sensitivity study complete:',len(plan['configs']),'configs,',len(all_rows),'evaluations;',result['seconds'],'seconds')

if __name__=='__main__':
    main()
