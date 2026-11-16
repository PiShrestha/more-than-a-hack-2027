"""Validated configuration and probability reuse for GPU-free research replay."""
import math
import numpy as np

def configuration(frozen,overrides=None):
    overrides=overrides or {}
    if not isinstance(overrides,dict):
        raise ValueError('Decoder overrides must be an object')
    unknown=set(overrides)-{'switch','die','revive','temp','tcn_weight','tcn_tags'}
    if unknown:
        raise ValueError('Unknown decoder fields: '+', '.join(sorted(unknown)))
    out=dict(frozen,**overrides)
    for key in ('switch','die','revive','tcn_weight','temp'):
        if key=='temp' and key not in out:
            continue
        value=out[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError(key+' must be a finite number')
        if value<0 or (key=='temp' and value==0):
            raise ValueError(key+' is outside its permitted range')
    if out['tcn_weight']>1:
        raise ValueError('tcn_weight must be between zero and one')
    tags=out['tcn_tags']
    if not isinstance(tags,(list,tuple)) or not tags or any(not isinstance(t,str) for t in tags):
        raise ValueError('tcn_tags must be a nonempty list of member names')
    if len(tags)!=len(set(tags)) or not set(tags)<=set(frozen['tcn_tags']):
        raise ValueError('tcn_tags must be unique members of the frozen ensemble')
    out['tcn_tags']=list(tags)
    return out

def probability(cache,params,frozen):
    """Penalty-only overrides reuse the exact cached ensemble, including its precision."""
    same_mix=(params['tcn_weight']==frozen['tcn_weight'] and
              set(params['tcn_tags'])==set(frozen['tcn_tags']))
    if same_mix:
        return cache['ensemble']
    weight=params['tcn_weight']
    if weight==0:
        return np.asarray(cache['lightgbm'],dtype=np.float64)
    neural=np.mean([cache['tcn'+(t or '_base')] for t in params['tcn_tags']],axis=0,dtype=np.float64)
    if weight==1:
        return neural
    return (1-weight)*np.asarray(cache['lightgbm'],dtype=np.float64)+weight*neural

def separate_periods(gt,pred,period):
    if not (len(gt)==len(pred)==len(period)):
        raise ValueError('Label and period lengths differ')
    halves=np.unique(period[period>0])
    if not len(halves):
        raise ValueError('No active periods')
    aa,bb=[],[]
    for half in halves:
        sel=period==half
        aa.extend([gt[sel],np.array([3])])
        bb.extend([pred[sel],np.array([3])])
    return np.concatenate(aa).astype(np.int8),np.concatenate(bb).astype(np.int8)
