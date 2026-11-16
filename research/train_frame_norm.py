"""Three fixed held-out folds for the normalization-scope ablation; no target labels."""
import json
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from possession import tcn,idsse,model
from research.frame_norm import FrameTCN,invariance_check

def main():
    fi=int(os.environ['SLURM_ARRAY_TASK_ID'])
    fold=((0,1),(2,3),(4,5,6))[fi]
    val=[idsse.MATCHES[i] for i in fold]
    train=[m for m in idsse.MATCHES if m not in val]
    out=Path(os.environ['RESEARCH_RESULTS'])/'frame-norm'
    out.mkdir(parents=True,exist_ok=True)
    protocol_hash=hashlib.sha256((ROOT/'research/frame-norm-protocol.json').read_bytes()).hexdigest()
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    assert torch.cuda.is_available()
    assert (tcn.N_REAL,tcn.EPOCHS,tcn.WIDTH,tcn.TSEED)==(4,30,96,0)
    check=invariance_check()
    print('Normalization invariance test:',check,flush=True)
    dest=out/f'fold-{fi}.pt'
    if dest.exists():
        ck=torch.load(dest,weights_only=False,map_location='cpu')
        assert ck['train']==train and ck['validation']==val
        assert ck['protocol_hash']==protocol_hash
        print('reusing trained fold',fi,flush=True)
        return
    started=time.time()
    series=tcn._realisations(train,model.bar_ath_ball_mask(),4,0)
    original=tcn.TCN
    try:
        # Keep the submitted optimizer, sampling, schedule and loss unchanged.
        # Restore the constructor after fitting; no source file is modified.
        tcn.TCN=FrameTCN
        net,stats=tcn.fit(series,'cuda')
    finally:
        tcn.TCN=original
    settings=dict(normalization='per-frame channel-group normalization',seed=0,
                  realisations=4,epochs=30,width=96,optimizer_and_sampling='unchanged possession.tcn.fit',
                  gpu=torch.cuda.get_device_name(),torch=torch.__version__,seconds=time.time()-started,
                  parameter_count=sum(p.numel() for p in net.parameters()))
    stock=torch.load(Path(os.environ['RESEARCH_RESULTS'])/'folds/tcn_base'/f'fold-{fi}.pt',weights_only=False,map_location='cpu')
    assert set(net.state_dict())==set(stock['state'])
    assert all(net.state_dict()[k].shape==v.shape for k,v in stock['state'].items())
    tmp=dest.with_suffix('.tmp')
    torch.save(dict(state=net.state_dict(),mu=stats[0],sd=stats[1],train=train,validation=val,
                    settings=settings,invariance_check=check,protocol_hash=protocol_hash),tmp)
    tmp.replace(dest)
    (out/f'fold-{fi}.json').write_text(json.dumps(dict(train=train,validation=val,settings=settings,invariance_check=check),indent=2)+'\n')
    print('saved',dest,settings,flush=True)

if __name__=='__main__':
    main()
