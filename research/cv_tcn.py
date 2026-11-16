"""Rebuild original match-held-out predictions and retain expensive fold models.

Variant settings come from the sbatch task. No competition labels are opened.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from possession import tcn, idsse
from possession.model import add_noise, bar_ath_ball_mask, SEED

@torch.no_grad()
def proba(net, stats, d):
    def one(flip):
        X = tcn.features(d, flip=flip)[0]
        Z = torch.from_numpy(tcn._prep(X, stats)).T[None]
        return torch.softmax(net(Z)[0], 0).T.numpy().astype(np.float64)
    return 0.5*(one(False)+one(True)[:, [1,0,2]])

def main():
    assert torch.cuda.is_available(), 'GPU job must provide working CUDA'
    # Original inference uses eight CPU threads; match the Slurm request here.
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    mask = bar_ath_ball_mask()
    out = Path(os.environ['RESEARCH_RESULTS'])/'folds'/('tcn'+(tcn.TAG or '_base'))
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(tag=tcn.TAG, seed=tcn.TSEED, realisations=tcn.N_REAL,
                    epochs=tcn.EPOCHS, width=tcn.WIDTH,
                    source=(ROOT/'research/SOURCE_COMMIT').read_text().strip(),
                    gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    for fi, fold in enumerate(((0,1),(2,3),(4,5,6))):
        started = time.time()
        val = [idsse.MATCHES[i] for i in fold]
        train = [m for m in idsse.MATCHES if m not in val]
        ck_path = out/f'fold-{fi}.pt'
        assert not set(train) & set(val)
        if ck_path.exists():
            ck = torch.load(ck_path, weights_only=False, map_location='cpu')
            assert ck['train'] == train and ck['validation'] == val
            net = tcn.TCN(len(ck['mu']))
            net.load_state_dict(ck['state'])
            net.eval()
            stats = (ck['mu'], ck['sd'])
        else:
            net, stats = tcn.fit(tcn._realisations(train, mask, tcn.N_REAL, tcn.TSEED), 'cuda')
            torch.save(dict(state=net.state_dict(), mu=stats[0], sd=stats[1],
                            train=train, validation=val, settings=settings), ck_path)
        # Same seed, ordering and corruption as possession.model.cv.
        rng = np.random.default_rng(SEED+1)
        for match in val:
            d = add_noise(idsse.load(match), mask, rng)
            p = proba(net, stats, d)
            path = Path(idsse.SRC)/f'cvtcn{tcn.TAG}_{match}.npz'
            np.savez(path, p=p, y=d['y'], period=d['period'], bvis=~np.isnan(d['bx']))
            print('saved', path, flush=True)
        print('fold completed', fi, 'seconds', time.time()-started, flush=True)

if __name__ == '__main__':
    main()
