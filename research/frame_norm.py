"""Research-only ablation: same channel groups, normalized independently per frame."""
import torch
from possession import tcn

StockTCN=tcn.TCN

class FrameGroupNorm(torch.nn.GroupNorm):
    def forward(self,x):
        n,c,t=x.shape
        y=super().forward(x.transpose(1,2).reshape(n*t,c,1))
        return y.reshape(n,t,c).transpose(1,2)

class FrameTCN(StockTCN):
    def __init__(self,n_in,width=96):
        super().__init__(n_in,width)
        for block in self.blocks:
            old=block[1]
            replacement=FrameGroupNorm(old.num_groups,old.num_channels,eps=old.eps,affine=old.affine)
            replacement.load_state_dict(old.state_dict())
            block[1]=replacement

def invariance_check():
    torch.manual_seed(876)
    x=torch.randn(2,96,20)
    extra=torch.randn(2,96,37)*5+3
    layer=FrameGroupNorm(8,96).eval()
    a=layer(x)
    b=layer(torch.cat([x,extra],dim=-1))[:,:,:20]
    error=float((a-b).abs().max())
    if error>1e-6:
        raise RuntimeError('Per-frame normalization changed when unrelated context was appended')
    stock=torch.nn.GroupNorm(8,96).eval()
    stock_change=float((stock(x)-stock(torch.cat([x,extra],dim=-1))[:,:,:20]).abs().max())
    if stock_change<1e-3:
        raise RuntimeError('Sequence-wide control did not respond to the changed context')
    return dict(frame_norm_error=error,sequence_norm_change=stock_change)
