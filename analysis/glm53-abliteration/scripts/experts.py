import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))
import sys, numpy as np
sys.path.insert(0,SP); from glmlib import get, D, O, wd, wo
z=np.load(_os.path.join(_ROOT,'data')+'/lvec.npz'); T=list(z['trunk']); uD=z['uD']; uO=z['uO']
rg=uO[0]                                    # Orca global direction
rng=np.random.default_rng(1)
def rsvd(X,k=2,p=8,it=3):
    Om=rng.standard_normal((X.shape[1],k+p)).astype(np.float32); Q,_=np.linalg.qr(X@Om)
    for _ in range(it): Q,_=np.linalg.qr(X.T@Q); Q,_=np.linalg.qr(X@Q)
    Ub,s,_=np.linalg.svd(Q.T@X,full_matrices=False); return (Q@Ub)[:,:k],s[:k]
BASE=1.0/4096
print(f'random-direction baseline F = {BASE:.2e}\n')
CAND=[('shared','model.language_model.layers.%d.mlp.shared_experts.down_proj.weight'),
      ('exp0'  ,'model.language_model.layers.%d.mlp.experts.0.down_proj.weight'),
      ('exp7'  ,'model.language_model.layers.%d.mlp.experts.7.down_proj.weight'),
      ('exp101','model.language_model.layers.%d.mlp.experts.101.down_proj.weight'),
      ('exp200','model.language_model.layers.%d.mlp.experts.200.down_proj.weight')]
print(f"{'L':>3} {'tensor':>8} {'shape':>13} {'dRel':>7} {'F(r_g)':>9} {'F(r_l)':>9} {'rho1':>6} {'|u.r_g|':>8} {'|u.r_l|':>8}")
for L in (12,22,32,42):
    rl=uD[T.index(L)]
    for tag,pat in CAND:
        k=pat%L
        if k not in wd: print(f'{L:>3} {tag:>8}  MISSING'); continue
        a=np.asarray(get(D,wd,k)[0],dtype=np.float32); b=np.asarray(get(O,wo,k)[0],dtype=np.float32)
        X=a-b
        if X.shape[0]!=4096: X=X.T
        e=float((X.astype(np.float64)**2).sum()); nW=float(np.linalg.norm(a))
        Fg=float(np.linalg.norm(rg@X)**2)/e; Fl=float(np.linalg.norm(rl@X)**2)/e
        U,s=rsvd(X,k=2); u=U[:,0]
        print(f'{L:>3} {tag:>8} {str(X.shape):>13} {np.sqrt(e)/nW:>7.4f} {Fg:>9.2e} {Fl:>9.2e} '
              f'{float(s[0]**2)/e:>6.3f} {abs(u@rg):>8.4f} {abs(u@rl):>8.4f}')
        sys.stdout.flush()
