import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))
import sys, numpy as np
from glmlib import get, D, O, wd, wo
z=np.load(_os.path.join(_ROOT,'data')+'/lvec.npz'); T=list(z['trunk']); uD=z['uD']; uO=z['uO']; rg=uO[0]
rng=np.random.default_rng(2)
def rsvd(X,k=3,p=8,it=3):
    Om=rng.standard_normal((X.shape[1],k+p)).astype(np.float32); Q,_=np.linalg.qr(X@Om)
    for _ in range(it): Q,_=np.linalg.qr(X.T@Q); Q,_=np.linalg.qr(X@Q)
    Ub,s,_=np.linalg.svd(Q.T@X,full_matrices=False); return (Q@Ub)[:,:k],s[:k]
P='model.language_model.layers.%d.mlp.shared_experts.down_proj.weight'
LS=[l for l in T]
res={}
print(f"{'L':>3} {'dRel':>7} {'F(r_g)':>8} {'rho1':>6} {'rho2':>6} {'|u1.r_g|':>9} {'|u1.r_l|':>9}")
for L in LS:
    k=P%L
    if k not in wd: continue
    a=np.asarray(get(D,wd,k)[0],dtype=np.float32); b=np.asarray(get(O,wo,k)[0],dtype=np.float32)
    X=a-b
    e=float((X.astype(np.float64)**2).sum()); nW=float(np.linalg.norm(a))
    U,s=rsvd(X,k=3); res[L]=U[:,0].copy()
    rl=uD[T.index(L)]
    print(f'{L:>3} {np.sqrt(e)/nW:>7.4f} {float(np.linalg.norm(rg@X)**2)/e:>8.4f} '
          f'{float(s[0]**2)/e:>6.3f} {float((s[0]**2+s[1]**2))/e:>6.3f} {abs(U[:,0]@rg):>9.4f} {abs(U[:,0]@rl):>9.4f}')
    sys.stdout.flush()
ks=sorted(res)
V=np.stack([res[l] for l in ks]); G=np.abs(V@V.T)
print('\nshared-expert top-direction gram |u_i.u_j|')
print('    '+' '.join(f'{l:>5}' for l in ks))
for i,l in enumerate(ks): print(f'{l:>3} '+' '.join(f'{G[i,j]:>5.2f}' for j in range(len(ks))))
