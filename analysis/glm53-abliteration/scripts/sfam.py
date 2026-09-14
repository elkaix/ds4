import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))
import sys, numpy as np
from glmlib import get, D, O, wd, wo
z=np.load(_os.path.join(_ROOT,'data')+'/lvec.npz'); T=list(z['trunk']); uD=z['uD']; uO=z['uO']
rgA=uO[0].astype(np.float32)
rng=np.random.default_rng(5)
def rsvd(X,k=4,p=10,it=4):
    Om=rng.standard_normal((X.shape[1],k+p)).astype(np.float32); Q,_=np.linalg.qr(X@Om)
    for _ in range(it): Q,_=np.linalg.qr(X.T@Q); Q,_=np.linalg.qr(X@Q)
    Ub,s,Vt=np.linalg.svd(Q.T@X,full_matrices=False); return (Q@Ub)[:,:k],s[:k],Vt[:k]
EX='model.language_model.layers.%d.mlp.experts.%d.down_proj.weight'
SH='model.language_model.layers.%d.mlp.shared_experts.down_proj.weight'
# ---- 1. r_g estimated from routed-expert deltas (accumulate covariance) ----
C=np.zeros((4096,4096),dtype=np.float64)
for L in (12,22,32,42):
    for ei in (0,7,101,200):
        a=np.asarray(get(D,wd,EX%(L,ei))[0],dtype=np.float32); b=np.asarray(get(O,wo,EX%(L,ei))[0],dtype=np.float32)
        X=a-b; C+=(X@X.T).astype(np.float64)
w,V=np.linalg.eigh(C); rgE=V[:,-1].astype(np.float32)
print(f'|r_g^E . r_g^A| = {abs(float(rgE@rgA)):.6f}')
print(f'eig spectrum top5 / trace: {" ".join(f"{w[-i]/w.sum():.4f}" for i in range(1,6))}')
def strip(X,r): return X-np.outer(r,r@X)
print('\n=== shared expert, r_g^E removed ===')
print(f"{'L':>3} {'eta':>7} {'rho1':>7} {'stabR':>7} {'|s.rgA_res|':>12} {'|v.q|':>7} {'alphaS':>8} {'eps':>7} {'dRelS':>8}")
S={}
for L in T:
    a=np.asarray(get(D,wd,SH%L)[0],dtype=np.float32); b=np.asarray(get(O,wo,SH%L)[0],dtype=np.float32)
    X=a-b; eX=float((X.astype(np.float64)**2).sum())
    Y=strip(X,rgE); eY=float((Y.astype(np.float64)**2).sum())
    U,sv,Vt=rsvd(Y,k=4); s=U[:,0]; v=Vt[0]; S[L]=s.copy()
    r1=float(sv[0]**2)/eY
    q=b.T@s; nq=float(np.linalg.norm(q)); q=q/nq          # W_O^T s == W_stock^T s  (s ⟂ r_g)
    M=np.outer(s,b.T@s)
    al=-float((Y.astype(np.float64)*M).sum())/float((M.astype(np.float64)**2).sum())
    eps=float(np.linalg.norm(Y+al*M))/np.sqrt(eY)
    print(f'{L:>3} {eY/eX:>7.4f} {r1:>7.4f} {1/r1:>7.1f} {abs(float(s@rgA)):>12.4f} '
          f'{abs(v@q):>7.4f} {al:>8.4f} {eps:>7.4f} {sv[0]/np.linalg.norm(a):>8.5f}')
    sys.stdout.flush()
ks=sorted(S); V2=np.stack([S[l] for l in ks]); G=np.abs(V2@V2.T)
print('\nL12-18 mutual |s_i.s_j| after r_g^E removal: '
      + ' '.join(f'{ks[i]}-{ks[j]}:{G[i,j]:.2f}' for i in range(6) for j in range(i+1,6)))
band=[i for i,l in enumerate(ks) if 20<=l<=37]
print('band L20-37 max off-diag |s_i.s_j| = %.3f'%max(G[i,j] for i in band for j in band if i<j))
# cross-family
print('\n|S_l . A_l| in band: '+' '.join(f'{ks[i]}:{abs(float(S[ks[i]]@uD[T.index(ks[i])])):.3f}' for i in band))
