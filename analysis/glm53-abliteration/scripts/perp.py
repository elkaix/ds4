import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))
import sys, numpy as np
from glmlib import get, D, O, wd, wo
z=np.load(_os.path.join(_ROOT,'data')+'/lvec.npz'); T=list(z['trunk']); uD=z['uD']; uO=z['uO']; rg=uO[0].astype(np.float32)
rng=np.random.default_rng(3)
def rsvd(X,k=16,p=8,it=3):
    Om=rng.standard_normal((X.shape[1],k+p)).astype(np.float32); Q,_=np.linalg.qr(X@Om)
    for _ in range(it): Q,_=np.linalg.qr(X.T@Q); Q,_=np.linalg.qr(X@Q)
    Ub,s,_=np.linalg.svd(Q.T@X,full_matrices=False); return (Q@Ub)[:,:k],s[:k]
def perp(v):
    w=v-rg*float(rg@v); return w/np.linalg.norm(w)
# random controls orthogonal to r_g
RC=[]
for _ in range(8):
    v=rng.standard_normal(4096).astype(np.float32); RC.append(perp(v))
RC=np.stack(RC)
def effrank(s,e):
    p=(s.astype(np.float64)**2); p=p/e; p=p[p>0]
    tail=max(0.0,1.0-p.sum())
    H=-(p*np.log(p)).sum()
    return float(np.exp(H))  # lower bound (tail ignored)
SH='model.language_model.layers.%d.mlp.shared_experts.down_proj.weight'
EX='model.language_model.layers.%d.mlp.experts.%d.down_proj.weight'
print('=== SHARED EXPERT, after removing r_g ===')
print(f"{'L':>3} {'eta':>7} {'rho1Y':>7} {'rho12Y':>7} {'effR':>6} {'F(rl~)':>9} {'Frand':>9} {'|s.rl~|':>8}")
S={}
for L in T:
    k=SH%L
    a=np.asarray(get(D,wd,k)[0],dtype=np.float32); b=np.asarray(get(O,wo,k)[0],dtype=np.float32)
    X=a-b; eX=float((X.astype(np.float64)**2).sum())
    Y=X-np.outer(rg,rg@X); eY=float((Y.astype(np.float64)**2).sum())
    U,s=rsvd(Y,k=16); S[L]=U[:,0].copy()
    rt=perp(uD[T.index(L)])
    Fr=float(np.linalg.norm(rt@Y)**2)/eY
    Fc=float(np.mean([np.linalg.norm(c@Y)**2 for c in RC]))/eY
    print(f'{L:>3} {eY/eX:>7.4f} {float(s[0]**2)/eY:>7.4f} {float(s[0]**2+s[1]**2)/eY:>7.4f} '
          f'{effrank(s,eY):>6.2f} {Fr:>9.2e} {Fc:>9.2e} {abs(U[:,0]@rt):>8.4f}')
    sys.stdout.flush()
ks=sorted(S); V=np.stack([S[l] for l in ks]); G=np.abs(V@V.T)
print('\nGram |s_i.s_j| of shared-expert top direction after r_g removal')
print('    '+' '.join(f'{l:>5}' for l in ks))
for i,l in enumerate(ks): print(f'{l:>3} '+' '.join(f'{G[i,j]:>5.2f}' for j in range(len(ks))))

print('\n=== ROUTED EXPERTS, after removing r_g ===')
print(f"{'L':>3} {'exp':>5} {'eta':>7} {'rho1Y':>7} {'F(rl~)':>9} {'Frand':>9} {'|s.rl~|':>8}")
for L in (12,22,32,42):
    rt=perp(uD[T.index(L)])
    for ei in (0,7,101,200):
        k=EX%(L,ei)
        a=np.asarray(get(D,wd,k)[0],dtype=np.float32); b=np.asarray(get(O,wo,k)[0],dtype=np.float32)
        X=a-b; eX=float((X.astype(np.float64)**2).sum())
        Y=X-np.outer(rg,rg@X); eY=float((Y.astype(np.float64)**2).sum())
        U,s=rsvd(Y,k=4)
        Fr=float(np.linalg.norm(rt@Y)**2)/eY
        Fc=float(np.mean([np.linalg.norm(c@Y)**2 for c in RC]))/eY
        print(f'{L:>3} {ei:>5} {eY/eX:>7.4f} {float(s[0]**2)/eY:>7.4f} {Fr:>9.2e} {Fc:>9.2e} {abs(U[:,0]@rt):>8.4f}')
        sys.stdout.flush()
