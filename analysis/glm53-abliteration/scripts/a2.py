import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))
import sys, numpy as np, time
from glmlib import get, D, O, wd, wo
z=np.load(_os.path.join(_ROOT,'data')+'/lvec.npz'); T=list(z['trunk']); uD=z['uD']; uO=z['uO']
rng=np.random.default_rng(21)
EX='model.language_model.layers.%d.mlp.experts.%d.down_proj.weight'
SH='model.language_model.layers.%d.mlp.shared_experts.down_proj.weight'
# r_g^E from a held-out expert set so the test set is independent of the estimate
C=np.zeros((4096,4096),dtype=np.float64)
for L in (12,22,32,42):
    for ei in (5,90,175,260):
        a=np.asarray(get(D,wd,EX%(L,ei))[0],dtype=np.float32); b=np.asarray(get(O,wo,EX%(L,ei))[0],dtype=np.float32)
        X=a-b; C+=(X@X.T).astype(np.float64)
w,V=np.linalg.eigh(C); rgE=V[:,-1].astype(np.float32)
print(f'r_g^E (held-out estimate) . r_g^A = {abs(float(rgE@uO[0])):.6f}',flush=True)
def perp(v):
    u=v-rgE*float(rgE@v); return u/np.linalg.norm(u)
RC=np.stack([perp(rng.standard_normal(4096).astype(np.float32)) for _ in range(16)])
def rsvd(X,k=2,p=8,it=3):
    Om=rng.standard_normal((X.shape[1],k+p)).astype(np.float32); Q,_=np.linalg.qr(X@Om)
    for _ in range(it): Q,_=np.linalg.qr(X.T@Q); Q,_=np.linalg.qr(X@Q)
    Ub,s,_=np.linalg.svd(Q.T@X,full_matrices=False); return (Q@Ub)[:,:k],s[:k]
EIDS=[0,37,74,111,148,185,222,259]
rows=[]; t0=time.time()
for L in T:
    at=perp(uD[T.index(L)])
    for ei in EIDS:
        a=np.asarray(get(D,wd,EX%(L,ei))[0],dtype=np.float32); b=np.asarray(get(O,wo,EX%(L,ei))[0],dtype=np.float32)
        X=a-b; Y=X-np.outer(rgE,rgE@X); eY=float((Y.astype(np.float64)**2).sum())
        U,s=rsvd(Y,k=2)
        rows.append((L,ei,float(s[0]**2)/eY,
                     float(np.linalg.norm(at@Y)**2)/eY,
                     float(np.mean([np.linalg.norm(c@Y)**2 for c in RC]))/eY,
                     abs(float(U[:,0]@at))))
    print(f'L{L} done ({time.time()-t0:.0f}s)',flush=True)
R=np.array(rows); np.save(_os.path.join(_ROOT,'data')+'/a2.npy',R)
r1=R[:,2]; Fa=R[:,3]; Fr=R[:,4]; cs=R[:,5]
print(f'\n=== {len(R)} routed-expert cells, {len(T)} layers x {len(EIDS)} experts ===')
print(f'rho1(Y):   min {r1.min():.4f}  med {np.median(r1):.4f}  max {r1.max():.4f}')
print(f'F(A_l):    min {Fa.min():.2e}  med {np.median(Fa):.2e}  max {Fa.max():.2e}')
print(f'F(random): min {Fr.min():.2e}  med {np.median(Fr):.2e}  max {Fr.max():.2e}')
print(f'|u1.A_l|:  max {cs.max():.4f}')
i=int(np.argmax(r1)); print(f'worst-case cell: L{int(R[i,0])} expert {int(R[i,1])} rho1={r1[i]:.4f} F(A)={Fa[i]:.2e}')
j=int(np.argmax(Fa)); print(f'max F(A_l) cell: L{int(R[j,0])} expert {int(R[j,1])} F(A)={Fa[j]:.2e} ratio-to-random={Fa[j]/Fr[j]:.2f}x')
print(f'cells with F(A_l) > 3x random: {(Fa>3*Fr).sum()}/{len(R)}')
# positive control: shared expert must show structure
print('\n=== positive control: shared expert ===')
for L in (12,25,32,42):
    a=np.asarray(get(D,wd,SH%L)[0],dtype=np.float32); b=np.asarray(get(O,wo,SH%L)[0],dtype=np.float32)
    X=a-b; Y=X-np.outer(rgE,rgE@X); eY=float((Y.astype(np.float64)**2).sum())
    U,s=rsvd(Y,k=2); print(f'  L{L}: rho1(Y) = {float(s[0]**2)/eY:.4f}')
