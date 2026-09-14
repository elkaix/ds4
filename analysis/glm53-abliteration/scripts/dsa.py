import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))
import sys, numpy as np
from glmlib import get, D, O, wd, wo
from gguf import read_dir, load
GG='/Users/panda/models/gguf/GLM-5.3-Flash-Q2.gguf'
infos, ds = read_dir(GG); rng=np.random.default_rng(11)
NOISE=0.00555
z=np.load(_os.path.join(_ROOT,'data')+'/lvec.npz'); T=list(z['trunk']); uD=z['uD']; U44=z['U44']; U45=z['U45']
def rsvd(X,k=8,p=10,it=4):
    Om=rng.standard_normal((X.shape[1],k+p)).astype(np.float32); Q,_=np.linalg.qr(X@Om)
    for _ in range(it): Q,_=np.linalg.qr(X.T@Q); Q,_=np.linalg.qr(X@Q)
    Ub,s,Vt=np.linalg.svd(Q.T@X,full_matrices=False); return (Q@Ub)[:,:k],s[:k],Vt[:k]
def fit(X,W,u):
    M=np.outer(u,W.T@u)
    a=-float((X.astype(np.float64)*M).sum())/float((M.astype(np.float64)**2).sum())
    return a, float(np.linalg.norm(X+a*M))/float(np.linalg.norm(X))
K='model.language_model.layers.%d.self_attn.o_proj.weight'
DSA=[l for l in range(46) if l%4==3 and l!=45]
print(f"{'L':>3} {'dD':>7} {'dO':>7} {'r1D':>7} {'r1Dc':>7} {'r1O':>7} {'|vq|D':>7} {'alphaD':>8} "
      f"{'epsDc':>7} {'|vq|O':>7} {'alphaO':>8} {'r12D':>7} {'r18D':>7} {'cosPrev':>8} {'cosNext':>8}")
U={}
for L in DSA:
    a=np.asarray(get(D,wd,K%L)[0],dtype=np.float32); b=np.asarray(get(O,wo,K%L)[0],dtype=np.float32)
    W=load(GG,f'blk.{L}.attn_output.weight',infos,ds,a.shape); nW=float(np.linalg.norm(W))
    Dl=a-W; Ol=b-W
    dD=float(np.linalg.norm(Dl))/nW; dO=float(np.linalg.norm(Ol))/nW
    eD=float((Dl.astype(np.float64)**2).sum()); eO=float((Ol.astype(np.float64)**2).sum())
    Ud,sd,Vd=rsvd(Dl,k=8); Uo,so,Vo=rsvd(Ol,k=2); U[L]=Ud[:,0].copy()
    r1D=float(sd[0]**2)/eD; r1O=float(so[0]**2)/eO
    f=(NOISE/dD)**2 if dD>NOISE else 1.0
    r1Dc=min(1.0,r1D/max(1e-9,1-f))
    r12=float((sd[:2].astype(np.float64)**2).sum())/eD
    r18=float((sd[:8].astype(np.float64)**2).sum())/eD
    qd=W.T@Ud[:,0]; qd/=np.linalg.norm(qd); qo=W.T@Uo[:,0]; qo/=np.linalg.norm(qo)
    ad,ed=fit(Dl,W,Ud[:,0]); ao,eo=fit(Ol,W,Uo[:,0])
    edc=np.sqrt(max(0.0,ed**2-f))
    lo=[x for x in T if x<L]; hi=[x for x in T if x>L]
    cp=abs(float(Ud[:,0]@uD[T.index(lo[-1])])) if lo else float('nan')
    if hi: cn=abs(float(Ud[:,0]@uD[T.index(hi[0])]))
    elif L==43: cn=abs(float(Ud[:,0]@U44[:,0]))
    else: cn=float('nan')
    print(f'{L:>3} {dD:>7.4f} {dO:>7.4f} {r1D:>7.4f} {r1Dc:>7.4f} {r1O:>7.4f} {abs(Vd[0]@qd):>7.4f} '
          f'{ad:>8.4f} {edc:>7.4f} {abs(Vo[0]@qo):>7.4f} {ao:>8.4f} {r12:>7.4f} {r18:>7.4f} '
          f'{cp:>8.4f} {cn:>8.4f}')
    sys.stdout.flush()
print('\nL43 spectrum sig/sig1:', end=' ')
a=np.asarray(get(D,wd,K%43)[0],dtype=np.float32)
W=load(GG,'blk.43.attn_output.weight',infos,ds,a.shape)
Ud,sd,_=rsvd(a-W,k=8); print(' '.join(f'{x/sd[0]:.3f}' for x in sd))
print('|u43_1 . u44_k| :', ' '.join(f'k{i+1}:{abs(float(Ud[:,0]@U44[:,i])):.3f}' for i in range(4)))
print('|u43_1 . u_MTP| :', f'{abs(float(Ud[:,0]@U45[:,0])):.4f}')
