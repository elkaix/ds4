import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))
import sys, numpy as np
from glmlib import get, D, O, wd, wo
from gguf import read_dir, load
GG='/Users/panda/models/gguf/GLM-5.3-Flash-Q2.gguf'
infos, ds = read_dir(GG); rng=np.random.default_rng(7)
NOISE=0.00555
def rsvd(X,k=8,p=10,it=4):
    Om=rng.standard_normal((X.shape[1],k+p)).astype(np.float32); Q,_=np.linalg.qr(X@Om)
    for _ in range(it): Q,_=np.linalg.qr(X.T@Q); Q,_=np.linalg.qr(X@Q)
    Ub,s,Vt=np.linalg.svd(Q.T@X,full_matrices=False); return (Q@Ub)[:,:k],s[:k],Vt[:k]
def fitproj(X,W,u):
    M=np.outer(u,W.T@u)
    a=-float((X.astype(np.float64)*M).sum())/float((M.astype(np.float64)**2).sum())
    return a, float(np.linalg.norm(X+a*M))/float(np.linalg.norm(X))
K='model.language_model.layers.%d.self_attn.o_proj.weight'
LS=[l for l in range(46) if l%4!=3 and l!=45]
print(f"{'L':>3} {'dD':>7} {'dO':>7} {'r1D':>7} {'r1Dc':>7} {'r1O':>7} {'cosAdj':>8} "
      f"{'|vq|D':>7} {'alphaD':>8} {'epsDc':>7} {'|vq|O':>7} {'alphaO':>8} {'cos(rl,rprev)':>13}")
prev=None; rows=[]
for L in LS:
    k=K%L
    a=np.asarray(get(D,wd,k)[0],dtype=np.float32); b=np.asarray(get(O,wo,k)[0],dtype=np.float32)
    W=load(GG,f'blk.{L}.kda_output.weight',infos,ds,a.shape); nW=float(np.linalg.norm(W))
    Dl=a-W; Ol=b-W
    nD=float(np.linalg.norm(Dl)); nO=float(np.linalg.norm(Ol))
    dD,dO=nD/nW,nO/nW
    Ud,sd,Vd=rsvd(Dl,k=4); Uo,so,Vo=rsvd(Ol,k=4)
    r1D=float(sd[0]**2)/float((Dl.astype(np.float64)**2).sum())
    r1O=float(so[0]**2)/float((Ol.astype(np.float64)**2).sum())
    fD=(NOISE/dD)**2 if dD>NOISE else 1.0; fO=(NOISE/dO)**2
    r1Dc=min(1.0,r1D/max(1e-9,1-fD)); 
    ip=float((Dl.astype(np.float64)*Ol.astype(np.float64)).sum())
    cosAdj=(ip-NOISE**2*nW**2)/(nD*nO*np.sqrt(max(1e-9,(1-fD)*(1-fO))))
    qd=W.T@Ud[:,0]; qd/=np.linalg.norm(qd); qo=W.T@Uo[:,0]; qo/=np.linalg.norm(qo)
    ad,ed=fitproj(Dl,W,Ud[:,0]); ao,eo=fitproj(Ol,W,Uo[:,0])
    edc=np.sqrt(max(0.0,ed**2-fD))
    cp=abs(float(Ud[:,0]@prev)) if prev is not None else float('nan')
    prev=Ud[:,0].copy()
    print(f'{L:>3} {dD:>7.4f} {dO:>7.4f} {r1D:>7.4f} {r1Dc:>7.4f} {r1O:>7.4f} {cosAdj:>8.4f} '
          f'{abs(Vd[0]@qd):>7.4f} {ad:>8.4f} {edc:>7.4f} {abs(Vo[0]@qo):>7.4f} {ao:>8.4f} {cp:>13.4f}')
    rows.append((L,dD,dO,r1Dc,ad,abs(Vd[0]@qd),cp)); sys.stdout.flush()
np.save(_os.path.join(_ROOT,'data')+'/sweep.npy',np.array(rows))
