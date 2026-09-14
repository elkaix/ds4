import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))
import sys, numpy as np
from glmlib import get, D, O, wd, wo
from gguf import read_dir, load
GG='/Users/panda/models/gguf/GLM-5.3-Flash-Q2.gguf'
infos, ds = read_dir(GG); rng=np.random.default_rng(0)
def rsvd(X,k=8,p=10,it=4):
    Om=rng.standard_normal((X.shape[1],k+p)).astype(np.float32)
    Q,_=np.linalg.qr(X@Om)
    for _ in range(it):
        Q,_=np.linalg.qr(X.T@Q); Q,_=np.linalg.qr(X@Q)
    Ub,s,Vt=np.linalg.svd(Q.T@X,full_matrices=False)
    return (Q@Ub)[:,:k], s[:k], Vt[:k]
K='model.language_model.layers.%d.self_attn.o_proj.weight'
NOISE=0.00555
def fit(dl, W, u):
    p = W.T@u                    # in-space direction dictated by projection form
    M = np.outer(u, p)
    a = -float((dl.astype(np.float64)*M).sum())/float((M.astype(np.float64)**2).sum())
    r = float(np.linalg.norm(dl + a*M)); nd=float(np.linalg.norm(dl))
    return a, r/nd, p/np.linalg.norm(p)
print(f"{'L':>4} {'mdl':>5} {'|v.q|':>7} {'eps':>7} {'epsCorr':>8} {'alpha':>8} {'nD/nW':>7}")
for L in (12,22,32,40,42):
    k=K%L
    a=np.asarray(get(D,wd,k)[0],dtype=np.float32); b=np.asarray(get(O,wo,k)[0],dtype=np.float32)
    W=load(GG,f'blk.{L}.kda_output.weight',infos,ds,a.shape)
    nW=float(np.linalg.norm(W))
    for tag,X in (('D',a-W),('O',b-W)):
        U,s,Vt=rsvd(X,k=2); u=U[:,0]; v=Vt[0]
        al,eps,q=fit(X,W,u)
        nd=float(np.linalg.norm(X)); f=NOISE*nW/nd
        ec=float(np.sqrt(max(0.0,eps**2-f**2)))
        print(f"{L:>4} {tag:>5} {abs(v@q):>7.4f} {eps:>7.4f} {ec:>8.4f} {al:>8.4f} {nd/nW:>7.4f}")
    sys.stdout.flush()

# ---- L44 + MTP: multi-direction projection model ----
for L,gn in ((44,'kda_output'),(45,'attn_output')):
    kk=K%L
    a=np.asarray(get(D,wd,kk)[0],dtype=np.float32); b=np.asarray(get(O,wo,kk)[0],dtype=np.float32)
    W=load(GG,f'blk.{L}.{gn}.weight',infos,ds,a.shape); nW=float(np.linalg.norm(W))
    for tag,X in (('D',a-W),('O',b-W)):
        U,s,Vt=rsvd(X,k=8)
        nd2=float((X.astype(np.float64)**2).sum())
        print(f'\nL{L} {tag}  dRel={np.sqrt(nd2)/nW:.4f}')
        print(f"   {'k':>2} {'epsProj':>8} {'epsRankK':>9}")
        for k in (1,2,3,4,6,8):
            Uk=U[:,:k]; P=(Uk.T@W).astype(np.float64)      # k x N
            UtX=(Uk.T@X).astype(np.float64)
            A=-UtX@P.T@np.linalg.inv(P@P.T+1e-6*np.eye(k))
            res=float(((UtX+A@P)**2).sum())+ (nd2-float((UtX**2).sum()))
            epsP=np.sqrt(res/nd2)
            epsK=np.sqrt(max(0.0,1.0-float((s[:k].astype(np.float64)**2).sum())/nd2))
            print(f"   {k:>2} {epsP:>8.4f} {epsK:>9.4f}")
        sys.stdout.flush()
