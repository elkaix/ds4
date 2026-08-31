import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))
import sys, numpy as np
from glmlib import get, D, O, wd, wo
from gguf import read_dir, load
GG='/Users/panda/models/gguf/GLM-5.3-Flash-Q2.gguf'
infos, ds = read_dir(GG); rng=np.random.default_rng(0)
def spec(X,k=16,p=8,it=3):
    Om=rng.standard_normal((X.shape[1],k+p),dtype=np.float32)
    Q,_=np.linalg.qr(X@Om)
    for _ in range(it):
        Q,_=np.linalg.qr(X.T@Q); Q,_=np.linalg.qr(X@Q)
    s=np.linalg.svd(Q.T@X,compute_uv=False)[:k]
    e=float((X.astype(np.float64)**2).sum())
    return np.cumsum(s.astype(np.float64)**2)/e
K='model.language_model.layers.%d.self_attn.o_proj.weight'
LAYERS=[(0,'kda'),(5,'kda'),(8,'kda'),(10,'kda'),(12,'kda'),(14,'kda'),(16,'kda'),
        (22,'kda'),(32,'kda'),(37,'kda'),(40,'kda'),(42,'kda'),(44,'kda'),(45,'attn')]
print(f"{'L':>4} {'dD':>7} {'dO':>7} {'D/O':>6} {'cosRaw':>7} {'cosAdj':>7} "
      f"{'r1D':>6} {'r2D':>6} {'r4D':>6} {'r8D':>6} {'r1O':>6} {'r2O':>6} {'r8O':>6}")
rows={}
for L,gn in LAYERS:
    k=K%L
    a=np.asarray(get(D,wd,k)[0],dtype=np.float32); b=np.asarray(get(O,wo,k)[0],dtype=np.float32)
    nm=f'blk.{L}.{"kda_output" if gn=="kda" else "attn_output"}.weight'
    s=load(GG,nm,infos,ds,a.shape); ns=float(np.linalg.norm(s))
    Dl=a-s; Ol=b-s
    nD=float(np.linalg.norm(Dl)); nO=float(np.linalg.norm(Ol))
    ip=float((Dl.astype(np.float64)*Ol.astype(np.float64)).sum())
    cos=ip/(nD*nO)
    cD=spec(Dl); cO=spec(Ol)
    rows[L]=(nD/ns,nO/ns,cos,nD,nO,ns)
    print(f"{L:>4} {nD/ns:>7.4f} {nO/ns:>7.4f} {nD/nO:>6.2f} {cos:>7.4f} {'':>7} "
          f"{cD[0]:>6.3f} {cD[1]:>6.3f} {cD[3]:>6.3f} {cD[7]:>6.3f} {cO[0]:>6.3f} {cO[1]:>6.3f} {cO[7]:>6.3f}")
    sys.stdout.flush()
np.save(_os.path.join(_ROOT,'data','base_rows.npy'),
        np.array([[L]+list(rows[L]) for L,_ in LAYERS]))
