import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))
import sys, numpy as np
from glmlib import get, D, O, wd, wo
z=np.load(_os.path.join(_ROOT,'data')+'/lvec.npz'); T=list(z['trunk'])
rng=np.random.default_rng(31)
def rsvd(X,k=2,p=8,it=3):
    Om=rng.standard_normal((X.shape[1],k+p)).astype(np.float32); Q,_=np.linalg.qr(X@Om)
    for _ in range(it): Q,_=np.linalg.qr(X.T@Q); Q,_=np.linalg.qr(X@Q)
    Ub,s,_=np.linalg.svd(Q.T@X,full_matrices=False); return (Q@Ub)[:,:k],s[:k]
EX='model.language_model.layers.%d.mlp.experts.%d.down_proj.weight'
SH='model.language_model.layers.%d.mlp.shared_experts.down_proj.weight'
# r_g^E, held-out estimator set (same as A2)
C=np.zeros((4096,4096),dtype=np.float64)
for L in (12,22,32,42):
    for ei in (5,90,175,260):
        a=np.asarray(get(D,wd,EX%(L,ei))[0],dtype=np.float32); b=np.asarray(get(O,wo,EX%(L,ei))[0],dtype=np.float32)
        X=a-b; C+=(X@X.T).astype(np.float64)
w,V=np.linalg.eigh(C); rgE=V[:,-1].astype(np.float32)
print('r_g^E . r_g^A =', abs(float(rgE@z['uO'][0])))
# S_l family
S=[]
for L in T:
    a=np.asarray(get(D,wd,SH%L)[0],dtype=np.float32); b=np.asarray(get(O,wo,SH%L)[0],dtype=np.float32)
    X=a-b; Y=X-np.outer(rgE,rgE@X)
    U,s=rsvd(Y,k=2); S.append(U[:,0])
S=np.stack(S)
OUT='/Users/panda/Projects/open-source/ds4-glm53/analysis/glm53-abliteration/data/directions.npz'
np.savez_compressed(OUT,
    trunk_layers=np.array(T),
    A=z['uD'],            # Dealign attention o_proj direction per clean trunk layer
    S=S,                  # Dealign shared-expert down_proj direction per clean trunk layer
    rgA=z['uO'][0],       # Orca global attention direction
    rgE=rgE,              # Orca expert-family direction (held-out estimate)
    orca_per_layer=z['uO'],
    U44=z['U44'], s44=z['s44'], U45=z['U45'], s45=z['s45'],
    U44_orca=z['U44o'], U45_orca=z['U45o'])
print('wrote', OUT)
