import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'lib'))
import sys, numpy as np
from glmlib import get, D, O, wd, wo
from gguf import read_dir, load
GG='/Users/panda/models/gguf/GLM-5.3-Flash-Q2.gguf'
infos, ds = read_dir(GG); rng=np.random.default_rng(0)

def rsvd(X,k=8,p=10,it=4):
    """left singular vectors (rows-space, R^4096) + singular values"""
    Om=rng.standard_normal((X.shape[1],k+p)).astype(np.float32)
    Q,_=np.linalg.qr(X@Om)
    for _ in range(it):
        Q,_=np.linalg.qr(X.T@Q); Q,_=np.linalg.qr(X@Q)
    B=Q.T@X
    Ub,s,_=np.linalg.svd(B,full_matrices=False)
    U=Q@Ub
    return U[:,:k], s[:k]

K='model.language_model.layers.%d.self_attn.o_proj.weight'
TRUNK=[l for l in range(12,43) if l%4!=3]
def delta(L, gn):
    k=K%L
    a=np.asarray(get(D,wd,k)[0],dtype=np.float32); b=np.asarray(get(O,wo,k)[0],dtype=np.float32)
    nm=f'blk.{L}.{"kda_output" if gn=="kda" else "attn_output"}.weight'
    s=load(GG,nm,infos,ds,a.shape)
    return a-s, b-s

uD={}; uO={}; sD={}; energy={}
for L in TRUNK:
    Dl,Ol=delta(L,'kda')
    U,sv=rsvd(Dl,k=8); uD[L]=U[:,0].copy(); sD[L]=sv.copy()
    Uo,so=rsvd(Ol,k=2); uO[L]=Uo[:,0].copy()
    energy[L]=(float(np.linalg.norm(Dl)),float(np.linalg.norm(Ol)))
    print('trunk',L,'done',flush=True)

# L44 (kda) full subspace + MTP L45 (attn)
D44,O44=delta(44,'kda')
U44,s44=rsvd(D44,k=8)
U44o,_=rsvd(O44,k=2)
D45,O45=delta(45,'attn')
U45,s45=rsvd(D45,k=8)
U45o,_=rsvd(O45,k=2)
print('L44/L45 done',flush=True)

np.savez(_os.path.join(_ROOT,'data')+'/lvec.npz',
  trunk=np.array(TRUNK),
  uD=np.stack([uD[l] for l in TRUNK]), uO=np.stack([uO[l] for l in TRUNK]),
  sD=np.stack([sD[l] for l in TRUNK]),
  U44=U44, s44=s44, U44o=U44o, U45=U45, s45=s45, U45o=U45o,
  nD44=np.linalg.norm(D44), nD45=np.linalg.norm(D45),
  energy=np.array([energy[l] for l in TRUNK]))

# --- reports ---
def gram(V):
    C=np.abs(V@V.T); return C
GD=gram(np.stack([uD[l] for l in TRUNK]))
GO=gram(np.stack([uO[l] for l in TRUNK]))
print('\n== Dealign |u_i.u_j| trunk ==')
print('     '+' '.join(f'{l:>5}' for l in TRUNK))
for i,l in enumerate(TRUNK):
    print(f'{l:>4} '+' '.join(f'{GD[i,j]:>5.2f}' for j in range(len(TRUNK))))
print('\n== Orca |u_i.u_j| trunk ==  min=%.4f mean=%.4f'%(GO[np.triu_indices(len(TRUNK),1)].min(),GO[np.triu_indices(len(TRUNK),1)].mean()))
print('     '+' '.join(f'{l:>5}' for l in TRUNK))
for i,l in enumerate(TRUNK):
    print(f'{l:>4} '+' '.join(f'{GO[i,j]:>5.2f}' for j in range(len(TRUNK))))
off=GD[np.triu_indices(len(TRUNK),1)]
print('\nDealign off-diag |cos|: min %.4f  med %.4f  max %.4f'%(off.min(),np.median(off),off.max()))

# trunk representative direction (sign-aligned mean)
ref=uD[TRUNK[0]]
Vs=np.stack([uD[l]*np.sign(uD[l]@ref) for l in TRUNK])
uT=Vs.mean(0); uT/=np.linalg.norm(uT)
print('trunk mean-direction |u_T.u_l|:', ' '.join(f'{l}:{abs(uT@uD[l]):.3f}' for l in TRUNK))
refO=uO[TRUNK[0]]
VsO=np.stack([uO[l]*np.sign(uO[l]@refO) for l in TRUNK])
uTO=VsO.mean(0); uTO/=np.linalg.norm(uTO)

print('\n== L44 ==')
for k in (1,2,4,8):
    print(f'  P_44<-T (k={k}) = {float(np.sum((U44[:,:k].T@uT)**2)):.4f}')
FT=float(np.linalg.norm(uT@D44)**2/np.linalg.norm(D44)**2)
print(f'  F_T (energy of D44 along u_T) = {FT:.4f}')
print(f'  |u_44,1 . u_T| = {abs(U44[:,0]@uT):.4f}')
print(f'  Orca L44: |u_44^O . u_T^O| = {abs(U44o[:,0]@uTO):.4f}')
print(f'  s44 (norm) = {" ".join(f"{x/s44[0]:.3f}" for x in s44)}')

print('\n== MTP L45 ==')
uM=U45[:,0]
print(f'  c_MT = |u_MTP . u_T|      = {abs(uM@uT):.4f}')
print(f'  c_M44= |u_MTP . u_44,1|   = {abs(uM@U44[:,0]):.4f}')
for k in (1,2,4,8):
    print(f'  P_M->44 (k={k}) = {float(np.sum((U44[:,:k].T@uM)**2)):.4f}')
FTM=float(np.linalg.norm(uT@D45)**2/np.linalg.norm(D45)**2)
print(f'  F_T on D45 = {FTM:.4f}')
print(f'  s45 (norm) = {" ".join(f"{x/s45[0]:.3f}" for x in s45)}')
print(f'  Orca MTP: |u_45^O . u_T^O| = {abs(U45o[:,0]@uTO):.4f}')
