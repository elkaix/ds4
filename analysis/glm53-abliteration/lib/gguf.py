import struct, numpy as np

def _rd(f, fmt):
    n = struct.calcsize(fmt); return struct.unpack(fmt, f.read(n))[0]
def _rstr(f):
    n = _rd(f,'<Q'); return f.read(n).decode('utf-8','replace')

_VT = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
def _skipval(f, t):
    if t == 8: _rstr(f)
    elif t == 9:
        et = _rd(f,'<I'); n = _rd(f,'<Q')
        for _ in range(n): _skipval(f, et)
    else: f.read(struct.calcsize(_VT[t]))

def read_dir(path):
    """Return {name: (shape_ne, ggml_type, abs_offset)} plus alignment."""
    f = open(path,'rb')
    assert f.read(4) == b'GGUF'
    ver = _rd(f,'<I'); ntensor = _rd(f,'<Q'); nkv = _rd(f,'<Q')
    align = 32
    for _ in range(nkv):
        k = _rstr(f); t = _rd(f,'<I')
        if k == 'general.alignment' and t == 4:
            align = _rd(f,'<I')
        else:
            _skipval(f, t)
    infos = {}
    for _ in range(ntensor):
        name = _rstr(f); nd = _rd(f,'<I')
        ne = [_rd(f,'<Q') for _ in range(nd)]
        tt = _rd(f,'<I'); off = _rd(f,'<Q')
        infos[name] = (ne, tt, off)
    pos = f.tell()
    data_start = (pos + align - 1) // align * align
    f.close()
    return infos, data_start

def dequant_q8_0(buf, n):
    """buf: raw bytes of Q8_0 blocks; n: element count."""
    nb = n // 32
    a = np.frombuffer(buf, dtype=np.uint8, count=nb*34).reshape(nb, 34)
    d = a[:, :2].copy().view(np.float16).astype(np.float32)      # (nb,1)
    q = a[:, 2:].view(np.int8).astype(np.float32)                # (nb,32)
    return (q * d).reshape(-1)[:n]

def load(path, name, infos, data_start, out_shape):
    ne, tt, off = infos[name]
    n = int(np.prod(ne))
    if tt == 8:      # Q8_0
        nbytes = (n // 32) * 34
        with open(path,'rb') as f:
            f.seek(data_start + off); buf = f.read(nbytes)
        v = dequant_q8_0(buf, n)
    elif tt == 30:   # BF16
        with open(path,'rb') as f:
            f.seek(data_start + off); buf = f.read(n*2)
        u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
        v = u.view(np.float32)
    elif tt == 0:    # F32
        with open(path,'rb') as f:
            f.seek(data_start + off); buf = f.read(n*4)
        v = np.frombuffer(buf, dtype=np.float32)
    else:
        raise SystemExit(f'{name}: unhandled ggml type {tt}')
    return np.asarray(v, dtype=np.float32).reshape(out_shape)
