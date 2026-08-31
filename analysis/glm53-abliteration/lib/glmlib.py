import json, os, re, struct, sys
import numpy as np

D = '/Users/panda/models/hf/GLM-5.3-Flash-ABLITERATED-FP8'   # dealignai
O = '/Users/panda/models/hf/GLM-5.3-Flash-Uncensored-FP8'    # orcarouter

# ---- OCP E4M3 (e4m3fn) decode LUT ----
def e4m3_lut():
    out = np.zeros(256, dtype=np.float32)
    for i in range(256):
        s = -1.0 if (i >> 7) else 1.0
        e = (i >> 3) & 0xF
        m = i & 0x7
        if e == 0:
            v = (m / 8.0) * (2.0 ** -6)
        elif e == 0xF and m == 0x7:
            v = np.nan
        else:
            v = (1.0 + m / 8.0) * (2.0 ** (e - 7))
        out[i] = s * v
    return out
LUT = e4m3_lut()

def wmap(d):
    return json.load(open(os.path.join(d,'model.safetensors.index.json')))['weight_map']
wd, wo = wmap(D), wmap(O)

_hdr = {}
def header(path):
    h = _hdr.get(path)
    if h is None:
        with open(path,'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            h = (json.loads(f.read(n)), 8+n)
        _hdr[path] = h
    return h

def raw(d, wm, k):
    path = os.path.join(d, wm[k]); meta, base = header(path); e = meta[k]
    start, end = e['data_offsets']; shape = tuple(e['shape']); dt = e['dtype']
    if dt == 'BF16':
        a = np.memmap(path, dtype=np.uint16, mode='r', offset=base+start, shape=(int(np.prod(shape)),))
        return (a.astype(np.uint32) << 16).view(np.float32).reshape(shape), dt
    if dt == 'F8_E4M3':
        a = np.memmap(path, dtype=np.uint8, mode='r', offset=base+start, shape=shape)
        return LUT[np.asarray(a)], dt
    if dt == 'F32':
        return np.asarray(np.memmap(path, dtype=np.float32, mode='r', offset=base+start, shape=shape)), dt
    raise SystemExit(f'unhandled {dt} {k}')

def get(d, wm, k):
    """float32 weights, dequantized if block-FP8."""
    w, dt = raw(d, wm, k)
    sk = k + '_scale_inv'
    if dt == 'F8_E4M3' and sk in wm:
        s, _ = raw(d, wm, sk)
        r, c = w.shape
        br = int(np.ceil(r / s.shape[0])); bc = int(np.ceil(c / s.shape[1]))
        # standard DeepSeek/GLM block-FP8: 128x128 blocks
        full = np.repeat(np.repeat(s, 128, axis=0), 128, axis=1)[:r, :c]
        w = w * full
        dt = 'F8+s'
    return np.nan_to_num(w.astype(np.float32)), dt

def layer_of(k):
    m = re.search(r'layers\.(\d+)\.', k); return int(m.group(1)) if m else -1

