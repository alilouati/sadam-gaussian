import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, os, gzip, pickle, glob
from PIL import Image
D = _os.environ.get("SADAM_DATA", _os.path.join(ROOT, "data"))
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def mnist():
    c = f"{D}/mnist.npz"
    if not os.path.exists(c):
        tr, va, te = pickle.load(gzip.open(f"{D}/nndl/data/mnist.pkl.gz"), encoding="latin1")
        np.savez(c, xtr=np.concatenate([tr[0], va[0]]), ytr=np.concatenate([tr[1], va[1]]), xte=te[0], yte=te[1])
    z = np.load(c); f = lambda x: ((x - 0.1307) / 0.3081).astype(np.float32)
    return f(z["xtr"]), z["ytr"].astype(np.int64), f(z["xte"]), z["yte"].astype(np.int64)

def cifar():
    c = f"{D}/cifar.npz"
    if not os.path.exists(c):
        out = {}
        for split in ("train", "test"):
            X, Y = [], []
            for k, cl in enumerate(CLASSES):
                for fn in sorted(glob.glob(f"{D}/cifar_imgs/{split}/{cl}/*.jpg")):
                    X.append(np.asarray(Image.open(fn).convert("RGB"))); Y.append(k)
            out["x" + split[:2]] = np.stack(X); out["y" + split[:2]] = np.array(Y)
        np.savez(c, **out)
    z = np.load(c)
    mean = np.array([0.4914, 0.4822, 0.4465], np.float32); std = np.array([0.2470, 0.2435, 0.2616], np.float32)
    f = lambda x: ((x.astype(np.float32) / 255 - mean) / std).astype(np.float32)
    return f(z["xtr"]), z["ytr"].astype(np.int64), f(z["xte"]), z["yte"].astype(np.int64)
