"""Minimal NumPy deep-learning layers (NHWC, float32) with explicit backward passes.
Used because the compute environment of this revision had no deep-learning framework."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

F32 = np.float32


class Param:
    def __init__(self, data, kind, fan_in=None, fan_out=None):
        self.data = data.astype(F32); self.grad = np.zeros_like(self.data)
        self.kind = kind            # 'weight' (matrix/conv kernel), 'bias', 'bn'
        self.fan_in, self.fan_out = fan_in, fan_out


class Linear:
    def __init__(self, din, dout, rng, init="he"):
        std = np.sqrt(2.0 / din) if init == "he" else np.sqrt(2.0 / (din + dout))
        if init == "xavier_uniform":
            a = np.sqrt(6.0 / (din + dout)); w = rng.uniform(-a, a, (din, dout))
        else:
            w = rng.normal(0, std, (din, dout))
        self.W = Param(w, "weight", din, dout); self.b = Param(np.zeros(dout), "bias")
        self.params = [self.W, self.b]

    def fwd(self, x, train=True):
        self.x = x; return x @ self.W.data + self.b.data

    def bwd(self, d):
        self.W.grad += self.x.T @ d; self.b.grad += d.sum(0); return d @ self.W.data.T


class Conv:
    """3x3 (or kxk) convolution, stride 1 or 2, 'same' padding, NHWC."""

    def __init__(self, cin, cout, rng, k=3, stride=1, bias=True, need_dx=True):
        self.k, self.s, self.cin, self.cout, self.need_dx = k, stride, cin, cout, need_dx
        w = rng.normal(0, np.sqrt(2.0 / (cin * k * k)), (cin * k * k, cout))  # He (fan_in)
        self.W = Param(w, "weight", cin * k * k, cout * k * k)
        self.params = [self.W]
        self.has_b = bias
        if bias:
            self.b = Param(np.zeros(cout), "bias"); self.params.append(self.b)

    def fwd(self, x, train=True):
        N, H, W_, C = x.shape; k, s, p = self.k, self.s, self.k // 2
        xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
        win = sliding_window_view(xp, (k, k), axis=(1, 2))[:, ::s, ::s]  # N,Ho,Wo,C,k,k
        Ho, Wo = win.shape[1], win.shape[2]
        cols = np.ascontiguousarray(win.transpose(0, 1, 2, 4, 5, 3)).reshape(N * Ho * Wo, k * k * C)
        self.cache = (x.shape, xp.shape, Ho, Wo, cols)
        out = cols @ self.W.data
        if self.has_b: out += self.b.data
        return out.reshape(N, Ho, Wo, self.cout)

    def bwd(self, d):
        xshape, xpshape, Ho, Wo, cols = self.cache
        N, H, W_, C = xshape; k, s, p = self.k, self.s, self.k // 2
        d2 = d.reshape(-1, self.cout)
        self.W.grad += cols.T @ d2
        if self.has_b: self.b.grad += d2.sum(0)
        if not self.need_dx:
            return None
        dcols = (d2 @ self.W.data.T).reshape(N, Ho, Wo, k, k, C)
        dxp = np.zeros(xpshape, F32)
        for i in range(k):
            for j in range(k):
                dxp[:, i:i + s * Ho:s, j:j + s * Wo:s, :] += dcols[:, :, :, i, j, :]
        return dxp[:, p:p + H, p:p + W_, :]


class BN:
    def __init__(self, c, mom=0.1, eps=1e-5):
        self.g = Param(np.ones(c), "bn"); self.b = Param(np.zeros(c), "bn")
        self.params = [self.g, self.b]; self.rm = np.zeros(c, F32); self.rv = np.ones(c, F32)
        self.mom, self.eps = mom, eps

    def fwd(self, x, train=True):
        ax = tuple(range(x.ndim - 1))
        if train:
            mu = x.mean(ax); var = x.var(ax)
            n = x.size // x.shape[-1]
            self.rm = (1 - self.mom) * self.rm + self.mom * mu
            self.rv = (1 - self.mom) * self.rv + self.mom * var * n / max(n - 1, 1)
        else:
            mu, var = self.rm, self.rv
        self.inv = 1.0 / np.sqrt(var + self.eps); self.xh = (x - mu) * self.inv
        return self.g.data * self.xh + self.b.data

    def bwd(self, d):
        ax = tuple(range(d.ndim - 1)); n = d.size // d.shape[-1]
        self.g.grad += (d * self.xh).sum(ax); self.b.grad += d.sum(ax)
        dxh = d * self.g.data
        return self.inv / n * (n * dxh - dxh.sum(ax) - self.xh * (dxh * self.xh).sum(ax))


class ReLU:
    params = []
    def fwd(self, x, train=True):
        self.m = x > 0; return x * self.m
    def bwd(self, d): return d * self.m


class MaxPool2:
    params = []
    def fwd(self, x, train=True):
        N, H, W, C = x.shape
        r = x.reshape(N, H // 2, 2, W // 2, 2, C); out = r.max((2, 4))
        self.mask = (r == out[:, :, None, :, None, :]); self.shape = x.shape
        return out
    def bwd(self, d):
        dd = self.mask * d[:, :, None, :, None, :]
        dd = dd / np.maximum(self.mask.sum((2, 4), keepdims=True), 1).astype(F32)
        return dd.astype(F32, copy=False).reshape(self.shape)


class Flatten:
    params = []
    def fwd(self, x, train=True):
        self.shape = x.shape; return x.reshape(x.shape[0], -1)
    def bwd(self, d): return d.reshape(self.shape)


class GAP:
    params = []
    def fwd(self, x, train=True):
        self.shape = x.shape; return x.mean((1, 2))
    def bwd(self, d):
        N, H, W, C = self.shape; return np.broadcast_to(d[:, None, None, :] / (H * W), self.shape).astype(F32)


class Seq:
    def __init__(self, layers):
        self.layers = layers; self.params = [p for l in layers for p in l.params]
    def fwd(self, x, train=True):
        for l in self.layers: x = l.fwd(x, train)
        return x
    def bwd(self, d):
        for l in reversed(self.layers): d = l.bwd(d)
        return d


class Block:
    """ResNet basic block: conv-BN-ReLU-conv-BN + shortcut (1x1 conv-BN when shape changes), ReLU."""
    def __init__(self, cin, cout, stride, rng):
        self.main = Seq([Conv(cin, cout, rng, 3, stride, bias=False), BN(cout), ReLU(),
                         Conv(cout, cout, rng, 3, 1, bias=False), BN(cout)])
        self.short = Seq([Conv(cin, cout, rng, 1, stride, bias=False), BN(cout)]) if (stride != 1 or cin != cout) else None
        self.relu = ReLU()
        self.params = self.main.params + (self.short.params if self.short else [])
    def fwd(self, x, train=True):
        y = self.main.fwd(x, train) + (self.short.fwd(x, train) if self.short else x)
        return self.relu.fwd(y, train)
    def bwd(self, d):
        d = self.relu.bwd(d)
        return self.main.bwd(d) + (self.short.bwd(d) if self.short else d)


def mlp(rng):
    return Seq([Linear(784, 128, rng, "xavier_uniform"), ReLU(), Linear(128, 64, rng, "xavier_uniform"), ReLU(),
                Linear(64, 32, rng, "xavier_uniform"), ReLU(), Linear(32, 10, rng, "xavier_uniform")])


def small_cnn(rng):
    return Seq([Conv(3, 32, rng, need_dx=False), ReLU(), MaxPool2(), Conv(32, 64, rng), ReLU(), MaxPool2(), Flatten(),
                Linear(64 * 8 * 8, 128, rng), ReLU(), Linear(128, 10, rng)])


def resnet(rng, widths=(16, 32, 64), blocks=(1, 1, 1)):
    """CIFAR-style ResNet (He et al., 2016): ResNet-8 for blocks=(1,1,1), ResNet-20 for (3,3,3)."""
    L = [Conv(3, widths[0], rng, 3, 1, bias=False, need_dx=False), BN(widths[0]), ReLU()]
    cin = widths[0]
    for si, (w, nb) in enumerate(zip(widths, blocks)):
        for bi in range(nb):
            L.append(Block(cin, w, 2 if (si > 0 and bi == 0) else 1, rng)); cin = w
    L += [GAP(), Linear(cin, 10, rng)]
    return Seq(L)


def softmax_ce(logits, y):
    z = logits - logits.max(1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True)
    n = len(y); loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    d = p; d[np.arange(n), y] -= 1; return float(loss), (d / n).astype(F32)
