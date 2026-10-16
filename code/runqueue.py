import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import itertools, os, subprocess, sys, threading, queue, time
R = _os.path.join(ROOT, "results")
OPTS = ["adam", "nadam", "aadam", "adamw", "radam", "adabelief", "adam_warmup", "adam_v0rnd", "sadam", "sadamw"]
G = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
cmds = []
def add(sub, ds, model, opt, seed, ep, bs, sm=0.01, sv=0.01, aug=0, tag=""):
    cmds.append([sys.executable, "run.py", "--dataset", ds, "--model", model, "--opt", opt, "--seed", str(seed), "--epochs", str(ep),
                 "--bs", str(bs), "--sigma_m", str(sm), "--sigma_v", str(sv), "--augment", str(aug), "--tag", tag, "--out", f"{R}/{sub}"])
for s in range(10):
    for o in OPTS: add("E1", "mnist", "mlp", o, s, 30, 256)
for sm, sv in itertools.product(G, G):
    for s in range(5): add("E5", "mnist", "mlp", "sadam", s, 30, 256, sm, sv, tag="sadam")
for s in range(3):
    for o in OPTS: add("E2", "cifar10", "smallcnn", o, s, 30, 256)
for s in (3, 4):  # two extra paired seeds for Adam and SADAM (paired statistics use 5 pairs)
    for o in ["adam", "sadam"]: add("E2", "cifar10", "smallcnn", o, s, 30, 256)
for sm, sv in [(1e-4, 1e-4), (1e-3, 1e-3), (1e-1, 1e-1), (0.0, 1e-2), (1e-2, 0.0)]:
    for s in range(2): add("E5", "cifar10", "smallcnn", "sadam", s, 15, 256, sm, sv, tag="sadam")
for s in range(3):
    for o in ["adam", "adamw", "radam", "adabelief", "adam_warmup", "sadam"]: add("E3", "cifar10", "resnet8", o, s, 10, 128, aug=1)
print(len(cmds), "runs", flush=True)
q = queue.Queue(); [q.put(c) for c in cmds]
def worker(w):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    while True:
        try: c = q.get_nowait()
        except queue.Empty: return
        t = time.time(); r = subprocess.run(c, env=env, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        last = (r.stdout.strip().splitlines() or [""])[-1]
        print(f"[w{w}] {time.time()-t:.0f}s rc={r.returncode} {' '.join(c[3:13])} | {last} {r.stderr[-300:] if r.returncode else ''}", flush=True)
ths = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
[t.start() for t in ths]; [t.join() for t in ths]
print("ALL DONE", flush=True)
# afterwards: python gradrms.py ; python npanalyze.py ; python replot.py
