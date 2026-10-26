"""Catastrophic forgetting on a REAL dataset (MNIST / Fashion-MNIST), single-pass stream.

Class-incremental setting: 5 tasks of 2 classes each, one shared 10-way output head.
Each task arrives as a stream that is seen ONCE (batch size 32, no epochs), so data is
transient. Methods compared:
  naive       : fine-tune on the stream, no memory (expected: catastrophic forgetting)
  replay-N    : reservoir-sampling buffer of N stored samples, mixed into every step
  joint       : i.i.d. stream over all data in one pass (rough upper bound)
Metrics: average final accuracy over all 5 tasks, and average forgetting.
Usage:  python continual.py --dataset mnist --seeds 5
"""
import argparse, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
BATCH = 32

def load(name, root="data"):
    from torchvision import datasets
    cls = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}[name]
    tr, te = cls(root, train=True, download=True), cls(root, train=False, download=True)
    def prep(d):
        return ((d.data.float() / 255 - 0.5) / 0.5).unsqueeze(1), d.targets
    return prep(tr), prep(te)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1)
        self.f1, self.f2 = nn.Linear(32 * 7 * 7, 128), nn.Linear(128, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.f2(F.relu(self.f1(x.flatten(1))))

class Reservoir:
    """Reservoir sampling: every sample seen so far has equal probability of being stored."""
    def __init__(self, cap, seed):
        self.cap, self.x, self.y, self.n, self.rng = cap, [], [], 0, random.Random(seed)
    def add(self, xb, yb):
        for xi, yi in zip(xb, yb):
            self.n += 1
            if len(self.x) < self.cap:
                self.x.append(xi); self.y.append(yi)
            else:
                j = self.rng.randrange(self.n)
                if j < self.cap:
                    self.x[j], self.y[j] = xi, yi
    def sample(self, k):
        idx = self.rng.sample(range(len(self.x)), min(k, len(self.x)))
        return torch.stack([self.x[i] for i in idx]), torch.stack([self.y[i] for i in idx])

def metrics(A):
    """A[t][j] = accuracy on task j after training on task t. Returns (avg final acc, forgetting)."""
    A, T = np.asarray(A), len(A)
    final = A[T - 1].mean()
    forget = np.mean([A[j:T - 1, j].max() - A[T - 1, j] for j in range(T - 1)])
    return final, forget

@torch.no_grad()
def evaluate(model, test):
    model.eval(); X, Y = test; accs = []
    for a, b in TASKS:
        m = (Y == a) | (Y == b)
        accs.append((model(X[m]).argmax(1) == Y[m]).float().mean().item())
    model.train(); return accs

def run(method, buf_size, seed, train, test):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model, X, Y = Net(), *train
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    buf = Reservoir(buf_size, seed) if method == "replay" else None
    def step(xb, yb):
        if buf is not None and buf.x:
            xr, yr = buf.sample(BATCH); xb2, yb2 = torch.cat([xb, xr]), torch.cat([yb, yr])
        else:
            xb2, yb2 = xb, yb
        opt.zero_grad(); F.cross_entropy(model(xb2), yb2).backward(); opt.step()
        if buf is not None: buf.add(xb, yb)
    if method == "joint":
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]; step(X[idx], Y[idx])
        a = evaluate(model, test); return [a] * len(TASKS)   # same model after the single stream
    A = []
    for a, b in TASKS:
        idx = torch.nonzero((Y == a) | (Y == b)).squeeze(1)
        idx = idx[torch.randperm(len(idx))]
        for i in range(0, len(idx), BATCH):                  # single pass over the task stream
            j = idx[i:i + BATCH]; step(X[j], Y[j])
        A.append(evaluate(model, test))
    return A

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"])
    p.add_argument("--seeds", type=int, default=5)
    a = p.parse_args()
    train, test = load(a.dataset)
    configs = [("naive", 0), ("replay", 200), ("replay", 1000), ("joint", 0)]
    rows = []
    for method, n in configs:
        for s in range(a.seeds):
            t0 = time.time(); A = run(method, n, s, train, test)
            fin, fgt = metrics(A)
            rows.append(dict(dataset=a.dataset, method=f"{method}-{n}" if n else method, seed=s,
                             final_acc=fin, forgetting=fgt))
            print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"results_{a.dataset}.csv", index=False)
    print("\n", df.groupby("method")[["final_acc", "forgetting"]].agg(["mean", "std"]).round(3))
