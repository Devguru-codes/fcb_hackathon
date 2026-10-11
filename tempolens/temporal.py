"""Bidirectional GRU over per-frame features: possession probabilities plus a change detector.

A frame classifier smears short possessions: its window features average over them. A recurrent
model reads the whole sequence around a frame, so it can say "this 1.5 s stretch belongs to the
other side" sharply, and its second head says where a possession changes, which the decoder uses
to make changes cheap exactly there.

Output per frame: log P(H), log P(A), log P(D), and the change logit.
Deterministic: seeded, and inference runs on the CPU unless asked otherwise.
"""

import numpy as np

from .features import TableSample, swap_labels
from .match import A, D, H, X

SEED = 0
CHUNK = 750          # 30 s of frames per training sequence
STRIDE = 375         # training chunks overlap by half
HIDDEN = 96
EPOCHS = 8            # ten matches x two noise draws: far more steps per epoch than one clean match
BATCH = 32
LR = 2e-3
CHANGE_DILATE = 2    # a change counts as "here" within +/-2 frames
CHANGE_WEIGHT = 0.5  # weight of the change loss beside the possession loss


def _torch():
    import torch
    return torch


class Normaliser:
    """Robust per-feature scaling fitted on training frames; NaN -> 0 plus a missing flag."""

    def __init__(self, x):
        self.median = np.nanmedian(x, axis=0)
        q75, q25 = np.nanpercentile(x, [75, 25], axis=0)
        self.scale = np.where(np.isfinite(q75 - q25) & (q75 - q25 > 1e-6), q75 - q25, 1.0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        self.flagged = np.flatnonzero(np.isnan(x).any(axis=0))

    def __call__(self, x):
        z = np.clip((x - self.median) / self.scale, -10, 10)
        flags = np.isnan(x[:, self.flagged]).astype(np.float32)
        return np.concatenate([np.nan_to_num(z, nan=0.0), flags], axis=1).astype(np.float32)

    def state(self):
        return {"median": self.median, "scale": self.scale, "flagged": self.flagged}

    @classmethod
    def from_state(cls, state):
        self = cls.__new__(cls)
        self.median, self.scale = state["median"], state["scale"]
        self.flagged = state["flagged"]
        return self


def change_targets(labels, dilate=CHANGE_DILATE):
    """1 where the label changes within +/-dilate frames, 0 elsewhere, -1 where unknown."""
    change = np.zeros(len(labels), dtype=np.float32)
    flip = np.flatnonzero((labels[1:] != labels[:-1]) & (labels[1:] != X) & (labels[:-1] != X))
    for k in range(-dilate, dilate + 1):
        idx = np.clip(flip + 1 + k, 0, len(labels) - 1)
        change[idx] = 1.0
    change[labels == X] = -1.0
    return change


def build_model(n_inputs, hidden=HIDDEN):
    torch = _torch()
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_inputs, hidden), nn.GELU(), nn.Dropout(0.2))
            self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=0.2)
            self.head = nn.Linear(2 * hidden, 4)   # H, A, D logits + change logit

        def forward(self, x):
            out, _ = self.rnn(self.inp(x))
            return self.head(out)

    return Net()


def _as_sample(item):
    return item if hasattr(item, "features") else TableSample(item)


def _sequences(samples, normaliser=None):
    """Per sample, perspective and period: (normalised float16 features, labels, change targets).

    float16 halves the memory of ten simulated matches x two noise draws x two perspectives;
    batches are cast back to float32 on the device.
    """
    samples = [_as_sample(s) for s in samples]
    if normaliser is None:
        stacked = np.concatenate([np.asarray(s.features(False))[s.period > 0][::10]
                                  for s in samples])
        normaliser = Normaliser(stacked)
    out = []
    for sample in samples:
        for swap in (False, True):
            x = sample.features(swap)
            y = swap_labels(sample.labels) if swap else sample.labels
            for start, stop in sample.runs():
                y_run = y[start:stop].astype(np.int64)
                change = change_targets(y_run)
                y_run[y_run == X] = -100   # CrossEntropyLoss ignore_index
                out.append((normaliser(np.asarray(x[start:stop])).astype(np.float16), y_run,
                            change))
    return out, normaliser


def _chunk_index(sequences, chunk=CHUNK, stride=STRIDE):
    """[(sequence, start)] of overlapping training windows, covering each half's tail too."""
    index = []
    for k, (x, _, _) in enumerate(sequences):
        if len(x) < chunk:
            continue
        starts = list(range(0, len(x) - chunk + 1, stride))
        if starts[-1] != len(x) - chunk:
            starts.append(len(x) - chunk)
        index.extend((k, start) for start in starts)
    return index


def train(samples, epochs=EPOCHS, device=None, seed=SEED, verbose=True):
    """Fit the network on labelled samples (or MatchTables). Returns (CPU model, normaliser)."""
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sequences, normaliser = _sequences(samples)
    index = _chunk_index(sequences)
    model = build_model(sequences[0][0].shape[1]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(index) / BATCH))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=LR, total_steps=steps)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none",
                                     pos_weight=torch.tensor(4.0, device=device))
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(index))
        total = 0.0
        for i in range(0, len(order), BATCH):
            batch = [index[j] for j in order[i:i + BATCH]]
            x = torch.from_numpy(np.stack([sequences[k][0][s:s + CHUNK] for k, s in batch]))
            y = torch.from_numpy(np.stack([sequences[k][1][s:s + CHUNK] for k, s in batch]))
            c = torch.from_numpy(np.stack([sequences[k][2][s:s + CHUNK] for k, s in batch]))
            x, y, c = x.to(device).float(), y.to(device), c.to(device)
            out = model(x)
            loss = ce(out[..., :3].reshape(-1, 3), y.reshape(-1))
            valid = c >= 0
            change_loss = ((bce(out[..., 3], c.clamp(min=0)) * valid).sum()
                           / valid.sum().clamp(min=1))
            loss = loss + CHANGE_WEIGHT * change_loss
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            total += float(loss) * len(batch)
        if verbose:
            print("  epoch {:>2}  loss {:.4f}".format(epoch + 1, total / len(index)), flush=True)
    return model.cpu().eval(), normaliser


def _infer(model, z, chunk=3000, overlap=500):
    """Run over a long sequence in overlapping windows, keeping each window's centre."""
    torch = _torch()
    n = len(z)
    out = np.zeros((n, 4), dtype=np.float64)
    step = chunk - 2 * overlap
    with torch.no_grad():
        for start in range(0, n, step):
            lo, hi = max(0, start - overlap), min(n, start + step + overlap)
            pred = model(torch.from_numpy(z[None, lo:hi]))[0].numpy()
            keep_lo, keep_hi = start - lo, min(start + step, n) - lo
            out[start:start + (keep_hi - keep_lo)] = pred[keep_lo:keep_hi]
    return out


def predict(model, normaliser, sample):
    """(n, 4): log P(H), log P(A), log P(D), change logit; averaged over both perspectives."""
    torch = _torch()
    torch.set_num_threads(4)
    sample = _as_sample(sample)
    result = np.zeros((sample.n_frames, 4), dtype=np.float64)
    result[:, :3] = np.log(1.0 / 3)
    for swap in (False, True):
        x = sample.features(swap)
        for start, stop in sample.runs():
            raw = _infer(model, normaliser(np.asarray(x[start:stop])))
            p = np.exp(raw[:, :3] - raw[:, :3].max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            if swap:
                p = p[:, [A, H, D]]
            if not swap:
                result[start:stop, :3] = 0.5 * p
                result[start:stop, 3] = 0.5 * raw[:, 3]
            else:
                result[start:stop, :3] += 0.5 * p
                result[start:stop, 3] += 0.5 * raw[:, 3]
    inside = sample.period > 0
    result[inside, :3] = np.log(np.clip(result[inside, :3], 1e-6, 1.0))
    # Float32 results can differ in the last bits between CPUs; rounding keeps such differences
    # from ever flipping a decoder decision, so the organisers' re-run gives the same file.
    return np.round(result, 4)


def train_and_predict(train_samples, test_sample):
    model, normaliser = train(train_samples)
    return predict(model, normaliser, test_sample)


def save(model, normaliser, path):
    torch = _torch()
    torch.save({"state_dict": model.state_dict(), "normaliser": normaliser.state(),
                "n_inputs": model.inp[0].in_features}, path)


def load(path):
    torch = _torch()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(blob["n_inputs"])
    model.load_state_dict(blob["state_dict"])
    return model.eval(), Normaliser.from_state(blob["normaliser"])
