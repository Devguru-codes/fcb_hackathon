"""Sequence models over per-frame features: possession probabilities plus a change detector.

A frame classifier smears short possessions: its window features average over them. A recurrent
model reads the whole sequence around a frame, so it can say "this 1.5 s stretch belongs to the
other side" sharply, and its second head says where a possession changes, which the decoder uses
to make changes cheap exactly there.

Output per frame: log P(H), log P(A), log P(D), and the change logit.
Deterministic: seeded, and inference runs on the CPU unless asked otherwise.

Variants, chosen by TemporalConfig and saved inside each weights file:
- arch: "gru" (bidirectional GRU), "tcn" (dilated temporal convolutions, ~40 s receptive field)
  or "transformer" (self-attention over 5 fps tokens with a full-rate skip path);
- encoder: "features" (the hand-made per-frame features) or "players" (those plus a per-frame
  attention encoder over the set of players -- a fully connected graph network, slot-free);
- aux_ball: weight of an auxiliary head that regresses the true ball position on frames where
  the simulator hid it, i.e. learning to place an unseen ball from how the players move.
"""

from dataclasses import asdict, dataclass

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


@dataclass(frozen=True)
class TemporalConfig:
    arch: str = "gru"            # gru | tcn | transformer
    encoder: str = "features"    # features | players
    hidden: int = HIDDEN
    aux_ball: float = 0.0        # weight of the hidden-ball regression loss; 0 = no ball head

    def tag(self):
        """Cache/file tag; the default config keeps the historical name "temporal"."""
        parts = [part for part, default in ((self.arch, "gru"), (self.encoder, "features"))
                 if part != default]
        if self.hidden != HIDDEN:
            parts.append("h{}".format(self.hidden))
        if self.aux_ball:
            parts.append("aux{:g}".format(self.aux_ball))
        return "temporal" + ("_" + "_".join(parts) if parts else "")


DEFAULT_CONFIG = TemporalConfig()


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


def build_model(n_inputs, config=DEFAULT_CONFIG, n_token_fields=None):
    torch = _torch()
    nn = torch.nn
    hidden = config.hidden

    class PlayerSetEncoder(nn.Module):
        """Per frame: embed each player, one self-attention layer across all players (a fully
        connected graph), then masked mean and max pooling per team."""

        def __init__(self, fields, width=32):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(fields, width), nn.GELU(),
                                       nn.Linear(width, width))
            self.attn = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(width)
            self.out_features = 4 * width

        def forward(self, tokens):
            b, t, p, f = tokens.shape
            z = tokens.reshape(b * t, p, f)
            present = z[..., 0] > 0.5
            h = self.embed(z)
            # A frame with nobody visible would make attention divide by zero: let such rows
            # attend to everything (they are zeros anyway) and pool to zeros below.
            blocked = ~present & present.any(dim=1, keepdim=True)
            attended, _ = self.attn(h, h, h, key_padding_mask=blocked, need_weights=False)
            h = self.norm(h + attended) * present[..., None]
            half = p // 2
            pooled = []
            for team in (slice(0, half), slice(half, p)):
                ht, mt = h[:, team], present[:, team, None]
                count = mt.sum(dim=1).clamp(min=1)
                pooled.append(ht.sum(dim=1) / count)
                pooled.append(ht.masked_fill(~mt, -1e4).amax(dim=1).clamp(min=-1e3)
                              * (mt.sum(dim=1) > 0))
            return torch.cat(pooled, dim=-1).reshape(b, t, -1)

    class TCN(nn.Module):
        """Non-causal dilated convolutions: kernel 5, dilations to 128, ~2.5k-frame field."""

        def __init__(self, width, dilations=(1, 2, 4, 8, 16, 32, 64, 128)):
            super().__init__()
            self.blocks = nn.ModuleList(
                nn.Sequential(nn.Conv1d(width, width, 5, padding=2 * d, dilation=d), nn.GELU(),
                              nn.Dropout(0.1), nn.Conv1d(width, width, 1))
                for d in dilations)
            self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in dilations)

        def forward(self, x):
            for block, norm in zip(self.blocks, self.norms):
                x = norm(x + block(x.transpose(1, 2)).transpose(1, 2))
            return x

    class Transformer(nn.Module):
        """Self-attention over 5 fps tokens (stride-5 conv), upsampled back, plus the full-rate
        input as a skip path so boundaries stay frame-accurate."""

        def __init__(self, width, layers=3, stride=5):
            super().__init__()
            self.stride = stride
            self.down = nn.Conv1d(width, width, stride, stride=stride)
            layer = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.1,
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.mix = nn.Linear(2 * width, width)

        def forward(self, x):
            b, t, w = x.shape
            pad = (-t) % self.stride
            xp = nn.functional.pad(x, (0, 0, 0, pad))
            tokens = self.down(xp.transpose(1, 2)).transpose(1, 2)
            position = torch.arange(tokens.shape[1], device=x.device, dtype=x.dtype)[:, None]
            freq = torch.exp(torch.arange(0, w, 2, device=x.device, dtype=x.dtype)
                             * (-np.log(10000.0) / w))
            enc = torch.zeros(tokens.shape[1], w, device=x.device, dtype=x.dtype)
            enc[:, 0::2], enc[:, 1::2] = torch.sin(position * freq), torch.cos(position * freq)
            ctx = self.encoder(tokens + enc)
            up = ctx.repeat_interleave(self.stride, dim=1)[:, :t]
            return self.mix(torch.cat([x, up], dim=-1))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            width_in = n_inputs
            if config.encoder == "players":
                self.players = PlayerSetEncoder(n_token_fields)
                width_in += self.players.out_features
            self.inp = nn.Sequential(nn.Linear(width_in, hidden), nn.GELU(), nn.Dropout(0.2))
            if config.arch == "gru":
                self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                                  bidirectional=True, dropout=0.2)
                width_out = 2 * hidden
            elif config.arch == "tcn":
                self.tcn = TCN(hidden)
                width_out = hidden
            elif config.arch == "transformer":
                self.transformer = Transformer(hidden)
                width_out = hidden
            else:
                raise ValueError("unknown arch {!r}".format(config.arch))
            self.head = nn.Linear(width_out, 4)   # H, A, D logits + change logit
            if config.aux_ball:
                self.ball_head = nn.Linear(width_out, 2)

        def forward(self, x, tokens=None):
            if config.encoder == "players":
                x = torch.cat([x, self.players(tokens)], dim=-1)
            h = self.inp(x)
            if config.arch == "gru":
                h, _ = self.rnn(h)
            elif config.arch == "tcn":
                h = self.tcn(h)
            else:
                h = self.transformer(h)
            out = self.head(h)
            if config.aux_ball:
                out = torch.cat([out, self.ball_head(h)], dim=-1)
            return out

    net = Net()
    net.config = config
    return net


def _as_sample(item):
    return item if hasattr(item, "features") else TableSample(item)


def _sequences(samples, config=DEFAULT_CONFIG, normaliser=None):
    """Per sample, perspective and period, a dict of training arrays.

    x: normalised float16 features; y: labels (-100 = ignore); c: change targets; and, when the
    config asks for them, tok: player tokens (memory-mapped where the sample caches them) and
    ball: the hidden-ball target. float16 keeps ten simulated matches x noise draws x two
    perspectives in memory; batches are cast to float32 on the device.
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
            tokens = sample.tokens(swap) if config.encoder == "players" else None
            ball = sample.ball_target(swap) if config.aux_ball else None
            for start, stop in sample.runs():
                y_run = y[start:stop].astype(np.int64)
                change = change_targets(y_run)
                y_run[y_run == X] = -100   # CrossEntropyLoss ignore_index
                seq = {"x": normaliser(np.asarray(x[start:stop])).astype(np.float16),
                       "y": y_run, "c": change}
                if tokens is not None:
                    seq["tok"] = tokens[start:stop]
                if ball is not None:
                    seq["ball"] = np.asarray(ball[start:stop], dtype=np.float32)
                out.append(seq)
    return out, normaliser


def _chunk_index(sequences, chunk=CHUNK, stride=STRIDE):
    """[(sequence, start)] of overlapping training windows, covering each half's tail too."""
    index = []
    for k, seq in enumerate(sequences):
        n = len(seq["x"])
        if n < chunk:
            continue
        starts = list(range(0, n - chunk + 1, stride))
        if starts[-1] != n - chunk:
            starts.append(n - chunk)
        index.extend((k, start) for start in starts)
    return index


def train(samples, epochs=EPOCHS, device=None, seed=SEED, verbose=True,
          config=DEFAULT_CONFIG):
    """Fit a network on labelled samples (or MatchTables). Returns (CPU model, normaliser)."""
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sequences, normaliser = _sequences(samples, config)
    index = _chunk_index(sequences)
    n_fields = sequences[0]["tok"].shape[-1] if "tok" in sequences[0] else None
    model = build_model(sequences[0]["x"].shape[1], config, n_fields).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(index) / BATCH))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=LR, total_steps=steps)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none",
                                     pos_weight=torch.tensor(4.0, device=device))
    huber = torch.nn.SmoothL1Loss(reduction="none", beta=0.1)
    rng = np.random.default_rng(seed)

    def batch_of(key, batch, dtype=None):
        array = np.stack([np.asarray(sequences[k][key][s:s + CHUNK]) for k, s in batch])
        tensor = torch.from_numpy(array).to(device)
        return tensor.float() if dtype == "float" else tensor

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(index))
        total = 0.0
        for i in range(0, len(order), BATCH):
            batch = [index[j] for j in order[i:i + BATCH]]
            x, y, c = batch_of("x", batch, "float"), batch_of("y", batch), batch_of("c", batch)
            tokens = batch_of("tok", batch, "float") if config.encoder == "players" else None
            out = model(x, tokens)
            loss = ce(out[..., :3].reshape(-1, 3), y.reshape(-1))
            valid = c >= 0
            change_loss = ((bce(out[..., 3], c.clamp(min=0)) * valid).sum()
                           / valid.sum().clamp(min=1))
            loss = loss + CHANGE_WEIGHT * change_loss
            if config.aux_ball:
                target = batch_of("ball", batch)
                known = torch.isfinite(target[..., 0])
                diff = huber(out[..., 4:6], torch.nan_to_num(target)).sum(dim=-1)
                loss = loss + config.aux_ball * (diff * known).sum() / known.sum().clamp(min=1)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            total += float(loss) * len(batch)
        if verbose:
            print("  epoch {:>2}  loss {:.4f}".format(epoch + 1, total / len(index)), flush=True)
    return model.cpu().eval(), normaliser


def _infer(model, z, tokens=None):
    """Run over a long sequence in overlapping windows, keeping each window's centre.

    Recurrent and convolutional models read 3000-frame windows; the transformer reads windows of
    its training length, since its positions are absolute within a window.
    """
    torch = _torch()
    chunk, overlap = (CHUNK, 150) if model.config.arch == "transformer" else (3000, 500)
    n = len(z)
    out = None
    step = chunk - 2 * overlap
    with torch.no_grad():
        for start in range(0, n, step):
            lo, hi = max(0, start - overlap), min(n, start + step + overlap)
            tok = None if tokens is None else torch.from_numpy(
                np.asarray(tokens[lo:hi], dtype=np.float32)[None])
            pred = model(torch.from_numpy(z[None, lo:hi]), tok)[0].numpy()
            if out is None:
                out = np.zeros((n, pred.shape[1]), dtype=np.float64)
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
        tokens = sample.tokens(swap) if model.config.encoder == "players" else None
        for start, stop in sample.runs():
            raw = _infer(model, normaliser(np.asarray(x[start:stop])),
                         None if tokens is None else tokens[start:stop])
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


def combine(outputs):
    """Average ensemble members' `predict` outputs: probabilities in probability space, change
    logits as they are. Rounded like `predict`, for the same cross-machine reason."""
    stack = np.stack(outputs)
    out = np.empty_like(stack[0])
    out[:, :3] = np.log(np.clip(np.exp(stack[:, :, :3]).mean(axis=0), 1e-6, 1.0))
    out[:, 3] = stack[:, :, 3].mean(axis=0)
    return np.round(out, 4)


def predict_ensemble(members, sample):
    """`predict` for each (model, normaliser) member, combined."""
    sample = _as_sample(sample)
    return combine([predict(model, normaliser, sample) for model, normaliser in members])


def train_and_predict(train_samples, test_sample, config=DEFAULT_CONFIG):
    model, normaliser = train(train_samples, config=config)
    return predict(model, normaliser, test_sample)


def save(model, normaliser, path):
    torch = _torch()
    n_fields = model.players.embed[0].in_features if hasattr(model, "players") else None
    n_inputs = model.inp[0].in_features - (model.players.out_features
                                           if hasattr(model, "players") else 0)
    torch.save({"state_dict": model.state_dict(), "normaliser": normaliser.state(),
                "n_inputs": n_inputs, "n_token_fields": n_fields,
                "config": asdict(model.config)}, path)


def load(path):
    """(model, normaliser). Weights saved before configs existed load as the default GRU."""
    torch = _torch()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    config = TemporalConfig(**blob.get("config", {}))
    model = build_model(blob["n_inputs"], config, blob.get("n_token_fields"))
    model.load_state_dict(blob["state_dict"])
    return model.eval(), Normaliser.from_state(blob["normaliser"])
