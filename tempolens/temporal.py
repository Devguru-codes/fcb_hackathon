"""Sequence models over per-frame features: possession probabilities plus a change detector.

A frame classifier smears short possessions: its window features average over them. A recurrent
model reads the whole sequence around a frame, so it can say "this 1.5 s stretch belongs to the
other side" sharply, and its second head says where a possession changes, which the decoder uses
to make changes cheap exactly there.

Output per frame: log P(H), log P(A), log P(D), and the change logit.
Deterministic: seeded, and inference runs on the CPU unless asked otherwise.

Variants, chosen by TemporalConfig and saved inside each weights file:
- arch: "gru" (bidirectional GRU), "lstm" (bidirectional LSTM), "tcn" (dilated temporal convolutions, ~40 s receptive field)
  or "transformer" (self-attention over 5 fps tokens with a full-rate skip path);
- encoder: "features" (the hand-made per-frame features) or "players" (those plus a per-frame
  attention encoder over the set of players -- a fully connected graph network, slot-free);
- aux_ball: weight of an auxiliary head that regresses the true ball position on frames where
  the simulator hid it, i.e. learning to place an unseen ball from how the players move;
- smooth: weight of MS-TCN++'s truncated smoothing loss (Li et al., TPAMI 2020) on frame-to-frame
  changes of the log-probabilities, which targets over-segmentation -- what spellF1 punishes;
- seg_balance: exponent a of a per-frame weight (median run length / own run length)^a, clipped
  to [0.25, 4] -- spellF1 counts every possession once, so short ones should not be outvoted by
  the frames of long ones (ASRF's segment-frequency balancing, Ishikawa et al. 2021);
- focal: focal-loss exponent (Lin et al. 2017): down-weights frames the model already gets right;
- mirror: reflect half the training sequences across the pitch's long axis (football is
  left-right symmetric; TacticAI builds the same symmetry in) -- more data for free;
- touch: weight of an auxiliary head that predicts, per frame, whether a home / away player is
  playing the ball (targets from the open matches' event data, snapped to the physical contact:
  touches.py) -- it makes the network learn where control happens, the oracle's headroom; the
  head is not used at inference;
- dead_weight: cross-entropy weight of D frames -- the real match's stoppages are under-called
  (dead-ball recall 0.53), so make missing one cost more.
"""

from dataclasses import asdict, dataclass

import hashlib

import numpy as np

from .features import FEATURE_VERSION, TableSample, swap_labels
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
    arch: str = "gru"            # gru | lstm | tcn | transformer
    encoder: str = "features"    # features | players
    hidden: int = HIDDEN
    aux_ball: float = 0.0        # weight of the hidden-ball regression loss; 0 = no ball head
    epochs: int = EPOCHS
    smooth: float = 0.0          # weight of the truncated smoothing loss; 0 = off
    seg_balance: float = 0.0     # exponent of the run-length frame weight; 0 = off
    focal: float = 0.0           # focal-loss gamma; 0 = plain cross-entropy
    mirror: bool = False         # reflect half the training sequences across the long axis
    dead_weight: float = 1.0     # cross-entropy weight of D frames; 1 = unweighted
    chunk: int = CHUNK           # frames per training sequence (inference reads 3000)
    touch: float = 0.0           # weight of the touch-head loss; 0 = no touch head
    drop: tuple = ()             # feature names fed to the network as zeros (masked out)

    def tag(self):
        """Cache/file tag; the default config keeps the historical name "temporal"."""
        parts = [part for part, default in ((self.arch, "gru"), (self.encoder, "features"))
                 if part != default]
        if self.hidden != HIDDEN:
            parts.append("h{}".format(self.hidden))
        if self.aux_ball:
            parts.append("aux{:g}".format(self.aux_ball))
        if self.epochs != EPOCHS:
            parts.append("e{}".format(self.epochs))
        if self.smooth:
            parts.append("sm{:g}".format(self.smooth))
        if self.seg_balance:
            parts.append("sb{:g}".format(self.seg_balance))
        if self.focal:
            parts.append("fg{:g}".format(self.focal))
        if self.mirror:
            parts.append("mir")
        if self.dead_weight != 1.0:
            parts.append("dw{:g}".format(self.dead_weight))
        if self.chunk != CHUNK:
            parts.append("c{}".format(self.chunk))
        if self.touch:
            parts.append("tch{:g}".format(self.touch))
        if self.drop:
            parts.append("drop" + hashlib.md5(",".join(sorted(self.drop)).encode()).hexdigest()[:6])
        return "temporal" + ("_" + "_".join(parts) if parts else "")


DEFAULT_CONFIG = TemporalConfig()


def _torch():
    import torch
    return torch


class Normaliser:
    """Robust per-feature scaling fitted on training frames; NaN -> 0 plus a missing flag."""

    def __init__(self, x, drop=()):
        self.drop = np.asarray(drop, dtype=np.int64)    # columns masked to 0, flags included
        self.median = np.nanmedian(x, axis=0)
        q75, q25 = np.nanpercentile(x, [75, 25], axis=0)
        self.scale = np.where(np.isfinite(q75 - q25) & (q75 - q25 > 1e-6), q75 - q25, 1.0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        self.flagged = np.flatnonzero(np.isnan(x).any(axis=0))

    def __call__(self, x):
        z = np.clip((x - self.median) / self.scale, -10, 10)
        flags = np.isnan(x[:, self.flagged]).astype(np.float32)
        if len(self.drop):
            z[:, self.drop] = 0.0
            flags[:, np.isin(self.flagged, self.drop)] = 0.0
        return np.concatenate([np.nan_to_num(z, nan=0.0), flags], axis=1).astype(np.float32)

    def state(self):
        return {"median": self.median, "scale": self.scale, "flagged": self.flagged,
                "drop": self.drop}

    @classmethod
    def from_state(cls, state):
        self = cls.__new__(cls)
        self.median, self.scale = state["median"], state["scale"]
        self.flagged = state["flagged"]
        self.drop = np.asarray(state.get("drop", ()), dtype=np.int64)
        return self


def adapt_normaliser(normaliser, sample, alpha=1.0):
    """Test-time adaptation by re-centring: the scaling moved `alpha` of the way to the target
    match's own medians and spreads (unlabelled; AdaBN for a model without normalisation
    layers). The missing-value flags keep the training columns, so the network's inputs keep
    their width."""
    sample = _as_sample(sample)
    own = Normaliser(np.asarray(sample.features(False))[sample.period > 0][::10])
    out = Normaliser.from_state(normaliser.state())
    out.median = (1 - alpha) * normaliser.median + alpha * own.median
    out.scale = (1 - alpha) * normaliser.scale + alpha * own.scale
    return out


def head_inputs(model, normaliser, sample, swap):
    """(n, width) per-frame inputs of the output layer from one team view, in the model's
    precision; zeros outside the halves."""
    torch = _torch()
    head, model.head = model.head, torch.nn.Identity()
    try:
        x = sample.features(swap)
        out = None
        for start, stop in sample.runs():
            h = _infer(model, normaliser(np.asarray(x[start:stop])))
            if out is None:
                out = np.zeros((sample.n_frames, h.shape[1]), dtype=h.dtype)
            out[start:stop] = h
        return out
    finally:
        model.head = head


def eata_inputs(model, normaliser, sample):
    """(h0, h1, inside, fit): both views' per-frame head inputs, the in-half mask, and EATA's fit
    set -- in-half frames whose missing-value flags are not all set (view 0 then view 1), every
    5th frame, in the head's precision."""
    torch = _torch()
    inside = np.zeros(sample.n_frames, dtype=bool)
    for start, stop in sample.runs():
        inside[start:stop] = True
    h0, h1 = (head_inputs(model, normaliser, sample, swap) for swap in (False, True))
    keep = inside
    if len(normaliser.flagged):
        x = np.asarray(sample.features(False))
        keep = inside & ~np.isnan(x[:, normaliser.flagged]).all(axis=1)
    fit = torch.from_numpy(np.concatenate([h0[keep], h1[keep]])[::5]).to(model.head.weight.dtype)
    return h0, h1, inside, fit


def adapt_head(model, normaliser, sample, steps, lr=1e-3, gate=0.4):
    """Test-time adaptation of the output layer (EATA, Niu et al. 2022): the head takes `steps`
    full-batch Adam steps on `eata_inputs`' fit set that lower the H/A/D entropy of frames already
    confident (entropy < gate * ln 3), each weighted exp(E0 - E). Unlabelled; CPU, in the model's
    precision and a fixed order, so deterministic. Modifies `model`."""
    sample = _as_sample(sample)
    fit = eata_inputs(model, normaliser, sample)[3]
    for _ in entropy_steps(model.head, fit, steps, lr, gate):
        pass
    return model.eval()


def entropy_steps(head, h, steps, lr=1e-3, gate=0.4):
    """The EATA update of `head` on head inputs `h`, one full-batch Adam step per iteration;
    yields the step count after each, so a caller can read the head at several step counts."""
    torch = _torch()
    e0 = gate * np.log(3)
    optimiser = torch.optim.Adam(head.parameters(), lr=lr)
    for step in range(steps):
        p = torch.softmax(head(h)[:, :3], dim=1)
        entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
        confident = entropy < e0
        if not confident.any():
            break
        weight = torch.exp(e0 - entropy[confident]).detach()
        loss = (weight * entropy[confident]).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        yield step + 1


def self_train(model, normaliser, sample, pseudo, epochs=2, lr=1e-4):
    """Self-training on the target match: fine-tune the last recurrent layer and the output layer
    on pseudo-labels (`pseudo`: (n,) codes H/A/D from the home view, -100 = unlabelled), both
    team views, cross-entropy only. CPU float64, dropout off, chunks in a fixed order: the same
    inputs give the same weights. Modifies and returns `model`."""
    torch = _torch()
    sample = _as_sample(sample)
    if model.config.arch not in ("gru", "lstm"):
        raise ValueError("self_train fine-tunes a recurrent model's last layer")
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("head.") or (name.startswith("rnn.") and "_l1" in name))
    trainable = [param for param in model.parameters() if param.requires_grad]
    dtype = next(model.parameters()).dtype
    chunks = []
    for swap in (False, True):
        z = normaliser(np.asarray(sample.features(swap)))
        y = pseudo.copy()
        if swap:
            y[pseudo == H], y[pseudo == A] = A, H
        for start, stop in sample.runs():
            starts = list(range(start, stop, CHUNK))
            for s in starts:
                e = min(stop, s + CHUNK)
                if (y[s:e] >= 0).any():
                    chunks.append((z[s:e], y[s:e]))
    optimiser = torch.optim.Adam(trainable, lr=lr)
    model.eval()
    for _ in range(epochs):
        for i in range(0, len(chunks), BATCH):
            group = chunks[i:i + BATCH]
            width = max(len(c[1]) for c in group)
            x = np.zeros((len(group), width, group[0][0].shape[1]), dtype=np.float64)
            y = np.full((len(group), width), -100, dtype=np.int64)
            for k, (zz, yy) in enumerate(group):
                x[k, :len(yy)], y[k, :len(yy)] = zz, yy
            out = model(torch.from_numpy(x).to(dtype))
            loss = torch.nn.functional.cross_entropy(out[..., :3].reshape(-1, 3),
                                                     torch.from_numpy(y).reshape(-1),
                                                     ignore_index=-100)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    for param in model.parameters():
        param.requires_grad_(True)
    return model.eval()


def pseudo_labels(logp, states, runs, min_len=75, edge=25, confidence=0.9):
    """(n,) pseudo-labels from a decoded match: frames at least `edge` frames inside a decoded H/A/D
    run of `min_len` frames or more where the ensemble's probability of that state is at least
    `confidence`; -100 elsewhere."""
    out = np.full(len(states), -100, dtype=np.int64)
    p = np.exp(logp[:, :3])
    for start, stop in runs:
        s = states[start:stop]
        bounds = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1, len(s)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            label = int(s[a])
            if label > D or b - a < min_len:
                continue
            lo, hi = start + a + edge, start + b - edge
            frames = np.arange(lo, hi)
            sure = frames[p[lo:hi, label] >= confidence]
            out[sure] = label
    return out


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
            elif config.arch == "lstm":
                self.rnn = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True,
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
            # H, A, D logits + change logit (+ home / away touch logits)
            self.head = nn.Linear(width_out, 6 if config.touch else 4)
            if config.aux_ball:
                self.ball_head = nn.Linear(width_out, 2)

        def forward(self, x, tokens=None):
            if config.encoder == "players":
                x = torch.cat([x, self.players(tokens)], dim=-1)
            h = self.inp(x)
            if config.arch in ("gru", "lstm"):
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


def mirror_y(x, names=None):
    """Features of the same frames reflected across the pitch's long axis (y -> -y).

    Only the per-team sorted y-coordinates depend on the side: they change sign and so re-sort
    (NaN padding stays last). Everything else is symmetric already (|y| distances, spreads).
    """
    from .features import feature_names
    names = names or feature_names()
    out = np.array(x, dtype=np.float32, copy=True)
    for team in ("t0", "t1"):
        cols = [names.index("{}_y{}".format(team, i)) for i in range(11)]
        out[:, cols] = np.sort(-out[:, cols], axis=1)
    return out


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
        from .features import feature_names
        names = feature_names()
        normaliser = Normaliser(stacked, [names.index(n) for n in config.drop])
    out = []
    flip = np.random.default_rng(12345)   # which sequences are mirrored: fixed, so runs repeat
    for sample in samples:
        for swap in (False, True):
            x = sample.features(swap)
            y = swap_labels(sample.labels) if swap else sample.labels
            tokens = sample.tokens(swap) if config.encoder == "players" else None
            ball = sample.ball_target(swap) if config.aux_ball else None
            touch = sample.touch_target(swap) if config.touch else None
            for start, stop in sample.runs():
                y_run = y[start:stop].astype(np.int64)
                change = change_targets(y_run)
                y_run[y_run == X] = -100   # CrossEntropyLoss ignore_index
                raw = np.asarray(x[start:stop])
                if config.mirror and flip.random() < 0.5:
                    raw = mirror_y(raw)
                seq = {"x": normaliser(raw).astype(np.float16),
                       "y": y_run, "c": change}
                if config.seg_balance:
                    seq["w"] = run_weights(y_run, config.seg_balance)
                if tokens is not None:
                    seq["tok"] = tokens[start:stop]
                if ball is not None:
                    seq["ball"] = np.asarray(ball[start:stop], dtype=np.float32)
                if touch is not None:
                    seq["touch"] = np.asarray(touch[start:stop], dtype=np.float32)
                out.append(seq)
    return out, normaliser


def run_weights(labels, exponent, low=0.25, high=4.0):
    """Per frame (median run length / length of the run it sits in)^exponent, clipped; 0 where
    the label is unknown (-100). Every run of one label -- a possession or a stoppage -- then
    weighs about the same in the loss whatever its length."""
    bounds = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    lengths = np.repeat(np.diff(bounds), np.diff(bounds)).astype(np.float64)
    known = labels >= 0
    median = np.median(np.diff(bounds)[labels[bounds[:-1]] >= 0]) if known.any() else 1.0
    w = np.clip((median / lengths) ** exponent, low, high)
    return np.where(known, w, 0.0).astype(np.float32)


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


def train(samples, epochs=None, device=None, seed=SEED, verbose=True,
          config=DEFAULT_CONFIG, on_epoch=None):
    """Fit a network on labelled samples (or MatchTables). Returns (CPU model, normaliser).
    `epochs` overrides config.epochs; `on_epoch(epoch, model, normaliser)` is called after each
    epoch (the model still training, on its device)."""
    epochs = config.epochs if epochs is None else epochs
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sequences, normaliser = _sequences(samples, config)
    index = _chunk_index(sequences, config.chunk, config.chunk // 2)
    n_fields = sequences[0]["tok"].shape[-1] if "tok" in sequences[0] else None
    model = build_model(sequences[0]["x"].shape[1], config, n_fields).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(index) / BATCH))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=LR, total_steps=steps)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100, weight=None if config.dead_weight == 1.0
                                   else torch.tensor([1.0, 1.0, config.dead_weight], device=device))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none",
                                     pos_weight=torch.tensor(4.0, device=device))
    huber = torch.nn.SmoothL1Loss(reduction="none", beta=0.1)
    rng = np.random.default_rng(seed)

    def batch_of(key, batch, dtype=None):
        array = np.stack([np.asarray(sequences[k][key][s:s + config.chunk]) for k, s in batch])
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
            if config.seg_balance or config.focal:
                logp = torch.log_softmax(out[..., :3].float(), dim=-1)
                known = y >= 0
                nll = -logp.gather(-1, y.clamp(min=0)[..., None])[..., 0]
                w = batch_of("w", batch) if config.seg_balance else known.float()
                if config.focal:
                    w = w * (1 - torch.exp(-nll)).pow(config.focal)
                w = w * known
                loss = (nll * w).sum() / (batch_of("w", batch) * known if config.seg_balance
                                           else known.float()).sum().clamp(min=1)
            else:
                loss = ce(out[..., :3].reshape(-1, 3), y.reshape(-1))
            valid = c >= 0
            change_loss = ((bce(out[..., 3], c.clamp(min=0)) * valid).sum()
                           / valid.sum().clamp(min=1))
            loss = loss + CHANGE_WEIGHT * change_loss
            if config.smooth:
                # truncated MSE between neighbouring frames' log-probabilities (MS-TCN++, tau=4)
                logp = torch.log_softmax(out[..., :3], dim=-1)
                step = (logp[:, 1:] - logp[:, :-1].detach()) ** 2
                labelled = (y[:, 1:] >= 0)[..., None]
                loss = loss + config.smooth * ((step.clamp(max=16.0) * labelled).sum()
                                               / labelled.sum().clamp(min=1) / 3)
            if config.aux_ball:
                target = batch_of("ball", batch)
                known = torch.isfinite(target[..., 0])
                first = 6 if config.touch else 4
                diff = huber(out[..., first:first + 2], torch.nan_to_num(target)).sum(dim=-1)
                loss = loss + config.aux_ball * (diff * known).sum() / known.sum().clamp(min=1)
            if config.touch:
                target = batch_of("touch", batch)
                known = torch.isfinite(target)
                touch_loss = bce(out[..., 4:6], torch.nan_to_num(target)) * known
                loss = loss + config.touch * touch_loss.sum() / known.sum().clamp(min=1)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            total += float(loss) * len(batch)
        if verbose:
            print("  epoch {:>2}  loss {:.4f}".format(epoch + 1, total / len(index)), flush=True)
        if on_epoch:
            on_epoch(epoch + 1, model, normaliser)
    return model.cpu().eval(), normaliser


def _infer(model, z, tokens=None):
    """Run over a long sequence in overlapping windows, keeping each window's centre.

    Recurrent and convolutional models read 3000-frame windows; the transformer reads windows of
    its training length, since its positions are absolute within a window. Full-width windows
    run as one batch (~8x faster on CPU); the shorter edge windows run alone, so no padding ever
    enters a bidirectional model.
    """
    torch = _torch()
    dtype = next(model.parameters()).dtype
    chunk, overlap = (CHUNK, 150) if model.config.arch == "transformer" else (3000, 500)
    n = len(z)
    step = chunk - 2 * overlap
    starts = list(range(0, n, step))
    spans = [(max(0, start - overlap), min(n, start + step + overlap)) for start in starts]
    width = max(hi - lo for lo, hi in spans)

    def run(group):
        x = torch.from_numpy(np.stack([z[lo:hi] for lo, hi in group])).to(dtype)
        tok = None if tokens is None else torch.from_numpy(np.stack(
            [np.asarray(tokens[lo:hi], dtype=np.float32) for lo, hi in group])).to(dtype)
        return model(x, tok).numpy()

    preds = {}
    with torch.no_grad():
        full = [k for k, (lo, hi) in enumerate(spans) if hi - lo == width]
        for i in range(0, len(full), 16):   # 16 windows at a time bounds memory
            batch = full[i:i + 16]
            preds.update(zip(batch, run([spans[k] for k in batch])))
        for k in set(range(len(spans))) - set(full):
            preds[k] = run([spans[k]])[0]
    out = np.zeros((n, preds[0].shape[1]), dtype=np.float64)
    for k, start in enumerate(starts):
        lo, keep = spans[k][0], min(start + step, n) - start
        out[start:start + keep] = preds[k][start - lo:start - lo + keep]
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
                "config": asdict(model.config), "feature_version": FEATURE_VERSION}, path)


def load(path):
    """(model, normaliser). Weights saved before configs existed load as the default GRU."""
    torch = _torch()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    # Weights read features by column: other features give a silently wrong model. Files from
    # before the version was recorded were trained on version 2.
    version = blob.get("feature_version", 2)
    if version != FEATURE_VERSION:
        raise ValueError("{} was trained on features v{}, the code computes v{}: retrain it "
                         "(python -m tempolens.train)".format(path, version, FEATURE_VERSION))
    config = TemporalConfig(**blob.get("config", {}))
    model = build_model(blob["n_inputs"], config, blob.get("n_token_fields"))
    model.load_state_dict(blob["state_dict"])
    # float64 inference: float32 sums differ in the last bits between CPU types (AVX2 vs
    # AVX-512), enough to flip the 4-decimal rounding of ~2% of values, and with it a decoder
    # tie somewhere; the rules require the organisers' re-run to give the same file.
    return model.double().eval(), Normaliser.from_state(blob["normaliser"])
