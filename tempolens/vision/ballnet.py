"""TrackNet-style ball heatmap network (Huang et al., 2019), small enough for a 4 GB GPU.

Input: 2*CONTEXT+1 consecutive frames stacked on channels, so the network sees motion -- a
moving ball is easier to separate from a white boot or a line marking than a still one.
Output: a per-pixel logit that the ball is there. Fully convolutional: trained on crops, run on
whole frames.
"""

import numpy as np

from .io import CONTEXT

SIGMA = 2.5        # px, width of the Gaussian training target
POS_WEIGHT = 30.0  # the ball is a few pixels in tens of thousands
EPOCHS = 12
BATCH = 32
LR = 2e-3
PEAK_K = 3         # candidates kept per frame
PEAK_MIN = 0.1     # minimum heatmap probability for a candidate


def _torch():
    import torch
    return torch


def build_model(frames=2 * CONTEXT + 1, width=16):
    torch = _torch()
    nn = torch.nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU())

    class BallNet(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.e1, self.e2 = block(3 * frames, w), block(w, 2 * w)
            self.e3, self.e4 = block(2 * w, 4 * w), block(4 * w, 6 * w)
            self.d3, self.d2, self.d1 = block(10 * w, 4 * w), block(6 * w, 2 * w), block(3 * w, w)
            self.out = nn.Conv2d(w, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            up = lambda t: nn.functional.interpolate(t, scale_factor=2, mode="nearest")
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            e4 = self.e4(self.pool(e3))
            d3 = self.d3(torch.cat([up(e4), e3], 1))
            d2 = self.d2(torch.cat([up(d3), e2], 1))
            d1 = self.d1(torch.cat([up(d2), e1], 1))
            return self.out(d1)[:, 0]

    return BallNet()


def _to_input(stack):
    """(B, k, H, W, 3) uint8 BGR -> (B, 3k, H, W) float in [0, 1]."""
    torch = _torch()
    x = torch.as_tensor(stack).float().div_(255.0)
    b, k, h, w, c = x.shape
    return x.permute(0, 1, 4, 2, 3).reshape(b, k * c, h, w)


def _targets(offsets, size):
    """(B, size, size) Gaussian heatmaps; all zero where the offset is NaN (negative crops)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    out = np.zeros((len(offsets), size, size), dtype=np.float32)
    for i, (x, y) in enumerate(offsets):
        if np.isfinite(x):
            out[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * SIGMA ** 2))
    return out


def train(crops_path, targets_path, holdout_every=10, epochs=EPOCHS, device=None, seed=0,
          verbose=True):
    """Fit on the crop memmap. Every `holdout_every`-th sample (in time order) is held out and
    scored at the end: share of positive crops whose top peak is within 5 px of the label."""
    torch = _torch()
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    crops = np.load(crops_path, mmap_mode="r")
    offsets = np.load(targets_path)
    n, size = len(crops), crops.shape[2]
    held = np.arange(n) % holdout_every == 0
    train_idx, test_idx = np.flatnonzero(~held), np.flatnonzero(held)
    model = build_model(crops.shape[1]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(train_idx) / BATCH))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=LR, total_steps=steps)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device=device))
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        order = train_idx[rng.permutation(len(train_idx))]
        for i in range(0, len(order), BATCH):
            idx = np.sort(order[i:i + BATCH])   # sorted reads are faster from the memmap
            stack = np.asarray(crops[idx])
            target = _targets(offsets[idx], size)
            if rng.random() < 0.5:   # horizontal flip
                stack = stack[:, :, :, ::-1].copy()
                target = target[:, :, ::-1].copy()
            x = _to_input(stack).to(device)
            y = torch.from_numpy(target).to(device)
            loss = loss_fn(model(x), y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss) * len(idx)
        if verbose:
            print("  ball epoch {:>2}  loss {:.4f}".format(epoch + 1, total / len(train_idx)),
                  flush=True)
    model = model.eval()
    report = evaluate_crops(model, crops, offsets, test_idx, device)
    return model.cpu(), report


def evaluate_crops(model, crops, offsets, idx, device="cpu"):
    """Held-out crop accuracy: hit rate within 5 px on positives, false-peak rate on negatives."""
    torch = _torch()
    hits = positives = false_peaks = negatives = 0
    with torch.no_grad():
        for i in range(0, len(idx), BATCH):
            chunk = idx[i:i + BATCH]
            prob = torch.sigmoid(model(_to_input(np.asarray(crops[chunk])).to(device))).cpu().numpy()
            for j, frame in enumerate(chunk):
                y, x = np.unravel_index(np.argmax(prob[j]), prob[j].shape)
                if np.isfinite(offsets[frame, 0]):
                    positives += 1
                    hits += np.hypot(x - offsets[frame, 0], y - offsets[frame, 1]) <= 5
                else:
                    negatives += 1
                    false_peaks += prob[j].max() > 0.5
    return {"heldout_positive_hit_rate_5px": hits / max(positives, 1),
            "heldout_negative_false_peak_rate": false_peaks / max(negatives, 1),
            "heldout_positives": positives, "heldout_negatives": negatives}


def peaks(model, stack, device="cpu", k=PEAK_K, minimum=PEAK_MIN):
    """Top-k heatmap peaks of one full frame stack (k, H, W, 3): [(x, y, prob)]."""
    torch = _torch()
    nn = torch.nn
    _, height, width, _ = stack.shape
    pad_h, pad_w = (-height) % 8, (-width) % 8
    x = _to_input(stack[None]).to(device)
    x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
    with torch.no_grad():
        # float32, no autocast: half precision changes detections between GPUs
        prob = torch.sigmoid(model(x))[0, :height, :width]
        local_max = nn.functional.max_pool2d(prob[None, None], 9, stride=1, padding=4)[0, 0]
        prob = torch.where(prob == local_max, prob, torch.zeros_like(prob))
        values, flat = prob.flatten().topk(k)
    out = []
    for value, index in zip(values.cpu().numpy(), flat.cpu().numpy()):
        if value >= minimum:
            out.append((float(index % width), float(index // width), float(value)))
    return out


def save(model, path):
    torch = _torch()
    torch.save({"state_dict": model.state_dict(),
                "frames": model.e1[0].in_channels // 3}, path)


def load(path):
    torch = _torch()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(blob["frames"])
    model.load_state_dict(blob["state_dict"])
    return model.eval()
