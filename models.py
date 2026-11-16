"""Model zoo: feature MLP, raw-window MLP, CNN-BiLSTM and LightTCN.

All four are plain PyTorch so that local epochs mean real gradient steps (the
submission used ``sklearn.MLPClassifier`` with ``max_iter=local_epochs`` and a
post-hoc "proximal correction", which is neither). Because they are torch modules,
one aggregation and one DP routine serves every architecture, which is exactly what
Task 1 asks for.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config

ARCHES = ("MLP", "MLP-Small", "MLP-Raw", "CNN-BiLSTM", "LightTCN")


class FeatureMLP(nn.Module):
    """MLP on the 117 hand-crafted features."""

    def __init__(self, in_dim, n_class=None, hidden=None):
        super().__init__()
        n_class = n_class or config.N_CLASS
        hidden = hidden or config.MLP_HIDDEN
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, n_class)]
        self.body = nn.Sequential(*layers[:-1])
        self.head = layers[-1]
        self.input_kind = "features"

    def forward(self, x):
        return self.head(self.body(x))


class RawMLP(nn.Module):
    """MLP on flattened windows - the representation-matched control (Task 3)."""

    def __init__(self, window_shape, n_class=None, hidden=None):
        super().__init__()
        n_class = n_class or config.N_CLASS
        in_dim = int(np.prod(window_shape))
        hidden = hidden or config.MLP_HIDDEN
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        # (the original dropped the output layer: its "head" was the last ReLU, so the
        # model emitted len(hidden[-1]) scores instead of n_class)
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_class)
        self.input_kind = "flat_windows"
        self.window_shape = tuple(window_shape)

    def forward(self, x):
        return self.head(self.body(x.reshape(x.shape[0], -1)))


class CNNBiLSTM(nn.Module):
    """Multi-branch conv (3/5/7) + 2-layer BiLSTM, as in the submission."""

    def __init__(self, n_ch=None, n_class=None, filters=32):
        super().__init__()
        n_ch = n_ch or config.N_CH_HARMONISED
        n_class = n_class or config.N_CLASS
        self.br3 = nn.Sequential(nn.Conv1d(n_ch, filters, 3, padding=1), nn.ReLU(),
                                 nn.Dropout(0.1))
        self.br5 = nn.Sequential(nn.Conv1d(n_ch, filters, 5, padding=2), nn.ReLU(),
                                 nn.Dropout(0.1))
        self.br7 = nn.Sequential(nn.Conv1d(n_ch, filters, 7, padding=3), nn.ReLU(),
                                 nn.Dropout(0.1))
        self.bilstm = nn.LSTM(filters * 3, 64, batch_first=True, bidirectional=True,
                              num_layers=2, dropout=0.3)
        self.body = nn.ModuleList([self.br3, self.br5, self.br7, self.bilstm])
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(),
                                  nn.Linear(64, n_class))
        self.input_kind = "windows"

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.cat([self.br3(x), self.br5(x), self.br7(x)], dim=1)
        x = x.permute(0, 2, 1)
        out, _ = self.bilstm(x)
        return self.head(out[:, -1, :])


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=self.pad, dilation=dilation)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]
        return F.relu(out) + self.res(x)


class LightTCN(nn.Module):
    def __init__(self, n_ch=None, n_class=None, ch=32):
        super().__init__()
        n_ch = n_ch or config.N_CH_HARMONISED
        n_class = n_class or config.N_CLASS
        self.body = nn.Sequential(TCNBlock(n_ch, ch, dilation=1),
                                  TCNBlock(ch, ch, dilation=2),
                                  TCNBlock(ch, ch, dilation=4))
        self.head = nn.Linear(ch, n_class)
        self.input_kind = "windows"

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.head(self.body(x)[:, :, -1])


def build(arch, feature_dim=None, window_shape=None, n_class=None, device=None):
    """window_shape is (WL, C): the channel count of the model follows the data, so a
    3-channel accelerometer-only input really gives a 3-channel network."""
    arch = arch.upper() if isinstance(arch, str) else arch
    key = {"MLP": "MLP", "MLP-SMALL": "MLP-Small", "MLP-RAW": "MLP-Raw",
           "CNN-BILSTM": "CNN-BiLSTM",
           "CNN": "CNN-BiLSTM", "LIGHTTCN": "LightTCN", "TCN": "LightTCN"}.get(arch)
    if key is None:
        raise ValueError("unknown arch %r (one of %s)" % (arch, ARCHES))
    n_ch = None
    if window_shape is not None and len(window_shape) > 1:
        n_ch = int(window_shape[1])
    if key == "MLP":
        m = FeatureMLP(feature_dim, n_class)
    elif key == "MLP-Small":
        # 117 -> 64 -> C: about 7.9k released coordinates instead of 71.8k, the
        # user-level DP arm whose noise-to-signal ratio is sqrt(d) smaller
        m = FeatureMLP(feature_dim, n_class, hidden=config.MLP_SMALL_HIDDEN)
    elif key == "MLP-Raw":
        m = RawMLP(window_shape, n_class)
    elif key == "CNN-BiLSTM":
        m = CNNBiLSTM(n_ch=n_ch, n_class=n_class)
    else:
        m = LightTCN(n_ch=n_ch, n_class=n_class)
    if device is not None:
        m = m.to(device)
    return m


def n_tensors(model):
    return len([p for p in model.parameters() if p.requires_grad])


def n_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def param_shapes(model):
    return [tuple(p.shape) for p in model.parameters() if p.requires_grad]


def get_weights(model):
    return [p.detach().cpu().numpy().astype(np.float32).copy()
            for p in model.parameters() if p.requires_grad]


def set_weights(model, weights):
    ps = [p for p in model.parameters() if p.requires_grad]
    assert len(ps) == len(weights), (len(ps), len(weights))
    with torch.no_grad():
        for p, w in zip(ps, weights):
            p.data.copy_(torch.as_tensor(w, dtype=p.dtype, device=p.device))


def zero_like(model):
    return [np.zeros_like(p.detach().cpu().numpy()) for p in model.parameters()
            if p.requires_grad]


def n_head_params(model):
    """How many trailing parameter tensors belong to the classifier head.

    FedPer keeps these private and shares only the feature extractor. For all four
    architectures the head's parameters are the last ones listed by
    ``model.parameters()``.
    """
    if not hasattr(model, "head"):
        return 0
    return len(list(model.head.parameters()))


def n_class_of(model):
    """Number of output classes, for heads that are Linear or Sequential."""
    head = getattr(model, "head", None)
    if isinstance(head, nn.Linear):
        return int(head.out_features)
    if isinstance(head, nn.Sequential):
        for layer in reversed(list(head)):
            if isinstance(layer, nn.Linear):
                return int(layer.out_features)
    return config.N_CLASS


def device_for(use_cuda=True):
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
