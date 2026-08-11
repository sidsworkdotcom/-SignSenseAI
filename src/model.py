"""Neural network architectures for SignSense AI.

SignNet     — MLP for *static* signs (a single frame of landmarks).
SignNetLSTM — LSTM for *dynamic* gestures (a sequence of landmark frames).

Design notes
------------
* BatchNorm after each Linear stabilizes training on small datasets.
* Dropout (0.3) fights overfitting — critical because our dataset is
  self-collected and relatively small.
* The MLP has only ~60K parameters, so inference is <1 ms on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


class SignNet(nn.Module):
    """MLP classifier: 63 → hidden layers → num_classes."""

    def __init__(
        self,
        num_classes: int,
        input_size: int = config.INPUT_SIZE,
        hidden_sizes: list[int] | None = None,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or config.HIDDEN_SIZES

        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SignNetLSTM(nn.Module):
    """LSTM classifier for dynamic gestures.

    Input shape: (batch, sequence_length, 63). The final hidden state feeds
    a small classification head.
    """

    def __init__(
        self,
        num_classes: int,
        input_size: int = config.INPUT_SIZE,
        hidden_size: int = config.LSTM_HIDDEN_SIZE,
        num_layers: int = config.LSTM_NUM_LAYERS,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # quick sanity check
    m = SignNet(num_classes=26)
    x = torch.randn(4, config.INPUT_SIZE)
    print("SignNet output:", m(x).shape, "| params:", count_parameters(m))

    lm = SignNetLSTM(num_classes=10)
    xs = torch.randn(4, config.SEQUENCE_LENGTH, config.INPUT_SIZE)
    print("SignNetLSTM output:", lm(xs).shape, "| params:", count_parameters(lm))
