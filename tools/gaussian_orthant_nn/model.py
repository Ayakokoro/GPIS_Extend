"""A shared, dimension-masked neural approximation to Gaussian log orthants.

The network sees only the standardized Gaussian, never its provenance. Sorting
by standardized mean reduces ordering sensitivity; equal-mean ties are stable
in input order and are not claimed to be exactly permutation invariant.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


FEATURE_SPEC = "asinh(z/2), rho, log(max(1-rho^2,1e-8))/8, valid-mask; stable z sort"


def canonicalize(z: torch.Tensor, corr: torch.Tensor, dim: torch.Tensor):
    if z.ndim != 2 or corr.shape != (z.shape[0], z.shape[1], z.shape[1]):
        raise ValueError("expected padded z[B,N] and corr[B,N,N]")
    dim = dim.to(dtype=torch.long, device=z.device).reshape(-1)
    if dim.shape[0] != z.shape[0]:
        raise ValueError("dimension count must match batch size")
    mask = torch.arange(z.shape[1], device=z.device)[None, :] < dim[:, None]
    order = torch.argsort(z.masked_fill(~mask, float("inf")), dim=1, stable=True)
    ordered_z = z.gather(1, order)
    ordered_corr = corr.gather(1, order[:, :, None].expand_as(corr))
    ordered_corr = ordered_corr.gather(2, order[:, None, :].expand_as(corr))
    return ordered_z.masked_fill(~mask, 0), ordered_corr, mask


def make_features(z: torch.Tensor, corr: torch.Tensor, dim: torch.Tensor) -> torch.Tensor:
    """Return 30 features for max_dim=5; input padding cannot affect output."""
    z, corr, mask = canonicalize(z, corr, dim)
    i, j = torch.triu_indices(z.shape[1], z.shape[1], offset=1, device=z.device)
    valid = mask[:, i] & mask[:, j]
    rho = corr[:, i, j].clamp(-1, 1).masked_fill(~valid, 0)
    near_singular = torch.log((1 - rho.square()).clamp_min(1e-8)) / 8
    return torch.cat((torch.asinh(z / 2), rho, near_singular, mask.to(z.dtype)), dim=1)


class OrthantNet(nn.Module):
    def __init__(self, max_dim: int = 5, width: int = 128, depth: int = 3):
        super().__init__()
        if not 1 <= max_dim <= 5 or width < 8 or depth < 1:
            raise ValueError("max_dim must be 1..5, width >= 8 and depth >= 1")
        self.max_dim, self.width, self.depth = max_dim, width, depth
        features = 2 * max_dim + max_dim * (max_dim - 1)
        layers = []
        for layer in range(depth):
            layers.extend((nn.Linear(features if layer == 0 else width, width), nn.SiLU()))
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        nn.init.zeros_(self.network[-1].weight)
        # Initial residual multiplier is 0.5; independent scaling keeps the
        # magnitude sensible for probabilities spanning many orders.
        nn.init.constant_(self.network[-1].bias, math.log(math.expm1(0.5)))

    def configuration(self) -> dict:
        return {"max_dim": self.max_dim, "width": self.width, "depth": self.depth}

    def forward(self, z: torch.Tensor, corr: torch.Tensor, dim: torch.Tensor) -> torch.Tensor:
        if z.shape[1] != self.max_dim:
            raise ValueError("input padding width differs from checkpoint max_dim")
        features = make_features(z, corr, dim)
        valid = torch.arange(self.max_dim, device=z.device)[None, :] < dim[:, None]
        log_cdf = torch.special.log_ndtr(z).masked_fill(~valid, 0)
        upper = log_cdf.masked_fill(~valid, float("inf")).amin(dim=1)
        independent_gap = (upper - log_cdf.sum(dim=1)).clamp_min(0)
        residual = F.softplus(self.network(features).squeeze(-1)) * (1 + independent_gap)
        # Frechet's union-bound lower probability. Summing Phi(-z) avoids
        # catastrophic cancellation in nearly certain positive orthants.
        failures = torch.exp(torch.special.log_ndtr(-z)).masked_fill(~valid, 0).sum(dim=1)
        lower = torch.log1p(-failures.clamp(max=1))
        lower = torch.minimum(lower, upper)  # Consistent bounds after rounding.
        lower_fraction = torch.exp(lower - upper).clamp(0, 1)
        remaining = upper + torch.log1p(-lower_fraction)
        # P = L + (U-L)*exp(-residual), smoothly confined to [L,U]. When L=0
        # this is exactly the original log-U minus residual, with no tail floor.
        estimate = torch.logaddexp(lower, remaining - residual)
        return torch.where(dim == 1, upper, estimate)
