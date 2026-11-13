"""Small, bounded correction of an analytic start-conditioned hazard.

The Gaussian statistics and the baseline hazard stay outside the network.
The probability transform uses the supplied log probability in its original
precision, including histories whose positive probability is almost one.
"""
from __future__ import annotations

import math

import torch
from torch import nn


FEATURE_NAMES = ("z_x", "z_u", "z_v", "rho_xu", "rho_xv", "rho_uv")
SCHEMA = "tungsten.single_point_history.v1"
MODEL_SCHEMA = "tungsten.single_point_history_mlp.v2"


def validate_correction_cap(cap: float | None) -> float | None:
    """A deployment cap preserves the R=1 analytic baseline; None disables it."""
    if cap is None:
        return None
    if isinstance(cap, bool):
        raise ValueError("correction cap must be a finite number >= 1, or None")
    cap = float(cap)
    if not math.isfinite(cap) or cap < 1:
        raise ValueError("correction cap must be a finite number >= 1, or None")
    return cap


def cap_log_correction(log_correction: torch.Tensor, cap: float | None) -> torch.Tensor:
    """Deployment-only upper cap. It never imposes a lower correction bound."""
    cap = validate_correction_cap(cap)
    return log_correction if cap is None else torch.clamp_max(log_correction, math.log(cap))


def log1mexp(log_probability: torch.Tensor) -> torch.Tensor:
    """Stable log(1-exp(x)) for x <= 0; x=0 deliberately returns -inf."""
    return torch.where(
        log_probability < -math.log(2.0),
        torch.log1p(-torch.exp(log_probability)),
        torch.log(-torch.expm1(log_probability)),
    )


def bounded_log_correction(residual: torch.Tensor, log_probability: torch.Tensor) -> torch.Tensor:
    """log R = -log[p + (1-p) exp(-f)], so 0 < R <= 1/p.

    This is an architectural bound for the neural model. A positive-correlation
    condition is additionally required to bound the numerical single-history
    teacher by the same analytic envelope; the data loader checks that domain.
    """
    return -torch.logaddexp(log_probability, log1mexp(log_probability) - residual)


class SinglePointResidualNet(nn.Module):
    def __init__(self, width: int = 32, feature_transform: str = "identity"):
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive")
        self.width = int(width)
        if feature_transform not in ("identity", "asinh_z"):
            raise ValueError("unknown feature transform")
        self.feature_transform = feature_transform
        self.register_buffer("feature_mean", torch.zeros(6))
        self.register_buffer("feature_scale", torch.ones(6))
        self.network = nn.Sequential(
            nn.Linear(6, self.width), nn.ReLU(),
            nn.Linear(self.width, self.width), nn.ReLU(),
            nn.Linear(self.width, 1),
        )
        # The initial hazard is exactly the analytic baseline in real arithmetic.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def set_normalization(self, mean, scale) -> None:
        mean = torch.as_tensor(mean, dtype=self.feature_mean.dtype, device=self.feature_mean.device)
        scale = torch.as_tensor(scale, dtype=self.feature_scale.dtype, device=self.feature_scale.device)
        if mean.shape != (6,) or scale.shape != (6,):
            raise ValueError("normalization must contain six means and scales")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("normalization must be finite with strictly positive scales")
        if not torch.equal(mean[3:], torch.zeros_like(mean[3:])) or not torch.equal(scale[3:], torch.ones_like(scale[3:])):
            raise ValueError("correlations retain their original scale")
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_(scale)

    def raw_residual(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 6:
            raise ValueError("expected features[N,6]")
        # No input clipping: out-of-domain evaluation must not masquerade as a
        # query at the edge of the training domain.
        if self.feature_transform == "asinh_z":
            features = torch.cat((torch.asinh(features[:, :3]), features[:, 3:]), dim=1)
        normalized = (features - self.feature_mean) / self.feature_scale
        return self.network(normalized).squeeze(-1)

    def forward(self, features: torch.Tensor, log_probability: torch.Tensor) -> torch.Tensor:
        # Training and checkpoint selection always use the original teacher
        # target and analytic 1/p envelope. No user deployment cap belongs here.
        if log_probability.ndim != 1 or log_probability.shape[0] != features.shape[0]:
            raise ValueError("expected one history log probability per feature row")
        residual = self.raw_residual(features)
        return bounded_log_correction(residual.to(log_probability.dtype), log_probability)

    @torch.no_grad()
    def inference_log_correction(self, features: torch.Tensor, log_probability: torch.Tensor,
                                 correction_cap: float | None = 2.0) -> torch.Tensor:
        return cap_log_correction(self(features, log_probability), correction_cap)

    def configuration(self) -> dict:
        return {"input_features": 6, "width": self.width, "hidden_layers": 2,
                "activation": "relu", "output_features": 1}
