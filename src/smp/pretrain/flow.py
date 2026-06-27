"""Conditional flow-matching formulation over the SMP DiT backbone.

A third ``model_family`` alongside ``edm`` and ``ddpm``. The network architecture
is unchanged -- we reuse ``DiffusionDenoiser`` as the trainable function, but
interpret its output as a **velocity field** ``v_theta(x_tau, tau)`` rather than a
noise / denoiser estimate.

Straight-line ("rectified flow") interpolant, with **tau=0 = noise, tau=1 = data**
(note: ``x_0`` in this codebase is *clean data*, so internally data is the tau=1
endpoint):

    x_tau = (1 - tau) * z + tau * x_data,      z ~ N(0, I)
    u     = dx_tau/dtau = x_data - z           (constant conditional velocity)
    L     = E_{tau, z} || v_theta(x_tau, tau) - (x_data - z) ||^2

No Karras-style preconditioning is needed: data is quantile-normalized to ~[-1, 1]
and ``z`` is unit-scale, so the network always sees ~unit-scale inputs.

``FMPrecond`` mirrors ``EDMPrecond``'s public surface on purpose -- it exposes
``sds_sigmas`` (here, the set of fixed *flow times* used for the reward),
``sds_err``, ``denoise`` and ``heun_sample`` with matching signatures -- so the RL
reward / GSI / viz code paths that branch on ``EDMPrecond`` work for flow models
with only the ``isinstance(...)`` tuples widened. The continuous DiT time encoder
accepts floats, so feeding ``c_time = time_emb_scale * tau`` needs no architecture
change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Fixed flow times tau in (0, 1) for the SDS-style reward path -- the flow-matching
# analog of EDM's ``DEFAULT_SDS_SIGMAS`` / DDPM's ``fixed_timesteps``.
DEFAULT_FM_TIMES: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)


@dataclass
class FMPrecond:
  """Rectified-flow velocity formulation. Holds no trainable weights.

  ``sds_sigmas`` holds the fixed flow times used by the guidance reward (the name
  is kept for drop-in compatibility with the EDM reward/GSI code paths).
  """

  time_emb_scale: float = 1000.0
  time_sampling: str = "logitnormal"  # "logitnormal" | "uniform"
  logitnorm_mu: float = 0.0
  logitnorm_sigma: float = 1.0
  sds_sigmas: tuple[float, ...] = field(default_factory=lambda: DEFAULT_FM_TIMES)

  # ---- velocity field v_theta(x_tau, tau) --------------------------------

  def _c_time(self, tau: torch.Tensor) -> torch.Tensor:
    """Scaled flow time fed to the DiT's sinusoidal time encoder."""
    return self.time_emb_scale * tau

  def velocity(
    self, model: torch.nn.Module, x: torch.Tensor, tau: torch.Tensor
  ) -> torch.Tensor:
    """Predicted velocity ``(B, W, F)`` at flow time ``tau`` (B,)."""
    return model(x, self._c_time(tau.flatten()))

  def denoise(
    self, model: torch.nn.Module, x: torch.Tensor, tau: torch.Tensor
  ) -> torch.Tensor:
    """Predicted clean-data endpoint from a point on the path.

    Along the path ``x_data = x_tau + (1 - tau) * u``; using the model's velocity
    gives ``x + (1 - tau) * v_theta(x, tau)``. Mirrors ``EDMPrecond.denoise`` so
    the compile-warmup path is identical.
    """
    s = tau.view(-1, 1, 1)
    return x + (1.0 - s) * self.velocity(model, x, tau)

  # ---- time sampling -----------------------------------------------------

  def _sample_tau(self, n: int, device: torch.device | str) -> torch.Tensor:
    if self.time_sampling == "uniform":
      return torch.rand(n, device=device)
    if self.time_sampling == "logitnormal":
      z = torch.randn(n, device=device) * self.logitnorm_sigma + self.logitnorm_mu
      return torch.sigmoid(z)
    msg = f"unknown time_sampling {self.time_sampling!r}"
    raise ValueError(msg)

  # ---- training loss (conditional flow matching) -------------------------

  def fm_loss(
    self, model: torch.nn.Module, x_0: torch.Tensor, num_noise_samples: int = 1
  ) -> torch.Tensor:
    """Flow-matching MSE. ``x_0`` is ``(B, W, F)`` clean normalized data.

    Each sample is paired with ``num_noise_samples`` independent (tau, z) draws,
    mirroring the EDM / DDPM variance reduction.
    """
    B = x_0.shape[0]
    K = num_noise_samples
    x_data = x_0[:, None].expand(B, K, *x_0.shape[1:]).reshape(B * K, *x_0.shape[1:])
    tau = self._sample_tau(B * K, x_data.device)
    z = torch.randn_like(x_data)
    s = tau.view(-1, 1, 1)
    x_tau = (1.0 - s) * z + s * x_data
    u = x_data - z
    v = self.velocity(model, x_tau, tau)
    return ((v - u) ** 2).mean()

  # ---- SDS-style residual for the reward path ----------------------------

  @torch.no_grad()
  def sds_err(
    self, model: torch.nn.Module, x_0: torch.Tensor, time_scalar: float
  ) -> torch.Tensor:
    """Per-sample squared velocity residual at a single flow time. Returns ``(B,)``.

    ``|| v_theta(x_tau, tau) - (x_data - z) ||^2`` with ``z ~ N(0, I)`` and
    ``x_tau = (1 - tau) z + tau x_data``. Like the EDM SDS residual this is
    nonzero even for a perfect model (the field predicts the *marginal* velocity),
    but it is much smaller on on-manifold windows than on off-manifold ones, which
    is what the reward needs.
    """
    B = x_0.shape[0]
    tau = x_0.new_full((B,), time_scalar)
    z = torch.randn_like(x_0)
    x_tau = (1.0 - time_scalar) * z + time_scalar * x_0
    u = x_0 - z
    v = self.velocity(model, x_tau, tau)
    return ((v - u) ** 2).mean(dim=(-1, -2))

  # ---- ODE sampler (offline + RL GSI) ------------------------------------

  @torch.no_grad()
  def heun_sample(
    self,
    model: torch.nn.Module,
    n: int,
    window_size: int,
    feature_dim: int,
    num_steps: int,
    device: torch.device | str,
  ) -> torch.Tensor:
    """Heun (2nd-order) integration of dx/dtau = v from tau=0 (noise) to tau=1.

    Returns ``(n, W, F)`` *normalized* windows (caller denormalizes). Signature
    matches ``EDMPrecond.heun_sample`` for drop-in GSI / viz use.
    """
    device = torch.device(device)
    x = torch.randn(n, window_size, feature_dim, device=device)
    dtau = 1.0 / num_steps
    for k in range(num_steps):
      t0 = x.new_full((n,), k * dtau)
      t1 = x.new_full((n,), (k + 1) * dtau)
      # ``velocity`` returns the raw model output; under a CUDA-graph-compiled
      # model that buffer is overwritten by the next call, so clone v0 to keep
      # it alive for the corrector average.
      v0 = self.velocity(model, x, t0).clone()
      x_pred = x + dtau * v0
      if k + 1 < num_steps:
        v1 = self.velocity(model, x_pred, t1)
        x = x + 0.5 * dtau * (v0 + v1)
      else:
        x = x_pred  # final step: plain Euler to tau=1
    return x


def fm_precond_from_cfg(cfg: dict) -> FMPrecond:
  """Build an ``FMPrecond`` from a checkpoint ``cfg`` dict (with defaults)."""
  times = cfg.get("fm_times", DEFAULT_FM_TIMES)
  return FMPrecond(
    time_emb_scale=float(cfg.get("fm_time_emb_scale", 1000.0)),
    time_sampling=str(cfg.get("fm_time_sampling", "logitnormal")),
    logitnorm_mu=float(cfg.get("fm_logitnorm_mu", 0.0)),
    logitnorm_sigma=float(cfg.get("fm_logitnorm_sigma", 1.0)),
    sds_sigmas=tuple(float(s) for s in times),
  )
