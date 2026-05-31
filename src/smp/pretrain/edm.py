"""EDM (Karras et al. 2022) diffusion formulation over the SMP DiT backbone.

The network architecture is unchanged — we reuse ``DiffusionDenoiser`` as the
trainable function ``F``. Only the diffusion *formulation* differs from
``scheduler.DDPMScheduler``:

  * Continuous noise level ``sigma`` instead of an integer timestep ``t``.
  * Forward process ``x_sigma = x + sigma * eps`` (no betas/alphas/abars).
  * Karras preconditioning so ``F`` always sees ~unit-scale inputs and predicts
    ~unit-scale targets:
        D(x; sigma) = c_skip(sigma)*x + c_out(sigma) * F(c_in(sigma)*x, c_noise)
  * Log-normal ``sigma`` sampling with the Karras Eq. 8 loss weighting.
  * Heun 2nd-order ODE sampling on the Karras rho schedule.

``EDMPrecond`` is deliberately weightless: it stores only the scalar
hyperparameters and takes the backbone ``model`` as an argument, mirroring how
``DDPMScheduler`` is separate from ``DiffusionDenoiser`` so the RL bundle layout
``(model, scheduler_or_precond, q_low, q_high, feature_dim, window_size)`` and
``torch.compile(model)`` both keep working unchanged.

``DiffusionDenoiser``'s timestep encoder accepts floats, so feeding the
continuous ``c_noise = time_emb_scale * 0.25 * log(sigma)`` needs no
architecture change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Default fixed sigmas for the SDS reward path (log-spaced over a useful band),
# the EDM analog of the DDPM ``fixed_timesteps=(8, 15, 22)``.
DEFAULT_SDS_SIGMAS: tuple[float, ...] = (0.1, 0.3, 0.6, 1.0, 1.5)


@dataclass
class EDMPrecond:
  """Karras preconditioning + noise schedule. Holds no trainable weights.

  Data is quantile-normalized to ~[-1, 1] upstream, so ``sigma_data`` defaults
  to 0.5 (its empirical std) rather than the z-scored 1.0 used by mjlab.
  """

  sigma_data: float = 0.5
  sigma_min: float = 0.002
  sigma_max: float = 80.0
  rho: float = 7.0
  p_mean: float = -1.2
  p_std: float = 1.2
  time_emb_scale: float = 1000.0
  sds_sigmas: tuple[float, ...] = field(default_factory=lambda: DEFAULT_SDS_SIGMAS)

  # ---- Karras denoiser D(x; sigma) ---------------------------------------

  def _c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
    """Scaled log-sigma fed to the DiT's sinusoidal time encoder."""
    return self.time_emb_scale * 0.25 * torch.log(sigma)

  def denoise(
    self, model: torch.nn.Module, x: torch.Tensor, sigma: torch.Tensor
  ) -> torch.Tensor:
    """Preconditioned denoiser estimate of the clean window.

    Args:
      model: DiT backbone with ``forward(x, t)`` signature.
      x:     ``(B, W, F)`` noisy window.
      sigma: ``(B,)`` per-sample noise level.

    Returns:
      ``D(x; sigma)`` of shape ``(B, W, F)``.
    """
    s = sigma.view(-1, 1, 1)
    var = s**2 + self.sigma_data**2
    c_skip = self.sigma_data**2 / var
    c_out = s * self.sigma_data / torch.sqrt(var)
    c_in = 1.0 / torch.sqrt(var)
    c_noise = self._c_noise(sigma.flatten())
    f_out = model(c_in * x, c_noise)
    return c_skip * x + c_out * f_out

  # ---- Training loss (Karras Eq. 7-8) ------------------------------------

  def edm_loss(
    self, model: torch.nn.Module, x_0: torch.Tensor, num_noise_samples: int = 1
  ) -> torch.Tensor:
    """Weighted denoising MSE. ``x_0`` is ``(B, W, F)`` clean normalized data.

    Each sample is paired with ``num_noise_samples`` independent (sigma, eps)
    draws, mirroring ``scheduler._diffusion_loss``'s variance reduction.
    """
    B = x_0.shape[0]
    K = num_noise_samples
    x = x_0[:, None].expand(B, K, *x_0.shape[1:]).reshape(B * K, *x_0.shape[1:])
    log_sigma = self.p_mean + self.p_std * torch.randn(B * K, device=x.device)
    sigma = log_sigma.exp()
    eps = torch.randn_like(x)
    x_noised = x + sigma.view(-1, 1, 1) * eps
    d = self.denoise(model, x_noised, sigma)
    weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
    per_sample = ((d - x) ** 2).mean(dim=(-1, -2))
    return (weight * per_sample).mean()

  # ---- SDS residual for the reward path ----------------------------------

  @torch.no_grad()
  def sds_err(
    self, model: torch.nn.Module, x_0: torch.Tensor, sigma_scalar: float
  ) -> torch.Tensor:
    """Per-sample squared SDS residual at a single sigma. Returns ``(B,)``.

    Uses the exact (per-sample, not just in-expectation) identity
        eps_hat - eps = (x_0 + sigma*eps - D(x_0 + sigma*eps; sigma)) / sigma
                      = (x_0 - D) / sigma
    so no scheduler step is needed and the estimator has low variance.
    """
    B = x_0.shape[0]
    sigma = x_0.new_full((B,), sigma_scalar)
    eps = torch.randn_like(x_0)
    x_noised = x_0 + sigma_scalar * eps
    d = self.denoise(model, x_noised, sigma)
    residual = (x_0 - d) / sigma_scalar
    return (residual**2).mean(dim=(-1, -2))

  # ---- Heun sampler (offline + RL GSI) -----------------------------------

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
    """Heun 2nd-order ODE sampling on the Karras sigma schedule.

    Returns ``(n, W, F)`` *normalized* windows (caller denormalizes).
    """
    device = torch.device(device)
    rho = self.rho
    i = torch.arange(num_steps, device=device)
    sigmas = (
      self.sigma_max ** (1 / rho)
      + i / (num_steps - 1) * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))
    ) ** rho
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])  # final sigma = 0

    x = sigmas[0] * torch.randn(n, window_size, feature_dim, device=device)
    for k in range(num_steps):
      s_cur = sigmas[k]
      s_nxt = sigmas[k + 1]
      d_cur = self.denoise(model, x, s_cur.expand(n))
      slope_cur = (x - d_cur) / s_cur
      x_next = x + (s_nxt - s_cur) * slope_cur
      if s_nxt > 0:
        d_nxt = self.denoise(model, x_next, s_nxt.expand(n))
        slope_nxt = (x_next - d_nxt) / s_nxt
        x_next = x + (s_nxt - s_cur) * 0.5 * (slope_cur + slope_nxt)
      x = x_next
    return x


def edm_precond_from_cfg(cfg: dict) -> EDMPrecond:
  """Build an ``EDMPrecond`` from a checkpoint ``cfg`` dict (with defaults)."""
  sds = cfg.get("edm_sds_sigmas", DEFAULT_SDS_SIGMAS)
  return EDMPrecond(
    sigma_data=float(cfg.get("edm_sigma_data", 0.5)),
    sigma_min=float(cfg.get("edm_sigma_min", 0.002)),
    sigma_max=float(cfg.get("edm_sigma_max", 80.0)),
    rho=float(cfg.get("edm_rho", 7.0)),
    p_mean=float(cfg.get("edm_p_mean", -1.2)),
    p_std=float(cfg.get("edm_p_std", 1.2)),
    time_emb_scale=float(cfg.get("edm_time_emb_scale", 1000.0)),
    sds_sigmas=tuple(float(s) for s in sds),
  )
