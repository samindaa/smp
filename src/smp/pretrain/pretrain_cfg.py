"""Diffusion pretraining configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from smp.utils import detect_device


@dataclass
class PretrainCfg:
  """Configuration for diffusion model pretraining."""

  # Data
  data_dir: str = "datasets/npz"
  norm_stats_file: str = "datasets/norm_stats.npz"
  """Path to q01/q99 quantile stats from compute_norm_stats.py."""
  train_split: float = 0.9

  # Diffusion formulation. "edm" (Karras preconditioning, continuous sigma) or
  # "ddpm" (cosine-beta scheduler). Both share the DiffusionDenoiser DiT.
  model_family: Literal["edm", "ddpm"] = "edm"
  # EDM hyperparameters (used only when model_family == "edm"). sigma_data=0.5
  # matches the std of the quantile-normalized (~[-1, 1]) features.
  edm_sigma_data: float = 0.5
  edm_sigma_min: float = 0.002
  edm_sigma_max: float = 80.0
  edm_rho: float = 7.0
  edm_p_mean: float = -1.2
  edm_p_std: float = 1.2
  edm_time_emb_scale: float = 1000.0
  edm_sample_steps: int = 18
  """Heun steps for the periodic SDS sanity check / sampling."""

  # Model. ``d_model = nhead · head_dim`` is the DiT inner dim; FF inner
  # dim is fixed at 4·d_model.
  d_model: int = 256
  nhead: int = 4
  num_layers: int = 2
  dropout: float = 0.0

  # Diffusion
  num_timesteps: int = 50
  num_noise_samples: int = 10
  """Random (t, ε) draws per data point in the diffusion loss."""

  # EMA
  use_ema: bool = False
  ema_decay: float = 0.9999

  # Training
  batch_size: int = 1024
  num_epochs: int = 2000
  lr: float = 3e-4
  weight_decay: float = 1e-4
  max_grad_norm: float = 1.0

  # Logging
  name: str = "pretrain"
  """Run identifier; used as the wandb run name and the save subfolder."""
  log_interval: int = 10
  save_interval: int = 100
  log_dir: str = "logs/pretrain"
  use_tensorboard: bool = True
  """Write scalars to TensorBoard under the run's save dir."""
  wandb_project: str = "smp"
  use_wandb: bool = False

  # Device
  device: str = ""

  # Reproducibility
  seed: int = 42

  def __post_init__(self) -> None:
    if not self.device:
      self.device = detect_device()
