"""Unconditionally generate a motion window with a trained SMP diffusion model
and visualize the predicted trajectory in a viser viewer.

Features carry ``root_pos`` (xy heading-inv + world z) and ``root_rot``
(6D tan-norm, heading-inv relative to the last-frame root), so the
world-frame pelvis trajectory is reconstructed directly from those two —
no velocity integration needed.  The last window frame is placed at a
chosen anchor pose (default: the robot's default standing state) and the
rest of the window is reconstructed relative to it.  EE positions come
from the sampled ``ee_pos`` feature lifted into world via the per-frame
pelvis pose.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import viser
from mjlab.entity import Entity
from mjlab.viewer.viser.scene import MjlabViserScene

from smp.pretrain.edm import EDMPrecond, edm_precond_from_cfg
from smp.pretrain.flow import FMPrecond, fm_precond_from_cfg
from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler
from smp.sampling.feature_to_state import (
  NUM_EE,
  window_to_ee_trajectories,
  window_to_pelvis_trajectory,
)
from smp.utils import detect_device


@dataclass
class Cfg:
  ckpt_path: str = ""
  """Path to a local SMP diffusion checkpoint .pt file."""
  device: str = ""
  """Compute device. Empty = auto."""
  fps: float = 50.0
  """Playback frame rate."""


def _resolve_ckpt_path(cfg: Cfg) -> str:
  """Validate and return the local checkpoint path."""
  if not cfg.ckpt_path:
    msg = "Specify --ckpt-path pointing to a local checkpoint .pt file"
    raise ValueError(msg)
  if not Path(cfg.ckpt_path).is_file():
    msg = f"Checkpoint not found: {cfg.ckpt_path}"
    raise FileNotFoundError(msg)
  return cfg.ckpt_path


def _build_model_and_sampler(
  ckpt: dict, device: torch.device
) -> tuple[
  DiffusionDenoiser, DDPMScheduler | EDMPrecond | FMPrecond, np.ndarray, np.ndarray
]:
  """Return (model, scheduler_or_precond, q_low, q_high).

  The second element is a ``DDPMScheduler`` (DDPM), ``EDMPrecond`` (EDM) or
  ``FMPrecond`` (flow); ``_run_generate`` branches on its type.
  """
  cfg = ckpt["cfg"]
  model = DiffusionDenoiser(
    feature_dim=cfg["feature_dim"],
    window_size=cfg["window_size"],
    d_model=cfg.get("d_model", 256),
    nhead=cfg.get("nhead", 8),
    num_layers=cfg.get("num_layers", 2),
    dropout=cfg.get("dropout", 0.0),
  ).to(device)
  state = ckpt.get("model_ema") or ckpt["model"]
  model.load_state_dict(state)
  model.eval()
  family = cfg.get("model_family", "ddpm")
  if family == "edm":
    sampler: DDPMScheduler | EDMPrecond | FMPrecond = edm_precond_from_cfg(cfg)
  elif family == "flow":
    sampler = fm_precond_from_cfg(cfg)
  else:
    sampler = DDPMScheduler(num_timesteps=cfg.get("num_timesteps", 50)).to(device)
  return model, sampler, ckpt["q_low"], ckpt["q_high"]


def _setup_g1_sim(device: str):
  """Build a single G1 sim. Mirrors scripts/csv_to_npz.py:_setup_sim."""
  from mjlab.scene import Scene
  from mjlab.sim.sim import Simulation, SimulationCfg
  from mjlab.tasks.tracking.config.g1.env_cfgs import (
    unitree_g1_flat_tracking_env_cfg,
  )

  sim_cfg = SimulationCfg()
  env_cfg = unitree_g1_flat_tracking_env_cfg()
  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene


def _quantile_denormalize(
  x: torch.Tensor, q_low: torch.Tensor, q_high: torch.Tensor
) -> torch.Tensor:
  return (x + 1.0) / 2.0 * (q_high - q_low) + q_low


@torch.no_grad()
def _run_generate(
  model: DiffusionDenoiser,
  sampler: DDPMScheduler | EDMPrecond | FMPrecond,
  q_low: np.ndarray,
  q_high: np.ndarray,
  window_size: int,
  feature_dim: int,
  device: torch.device,
  edm_steps: int,
) -> torch.Tensor:
  """Unconditional sampling. Returns (W, F) denormalized window on CPU.

  EDM and flow checkpoints use the Heun ODE sampler; DDPM checkpoints use
  ancestral sampling.
  """
  if isinstance(sampler, (EDMPrecond, FMPrecond)):
    x_0 = sampler.heun_sample(
      model, 1, window_size, feature_dim, edm_steps, device
    ).squeeze(0)
  else:
    x_t = torch.randn(1, window_size, feature_dim, device=device)
    for t in reversed(range(sampler.num_timesteps)):
      t_batch = torch.full((1,), t, dtype=torch.long, device=device)
      eps = model(x_t, t_batch)
      x_t = sampler.step(eps, x_t, t)
    x_0 = x_t.squeeze(0)
  q_low_t = torch.from_numpy(q_low).float().to(device)
  q_high_t = torch.from_numpy(q_high).float().to(device)
  return _quantile_denormalize(x_0, q_low_t, q_high_t).cpu()


def _write_pose_to_robot(
  robot: Entity,
  pelvis_pos: np.ndarray,
  pelvis_quat_wxyz: np.ndarray,
  joint_pos: np.ndarray,
  device: str,
) -> None:
  """Mirror scripts/csv_to_npz.py:_fk_motion's per-frame state write."""
  root = robot.data.default_root_state.clone()
  root[:, 0:3] = torch.as_tensor(pelvis_pos, device=device, dtype=root.dtype)
  root[:, 3:7] = torch.as_tensor(pelvis_quat_wxyz, device=device, dtype=root.dtype)
  robot.write_root_state_to_sim(root)
  jp = robot.data.default_joint_pos.clone()
  jp[:] = torch.as_tensor(joint_pos, device=device, dtype=jp.dtype)
  jv = robot.data.default_joint_vel.clone()
  robot.write_joint_state_to_sim(jp, jv)


def main(cfg: Cfg) -> None:
  device_str = cfg.device or detect_device()
  device = torch.device(device_str)
  print(f"Device: {device_str}")

  ckpt_path = _resolve_ckpt_path(cfg)
  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
  model, sampler, q_low, q_high = _build_model_and_sampler(ckpt, device)
  family = ckpt["cfg"].get("model_family", "ddpm")
  print(f"Loaded checkpoint epoch={ckpt.get('epoch')} family={family} from {ckpt_path}")

  feature_dim = int(ckpt["cfg"]["feature_dim"])
  window_size = int(ckpt["cfg"]["window_size"])
  # ODE step count for the Heun sampler (EDM / flow); ignored by DDPM.
  if family == "flow":
    edm_steps = int(ckpt["cfg"].get("fm_sample_steps", 8))
  else:
    edm_steps = int(ckpt["cfg"].get("edm_sample_steps", 18))

  sim_device = device_str
  sim, scene = _setup_g1_sim(sim_device)
  robot: Entity = scene["robot"]
  mj_model = sim.mj_model

  # Place the last window frame at the robot's default standing pose.
  anchor_pelvis_pos = robot.data.default_root_state[0, 0:3].detach().cpu()
  anchor_pelvis_quat = robot.data.default_root_state[0, 3:7].detach().cpu()

  def run() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_denorm = _run_generate(
      model,
      sampler,
      q_low,
      q_high,
      window_size,
      feature_dim,
      device,
      edm_steps,
    )
    p_pos, p_quat, p_joint = window_to_pelvis_trajectory(
      pred_denorm,
      anchor_pelvis_pos,
      anchor_pelvis_quat,
    )
    ee_pos = window_to_ee_trajectories(pred_denorm, p_pos, p_quat)
    return (
      p_pos.cpu().numpy(),
      p_quat.cpu().numpy(),
      p_joint.cpu().numpy(),
      ee_pos.cpu().numpy(),
    )

  state: dict = {"pred": run()}

  server = viser.ViserServer()
  viser_scene = MjlabViserScene(server, mj_model, num_envs=1)
  viser_scene.debug_visualization_enabled = True

  # /fixed_bodies parents under mjviser's camera-tracking scene offset, so
  # the points stay aligned with the re-centered robot.
  ee_points = server.scene.add_point_cloud(
    name="/fixed_bodies/predicted_ee_positions",
    points=np.zeros((NUM_EE, 3), dtype=np.float32),
    colors=np.tile(np.array([255, 80, 0], dtype=np.uint8), (NUM_EE, 1)),
    point_size=0.03,
  )

  with server.gui.add_folder("Generate"):
    frame_slider = server.gui.add_slider(
      "Frame", min=0, max=window_size - 1, step=1, initial_value=0
    )
    play_btn = server.gui.add_button("Play / Pause")
    resample_btn = server.gui.add_button("Resample")

  playing = {"v": True}

  @play_btn.on_click
  def _(_evt) -> None:
    playing["v"] = not playing["v"]

  @resample_btn.on_click
  def _(_evt) -> None:
    state["pred"] = run()

  def render(frame: int) -> None:
    p_pos, p_quat, p_joint, ee_pos = state["pred"]
    _write_pose_to_robot(robot, p_pos[frame], p_quat[frame], p_joint[frame], sim_device)
    sim.forward()
    wd = sim.wp_data
    viser_scene.update_from_arrays(
      body_xpos=np.asarray(wd.xpos.numpy()),
      body_xmat=np.asarray(wd.xmat.numpy()),
      qpos=np.asarray(wd.qpos.numpy()),
      env_idx=0,
    )
    ee_points.points = ee_pos[frame]
    viser_scene.refresh_visualization()

  print("Viser server running. Open the printed URL.")
  dt_play = 1.0 / cfg.fps
  try:
    while True:
      render(int(frame_slider.value))
      if playing["v"]:
        nxt = (int(frame_slider.value) + 1) % window_size
        frame_slider.value = nxt
      time.sleep(dt_play)
  except KeyboardInterrupt:
    print("Shutting down.")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
