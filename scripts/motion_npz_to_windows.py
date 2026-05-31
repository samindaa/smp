"""Convert preprocessed motion NPZ files to windowed NPZ files.

This is a sibling of ``csv_to_npz.py`` for inputs that are *already* FK'd.
The motion ``.npz`` files referenced by an SMP motion-mix manifest (e.g.
``mjlab.../smp/configs/g1_locomotion.yaml``) already contain world-frame body
state, so there is no CSV-to-sim replay step: we read the root state and
end-effector positions straight out of the file and feed them into the same
windowing function used by ``csv_to_npz`` (``_compute_windows``).

Each input file holds the per-frame arrays:

  joint_pos      (T, 29)
  body_pos_w     (T, n_bodies, 3)   body index 0 == root/pelvis
  body_quat_w    (T, n_bodies, 4)   wxyz
  body_lin_vel_w (T, n_bodies, 3)
  body_ang_vel_w (T, n_bodies, 3)
  fps            (1,)

The body ordering matches the runtime G1 scene, so we resolve the tracked
end-effector body indices from a scene built exactly like the env (see
``_setup_sim``). The output layout is identical to ``csv_to_npz``: a 59-dim
per-frame feature window (see that module's docstring).

Usage:
  uv run scripts/motion_npz_to_windows.py \
    --manifest ../mjlab/src/mjlab/learning/smp/configs/g1_locomotion.yaml \
    --output-dir datasets/npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import yaml
from mjlab.entity import Entity

from csv_to_npz import (
  EE_BODY_NAMES,
  NUM_EE,
  NUM_JOINTS,
  _compute_windows,
  _setup_sim,
)
from smp.utils import detect_device


@dataclass
class Cfg:
  manifest: str = "../mjlab/src/mjlab/learning/smp/configs/g1_locomotion.yaml"
  """Motion-mix YAML manifest (the ``weight`` field is ignored)."""
  output_dir: str = "datasets/npz"
  """Directory to write output NPZ window files."""
  window_size: int = 10
  """Number of frames per window."""
  stride: int = 1
  """Stride between consecutive windows."""
  device: str = ""
  """Compute device. Empty = auto (cuda if available else cpu)."""
  data_root: str = ""
  """Optional base dir for resolving relative manifest paths. Empty = search
  cwd, the manifest's parent, and each ancestor of the manifest."""


def _resolve_manifest(manifest_path: Path, data_root: str) -> list[Path]:
  """Read a motion-mix YAML and return the list of motion files (no weights).

  Relative entries are resolved against, in order: ``data_root`` (if given),
  the cwd, the manifest's parent, and every ancestor directory of the
  manifest. The first existing candidate wins.
  """
  with manifest_path.open() as f:
    raw = yaml.safe_load(f)
  if "motions" not in raw:
    msg = f"{manifest_path}: missing 'motions' top-level key"
    raise ValueError(msg)

  search_bases: list[Path] = []
  if data_root:
    search_bases.append(Path(data_root))
  search_bases.append(Path.cwd())
  search_bases.extend(manifest_path.resolve().parents)

  out: list[Path] = []
  for entry in raw["motions"]:
    rel = Path(entry["file"])
    if rel.is_absolute():
      out.append(rel)
      continue
    for base in search_bases:
      cand = (base / rel).resolve()
      if cand.exists():
        out.append(cand)
        break
    else:
      tried = ", ".join(str(b / rel) for b in search_bases)
      msg = f"{manifest_path}: motion {entry['file']!r} not found (tried: {tried})"
      raise FileNotFoundError(msg)
  if not out:
    msg = f"{manifest_path}: motion list is empty"
    raise ValueError(msg)
  return out


def _load_motion(
  npz_path: Path,
  ee_indexes: torch.Tensor,
  device: str,
) -> tuple[
  torch.Tensor,  # base_pos       (T, 3)
  torch.Tensor,  # base_quat      (T, 4)
  torch.Tensor,  # base_lin_vel   (T, 3)
  torch.Tensor,  # base_ang_vel   (T, 3)
  torch.Tensor,  # ee_pos         (T, NUM_EE, 3)
  torch.Tensor,  # joint_pos      (T, NUM_JOINTS)
  float,  # fps
]:
  """Read root state + end-effector positions from a motion NPZ."""
  raw = np.load(npz_path)
  required = (
    "joint_pos",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "fps",
  )
  for key in required:
    if key not in raw.files:
      msg = f"{npz_path.name} missing key {key!r}"
      raise KeyError(msg)

  def t(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)

  joint_pos = t(raw["joint_pos"])  # (T, J)
  body_pos_w = t(raw["body_pos_w"])  # (T, B, 3)
  body_quat_w = t(raw["body_quat_w"])  # (T, B, 4)
  body_lin_vel_w = t(raw["body_lin_vel_w"])  # (T, B, 3)
  body_ang_vel_w = t(raw["body_ang_vel_w"])  # (T, B, 3)
  fps = float(raw["fps"][0])

  # Body index 0 is the root/pelvis.
  base_pos = body_pos_w[:, 0]
  base_quat = body_quat_w[:, 0]
  base_lin_vel = body_lin_vel_w[:, 0]
  base_ang_vel = body_ang_vel_w[:, 0]
  ee_pos = body_pos_w[:, ee_indexes]  # (T, NUM_EE, 3)

  return base_pos, base_quat, base_lin_vel, base_ang_vel, ee_pos, joint_pos, fps


def main(cfg: Cfg) -> None:
  if not cfg.device:
    cfg.device = detect_device()
  print(f"Device: {cfg.device}")

  manifest_path = Path(cfg.manifest)
  if not manifest_path.exists():
    msg = f"Manifest not found: {manifest_path}"
    raise FileNotFoundError(msg)
  motion_files = _resolve_manifest(manifest_path, cfg.data_root)

  out_dir = Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  sim, scene = _setup_sim(cfg.device)
  robot: Entity = scene["robot"]
  ee_indexes = torch.tensor(
    robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=sim.device,
  )

  feature_dims = [3, 6, NUM_JOINTS, NUM_EE * 3, 3, 3]
  total_feature_dim = sum(feature_dims)

  print(f"Manifest: {manifest_path} ({len(motion_files)} motions)")
  print(f"Output: {out_dir}")
  print(f"Window: size={cfg.window_size} stride={cfg.stride}")
  print(f"End-effectors: {NUM_EE} {EE_BODY_NAMES} | Joints: {NUM_JOINTS}")
  print(
    f"Feature dim: {total_feature_dim} "
    f"(= 3 root_pos + 6 root_rot + {NUM_JOINTS} joint_pos + {NUM_EE * 3} "
    f"ee_pos + 3 lin_vel + 3 ang_vel)"
  )

  for i, npz_path in enumerate(motion_files):
    print(f"\n[{i + 1}/{len(motion_files)}] {npz_path.name}")
    (
      base_pos,
      base_quat,
      base_lin_vel,
      base_ang_vel,
      ee_pos,
      joint_pos,
      fps,
    ) = _load_motion(npz_path, ee_indexes, cfg.device)
    if joint_pos.shape[-1] != NUM_JOINTS:
      msg = (
        f"{npz_path.name}: expected {NUM_JOINTS} dof columns, "
        f"got {joint_pos.shape[-1]}"
      )
      raise ValueError(msg)

    windows = _compute_windows(
      base_pos,
      base_quat,
      base_lin_vel,
      base_ang_vel,
      ee_pos,
      joint_pos,
      cfg.window_size,
      cfg.stride,
    )
    if windows is None:
      print(f"  [SKIP] too short for window_size={cfg.window_size}")
      continue

    out_path = out_dir / f"{npz_path.stem}.npz"
    np.savez_compressed(
      out_path,
      windows=windows.cpu().numpy().astype(np.float32),
      fps=np.array([fps], dtype=np.float32),
      window_size=np.array([cfg.window_size], dtype=np.int32),
      stride=np.array([cfg.stride], dtype=np.int32),
      ee_body_names=np.array(EE_BODY_NAMES),
      feature_dims=np.array(feature_dims, dtype=np.int32),
    )
    print(f"  saved {out_path.name}: windows={tuple(windows.shape)}")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
