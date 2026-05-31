"""Play wrapper: registers SMP tasks, auto-resolves the latest LOCAL checkpoint,
then delegates to mjlab.scripts.play.main.

Fully local — no wandb. If you don't pass ``--checkpoint-file`` (and aren't
using any wandb flags), this finds the most recently modified
``model_*.pt`` under the log root and injects it as ``--checkpoint-file`` so
``uv run scripts/play.py <Task>`` just plays the latest trained model.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mjlab.scripts.play import main

import smp.rl.tasks  # noqa: F401  # registers Smp-* tasks in the mjlab registry

# mjlab's PlayConfig.log_root default; kept in sync here for checkpoint discovery.
_DEFAULT_LOG_ROOT = "logs/rsl_rl"
# Any of these means the user opted into a remote/explicit source — don't touch.
_OVERRIDE_FLAGS = (
  "--checkpoint-file",
  "--wandb-run-path",
  "--registry-name",
  "--wandb-checkpoint-name",
)


def _flag_present(argv: list[str], flag: str) -> bool:
  return any(a == flag or a.startswith(flag + "=") for a in argv)


def _flag_value(argv: list[str], flag: str) -> str | None:
  for i, a in enumerate(argv):
    if a == flag and i + 1 < len(argv):
      return argv[i + 1]
    if a.startswith(flag + "="):
      return a.split("=", 1)[1]
  return None


def _latest_checkpoint(log_root: Path) -> Path | None:
  ckpts = list(log_root.rglob("model_*.pt"))
  if not ckpts:
    return None
  return max(ckpts, key=lambda p: p.stat().st_mtime)


def _maybe_inject_latest_checkpoint(argv: list[str]) -> list[str]:
  # Respect explicit checkpoint/wandb choices.
  if any(_flag_present(argv, f) for f in _OVERRIDE_FLAGS):
    return argv
  log_root = Path(_flag_value(argv, "--log-root") or _DEFAULT_LOG_ROOT)
  ckpt = _latest_checkpoint(log_root)
  if ckpt is None:
    print(f"[play] no local model_*.pt found under {log_root}; deferring to mjlab")
    return argv
  print(f"[play] using latest local checkpoint: {ckpt}")
  return [*argv, "--checkpoint-file", str(ckpt)]


if __name__ == "__main__":
  sys.argv = _maybe_inject_latest_checkpoint(sys.argv)
  main()
