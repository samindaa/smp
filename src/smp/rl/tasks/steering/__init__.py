"""SMP steering tasks — registers ``Smp-Steering-G1``, ``Smp-Forward-G1``,
``Smp-Steering-Flow-G1`` and ``Smp-Forward-Flow-G1`` on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.steering.forward_env_cfg import g1_forward_smp_env_cfg
from smp.rl.tasks.steering.steering_env_cfg import g1_steering_smp_env_cfg

# Flow-matching locomotion prior (drop-in for the EDM prior in the SMP reward).
_FLOW_LOCO_CKPT = "logs/pretrain/flow_loco/20260619_225907/pretrained.pt"

_steering_rl = unitree_g1_smp_ppo_runner_cfg()
_steering_rl.experiment_name = "smp_steering_g1"
_steering_rl.run_name = "smp_steering_g1"

register_mjlab_task(
  task_id="Smp-Steering-G1",
  env_cfg=g1_steering_smp_env_cfg(play=False),
  play_env_cfg=g1_steering_smp_env_cfg(play=True),
  rl_cfg=_steering_rl,
)

# Same steering task, but the SMP guidance reward is driven by the flow-matching
# locomotion prior instead of the EDM prior.
_steering_flow_rl = unitree_g1_smp_ppo_runner_cfg()
_steering_flow_rl.experiment_name = "smp_steering_flow_g1"
_steering_flow_rl.run_name = "smp_steering_flow_g1"

register_mjlab_task(
  task_id="Smp-Steering-Flow-G1",
  env_cfg=g1_steering_smp_env_cfg(play=False, prior_ckpt=_FLOW_LOCO_CKPT),
  play_env_cfg=g1_steering_smp_env_cfg(play=True, prior_ckpt=_FLOW_LOCO_CKPT),
  rl_cfg=_steering_flow_rl,
)

_forward_rl = unitree_g1_smp_ppo_runner_cfg()
_forward_rl.experiment_name = "smp_forward_g1"
_forward_rl.run_name = "smp_forward_g1"

register_mjlab_task(
  task_id="Smp-Forward-G1",
  env_cfg=g1_forward_smp_env_cfg(play=False),
  play_env_cfg=g1_forward_smp_env_cfg(play=True),
  rl_cfg=_forward_rl,
)

# Same forward task, but the SMP guidance reward is driven by the flow-matching
# locomotion prior instead of the EDM prior.
_forward_flow_rl = unitree_g1_smp_ppo_runner_cfg()
_forward_flow_rl.experiment_name = "smp_forward_flow_g1"
_forward_flow_rl.run_name = "smp_forward_flow_g1"

register_mjlab_task(
  task_id="Smp-Forward-Flow-G1",
  env_cfg=g1_forward_smp_env_cfg(play=False, prior_ckpt=_FLOW_LOCO_CKPT),
  play_env_cfg=g1_forward_smp_env_cfg(play=True, prior_ckpt=_FLOW_LOCO_CKPT),
  rl_cfg=_forward_flow_rl,
)

__all__ = [
  "g1_forward_smp_env_cfg",
  "g1_steering_smp_env_cfg",
]
