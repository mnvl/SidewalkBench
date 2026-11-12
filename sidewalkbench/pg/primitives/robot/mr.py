"""The ``mr`` robot: the mobile-robotics project's differential-drive sidewalk robot, for its policy (``MR``).

A 0.40 x 0.30 x 0.40 m box (length, width, height) with a stereo pair on its
front panel, kinematic like ``--kinematics_only`` coco: the action writes the
root velocity every physics step.  Two things differ from coco's kinematic
action:

* It turns about the centre of the front panel, ``pivot_ahead_m`` ahead of the
  root, as the real robot and the simulator it was trained in do.  The root is
  the box's centre, where the benchmark's collision check puts the box.
* The ``(v, w)`` it takes is not a PD controller's: the policy's drive model
  (the mobile-robotics project's drive bridge) has already turned the
  policy's wheel torques into it, with that robot's acceleration limits and
  top speed.

The cameras sit on the box's front face, so the box is behind their near
plane and they never see it; it stays visible to every other view.  It
collides and rests on the ground, as coco's kinematic body does, but without
friction: the action rewrites the velocity every physics step, and friction
would take ~0.05 m/s off it in each one (mu = 1 over 5 ms), which the robot
would then report as its own deceleration.
"""

from __future__ import annotations

from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

# The mobile-robotics project's rl/sidewalkbench/isaac on PYTHONPATH; its evaluator puts it there.
from mr_isaac.kinematics import root_velocity_for_pivot

MR_BOX = (0.40, 0.30, 0.40)
MR_PIVOT_AHEAD = 0.5 * MR_BOX[0]

MR_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=MR_BOX,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        # The ground combines friction by multiplying (random_env_cfg.py), so zero here is zero.
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.0, dynamic_friction=0.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5 * MR_BOX[2])),
)


class PivotKinematicsAction(ActionTerm):
    """Writes the root velocity that moves the pivot at ``v`` along the heading while turning at ``w``."""

    cfg: PivotKinematicsActionCfg

    def __init__(self, cfg: PivotKinematicsActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.robot: RigidObject = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        yaw = self.robot.data.heading_w.torch
        velocity = root_velocity_for_pivot(self._raw_actions[:, 0], self._raw_actions[:, 1], yaw,
                                           self.cfg.pivot_ahead_m)
        self.robot.write_root_velocity_to_sim(velocity)


@configclass
class PivotKinematicsActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PivotKinematicsAction
    asset_name: str = MISSING
    pivot_ahead_m: float = MISSING
    """How far ahead of the root the robot turns about [m]."""


@configclass
class MRActionsCfg:
    drive = PivotKinematicsActionCfg(asset_name="robot", pivot_ahead_m=MR_PIVOT_AHEAD)
