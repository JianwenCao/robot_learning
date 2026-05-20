# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for SO-ARM101 Eval-2 (clutter pick-and-place with target color).

Layered on top of the Eval-1 (single-cube) MDP package:

* Re-exports everything from :mod:`isaaclab.envs.mdp` (generic terms).
* Re-exports the Eval-1 helpers we reuse verbatim (gripper_state, wrist
  image, bowl_xy, ee FK helpers, intrinsics loader, latch reset events).
* Overlays the Eval-2 specific bits: :class:`TargetColorCommand`,
  multi-cube placement event, target-cube-aware reach/lift/transport/
  release rewards, and a goal-conditioned target-color one-hot obs.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

# Reuse Eval-1 helpers that aren't cube-identity-specific.
from isaac_so_arm101.tasks.pickplace.mdp.observations import (  # noqa: F401
    bowl_xy,
    bowl_xyz,
    ee_proj_xy,
    ee_to_bowl_xy,
    ee_xyz_in_robot_root_frame,
    gripper_state,
    load_wrist_cam_intrinsics,
)
from isaac_so_arm101.tasks.pickplace.mdp.events import (  # noqa: F401
    randomize_wrist_hsv_dr,
    randomize_wrist_image_tint,
    reset_was_grasped,
    reset_was_over_bowl_above_rim,
)
from isaac_so_arm101.tasks.pickplace.mdp.rewards import (  # noqa: F401
    wrist_cam_table_clearance,
)

from .commands import (  # noqa: F401
    ClusterBowlPoseCommand,
    ClusterBowlPoseCommandCfg,
    TargetColorCommand,
    TargetColorCommandCfg,
)
from .events import (  # noqa: F401
    BLOCK_COLORS,
    COLOR_NAMES,
    HIDDEN_PARK_XY,
    NUM_COLORS,
    place_clutter_blocks,
    reset_cube_positions_bias,
    reset_target_latches,
)
from .observations import (  # noqa: F401
    cube_positions_xy_noisy,
    cube_visible_flags,
    distractor_block_position,
    target_block_position,
    target_block_to_bowl_xy,
    target_color_onehot,
    target_cube_pos_xy_noisy,
    target_gripper_to_block,
    target_is_grasped,
    wrist_rgb_dr,
    wrist_rgb_mask_dr,
)
from .rewards import (  # noqa: F401
    distractor_disturb_penalty,
    log_target_success_metrics,
    reach_target_block,
    release_target_in_bowl,
    target_grasp_event,
    target_gripper_open_above_bowl_lure,
    target_in_bowl,
    target_still_grasped_above_bowl_penalty,
    target_transport_to_bowl,
    wrong_block_in_bowl,
)
from .terminations import (  # noqa: F401
    block_off_table_any,
    target_task_success,
)
