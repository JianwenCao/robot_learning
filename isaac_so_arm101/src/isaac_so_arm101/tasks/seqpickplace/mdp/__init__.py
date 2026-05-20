# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for SO-ARM101 Eval-3 (sequential pick-and-place)."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

# Re-use Eval-1 generic helpers (state obs, intrinsics, latch resets that
# act on env._wrist_image_dr).
from isaac_so_arm101.tasks.pickplace.mdp.observations import (  # noqa: F401
    ee_proj_xy,
    ee_xyz_in_robot_root_frame,
    gripper_state,
    load_wrist_cam_intrinsics,
)
from isaac_so_arm101.tasks.pickplace.mdp.events import (  # noqa: F401
    randomize_wrist_hsv_dr,
    randomize_wrist_image_tint,
)
# AprilTag reset event is task-agnostic (uses NUM_COLORS / COLOR_NAMES
# from clutterpickplace events.py, which match seqpickplace's palette
# verbatim). Re-exported here so seqpickplace env cfgs can list it as a
# reset event without a cross-task import.
from isaac_so_arm101.tasks.clutterpickplace.mdp.events import (  # noqa: F401
    reset_cube_positions_bias,
)
from isaac_so_arm101.tasks.pickplace.mdp.rewards import (  # noqa: F401
    wrist_cam_table_clearance,
)

from .commands import SequentialGoalCommand, SequentialGoalCommandCfg  # noqa: F401
from .events import (  # noqa: F401
    BLOCK_COLORS,
    COLOR_NAMES,
    HIDDEN_PARK_XY,
    NUM_COLORS,
    N_ACTIVE_BLOCKS,
    N_GOAL_STEPS,
    place_seq_blocks,
    reset_seq_latches,
)
from .observations import (  # noqa: F401
    all_active_block_positions,
    cube_positions_xy_noisy,
    cube_visible_flags,
    current_step_onehot,
    current_target_block_position,
    current_target_block_to_bowl_xy,
    current_target_bowl_xy,
    current_target_color_onehot,
    current_target_gripper_to_block,
    seq_goal_vector,
    wrist_rgb_dr,
    wrist_rgb_mask_dr,
)
from .rewards import (  # noqa: F401
    log_seq_success_metrics,
    reach_current_target,
    lift_current_target,
    release_current_target_in_bowl,
    step_completion_bonus,
    transport_current_target_to_bowl,
    wrong_cube_in_current_bowl,
)
from .terminations import (  # noqa: F401
    active_block_off_table,
    all_steps_done,
)
