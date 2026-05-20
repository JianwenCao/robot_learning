# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the Bonus-B singulation task."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

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

from .commands import (  # noqa: F401
    SingulationBowlPoseCommand,
    SingulationBowlPoseCommandCfg,
)
from .events import (  # noqa: F401
    ARR_ID,
    ARRANGEMENT_NAMES,
    ARRANGEMENT_SPECS,
    BLOCK_COLORS,
    COLOR_NAMES,
    DEFAULT_ARRANGEMENT_WEIGHTS,
    HIDDEN_PARK_XY,
    MAX_N_ACTIVE,
    NUM_ARRANGEMENTS,
    NUM_COLORS,
    randomize_cube_physics,
    reset_singulation_latches,
    sample_active_set,
)
from .observations import (  # noqa: F401
    active_block_mask,
    all_cube_positions_robot_frame,
    arrangement_onehot,
    bowl_xy,
    mean_pairwise_xy_active,
    min_pairwise_xy_active,
    n_active_onehot,
    n_cubes_off_table,
    wrist_rgb_dr,
    wrist_rgb_union_mask_dr,
)
from .rewards import (  # noqa: F401
    all_cubes_on_table,
    bowl_avoidance,
    cube_overspeed_penalty,
    lift_then_place,
    log_singulation_metrics,
    mean_pairwise_xy,
    min_pairwise_xy,
    reach_closest_pair,
    singulation_success,
)
from .terminations import (  # noqa: F401
    active_cube_off_table,
    singulation_done,
)
