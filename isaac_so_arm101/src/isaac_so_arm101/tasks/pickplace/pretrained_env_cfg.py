# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env cfg subclass for EVAL1_PLAN §9 cold-start path.

Subclasses :class:`SoArm101PickPlaceBowlEnvCfg` (the §7 production env)
to override the curriculum, rewards, and events for **cold-start without
teacher distillation**. The §7 path stays untouched.

What this subclass changes vs §7:

1. **Curriculum** (`PretrainedColdStartCurriculumCfg`) — block-xy expand
   schedule extended 3×/2× and adds a `p_grasped` decay term for the
   bootstrap event below.
2. **Rewards** (`PretrainedRewardsCfg`) — `lifting_object` threshold
   lowered from 0.07 m back to **0.025 m** (stock Franka Lift / ManiSkill3
   value). Cold-start cannot discover a 7 cm lift through random
   exploration; the strict gate kills gradient signal. The 0.07 rim
   clearance is still enforced separately on the success / release path
   via the over-bowl-above-rim latch — see ``release_in_bowl`` and
   ``task_success``. So this only weakens the *intermediate* lift bonus,
   not the deploy-safety gate.
3. **Events** (`PretrainedEventCfg`) — adds an `init_block_in_gripper`
   bootstrap event with `p_grasped=0.5` (curriculum-decayed). Half of
   episodes spawn with the block already in the closed gripper at home
   pose, so the post-grasp reward stages (lift / transport / release)
   see strong gradient signal from iter 0. The from-scratch half inherits
   competence via shared weights as p_grasped decays.

The two changes (lift threshold + bootstrap-grasp) together close the
cold-start failure mode diagnosed across runs `pickplace_bowl_pretrained/
2026-05-11_17-43-35` (v1, fine-tune) and `2026-05-11_22-18-36` (v2,
frozen): both runs saturated reach at ~0.81 but never discovered lift
because σ collapsed around the hover-near-cube basin before random
exploration could produce a successful grasp + 7 cm lift.

This is the **ManiSkill3 PickCube** recipe pattern modulo we use
init-bootstrap instead of contact-detection grasp reward. ManiSkill3 uses
``agent.is_grasping(cube)`` (physics contact-pair query) to pay +1 per
step when grasped without a height gate; we encode the same idea by
forcing half of episodes to start grasped so the post-grasp reward
stages pay out immediately on those episodes. Same effect on the value
landscape; less Isaac-Lab plumbing.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaac_so_arm101.tasks.pickplace.mdp as mdp

from .joint_pos_env_cfg import (
    SoArm101PickPlaceBowlEnvCfg,
    SoArm101PickPlaceBowlEnvCfg_PLAY,
)
from .pickplace_env_cfg import CurriculumCfg, EventCfg, RewardsCfg


@configclass
class PretrainedRewardsCfg(RewardsCfg):
    """RewardsCfg for §9 cold-start (v5, 2026-05-13).

    **The load-bearing change is dropping `lifting_object` entirely.**

    v3/v4/v4.1 all failed at the same point: the policy reaches a
    "hover-with-grasp at z ≈ 0.08 over the bowl" state and never releases.
    Root cause: ``lifting_object`` (the stock Franka-Lift indicator
    ``block_z > minimal_height``) pays ~15/step indefinitely while the cube
    is lifted. The reward landscape becomes:

        hover at z=0.08:  reach(0.16) + lift(15) + transport(~11.8) ≈ 27/step
        lower to z=0.024: reach(0.16) + lift(0)  + transport(~14.7) ≈ 15/step   ← −12 cliff
        release at z=0.01: reach(0.1) + lift(0)  + transport(~15.5) + release(30) ≈ 46/step

    PPO is myopic about the +30/step release tail and refuses the immediate
    −12/step cliff. Dropping ``lifting_object`` removes the cliff: hover
    pays ~12, lowering pays ~15, release pays ~46 — monotone gradient, and
    PPO can find release through normal exploration.

    The "no independent lift bonus" pattern matches every peer project that
    succeeds at cold-start vision PPO on pick-and-place:

    - **ManiSkill3 PickCube** (StoneT2000 SO-100, 91.6 % zero-shot real):
      no lift term; place gradient pulls cube *toward goal* not *up*.
    - **Robosuite PickPlace**: staged composite, each stage gated on
      previous — no unconditional lift reward.
    - **IsaacGymEnvs FrankaCubeStack**: ``reach + is_grasped + dist_to_target
      + sparse_success`` — no lift term.

    Other changes from v4.1:

    - **``release_in_bowl.minimal_height`` 0.07 → 0.025** — makes the
      release-lift-latch easier to set so from-scratch envs that achieve a
      small lift (≥ 2.5 cm) can unlock the release reward, not just the
      ones that lift past 7 cm. Rim safety is still enforced separately by
      the ``_episode_over_bowl_high_mask`` latch (cube must have been ≥ 8 cm
      *and* over the bowl xy at some prior step).
    - **``closed_grasp_signal`` kept at weight 3.0 with pre-lift gate** —
      it does the same job ManiSkill3's contact-grasp signal does
      (gradient on the close-jaws-on-cube action), kinematic-proxied
      because the SO-ARM101 asset has ``activate_contact_sensors=False``.

    Why we don't need an explicit lift gradient: ``object_goal_tracking``
    is gated on ``_episode_lifted_mask`` (latches at z>0.025). Once any
    lift happens, the transport reward unlocks at ~15/step and pulls the
    cube down toward the bowl position. The first lift comes from
    (a) bootstrap envs that spawn with cube already at z≈0.083, providing
    policy gradient on the post-grasp action chain, and (b) random
    exploration with the closed_grasp gradient steering toward grasp.
    """

    # v8.3 (2026-05-15): lifting_object weight 15 → 0. This is the v5 fix
    # the EVAL1 §9 plan ultimately rejected, but the rejection rationale
    # (v5.1 comment) was specific to bootstrap envs with lift-latch already
    # set at spawn. Bootstrap is disabled in v5.3+ (``p_grasped=0``), so
    # that failure mode no longer applies.
    #
    # The v8.2 hover lock is exactly the v3/v4/v4.1 hover-attractor: the
    # step-function ``z > 0.025`` indicator pays 15/step indefinitely at
    # hover and creates a −15 cliff when the policy attempts to release.
    # With release_proximity weight 16 still capped at 5.3/step max
    # (xy×z×open factor product), the policy strictly prefers hover.
    #
    # Dropping the weight removes the cliff. Lift behavior is now emergent
    # from object_goal_tracking (which uses 3-D distance to goal_xyz with
    # bowl at z≈0, so it pulls the cube z DOWN, not up — but only after
    # the lift latch sets, which requires at least one moment of z>0.025).
    # The lift latch sets via random exploration + is_grasping_contact +
    # ee_descent gradient guiding the gripper into contact + close-jaws
    # near the cube. Same chain that worked in v5 before bootstrap was
    # tried.
    lifting_object = RewTerm(
        func=mdp.grasp_event,
        params={"minimal_height": 0.025},
        weight=0.0,
    )

    # v5.1: lower release thresholds for cold-start accessibility.
    # ``minimal_height=0.025`` (was 0.07 default) — release-lift-latch
    # satisfiable by small lifts, not just full 7 cm lifts.
    # ``rim_clearance=0.04`` (was 0.08 default) — over-bowl-above-rim latch
    # satisfiable by brief lift over bowl, not just full rim clearance.
    # v6 settled-gate relaxation kept (sim-side trick, not load-bearing).
    # v8 (2026-05-15): ``r_safe`` 0.06 → 0.035. The 6 cm gate was generous —
    # the user reported drop position "a few cm off" the bowl center, which
    # corresponds exactly to releases at the edge of the 6 cm disk still
    # earning the full +30/step. 3.5 cm is the real bowl interior radius
    # minus a small margin; xy-fine gradient outside this gate is provided
    # by the new ``release_proximity`` term below. NOTE: this also tightens
    # the ``success_rate`` metric (the latch is set inside this function),
    # so SR numbers will be a real measure of in-bowl placement.
    release_in_bowl = RewTerm(
        func=mdp.release_in_bowl,
        params={
            "minimal_height": 0.025,
            "rim_clearance": 0.04,
            "r_safe": 0.035,
            "block_speed_threshold": 0.5,
        },
        weight=30.0,
    )

    # v8 (2026-05-15): dense xy-fine release gradient. The binary +30 from
    # release_in_bowl only fires inside r_safe=3.5 cm — outside that, the
    # policy has no gradient pulling cube xy toward bowl center. This term
    # uses the pre-existing :func:`mdp.release_proximity` (added but unwired
    # in v6) with the wider catchment r_safe=0.06 m, so partial credit
    # starts decaying linearly from the bowl boundary. Gated on the same
    # lift + over-bowl-high latches as release_in_bowl so it cannot reward
    # a "drag laterally into bowl xy" exploit.
    #
    # v8.2 (2026-05-15): weight 8 → 16. v8.1 saw the policy converge on a
    # stable hover-with-grasp at z≈0.08 (mean_reward 48 plateau, lift 3.0,
    # is_grasping_contact 0.94) and never released. Reward arithmetic of
    # the hover state at v8.1 weights:
    #   hover-grasped:  lift(15) + transport(15) + grasping(4)            = 34/step
    #   open-jaws-low:  lift(15) + transport(15) + grasping(0) + rp(2.6)  = 32.6/step  ← worse
    # The −4 from losing is_grasping_contact + the multiplicative gating
    # of release_proximity (xy × z × open_factor, max ≈ 0.33 mid-descent)
    # meant opening jaws was reward-negative. Bumping weight to 16 makes
    # the equivalent term pay 5.3 at the same factor product, flipping
    # the inequality so PPO sees positive advantage on release.
    release_proximity = RewTerm(
        func=mdp.release_proximity,
        params={
            "r_safe": 0.06,
            "bowl_height": 0.06,
            "gripper_open_value": 1.5,
            "minimal_height": 0.025,
            "rim_clearance": 0.04,
        },
        weight=16.0,
    )

    # v7 (2026-05-15): replace kinematic closed_grasp_signal with the
    # physical ContactSensor-based is_grasping_contact reward. The
    # kinematic proxy paid for "jaws closed near cube" regardless of
    # whether actual physical contact existed — PPO learned the proxy by
    # hovering near cube with jaws partially closed, never committing to
    # a real grasp. The physical signal is unambiguous: 1.0 only when both
    # gripper jaws are actually in contact with the cube. Weight 4.0
    # (slightly higher than v5.4's closed_grasp at 1.5) reflects the
    # cleaner gradient: any time PPO actually grasps it gets the full
    # reward, no partial credit for fake-grasping. This is the
    # ManiSkill3-equivalent ``agent.is_grasping(cube)`` reward — the same
    # mechanism that gets their PickCube to 91.6% real success on SO-100.
    is_grasping_contact = RewTerm(
        func=mdp.is_grasping_contact,
        params={
            "force_threshold": 0.05,
            "require_both_jaws": True,
        },
        weight=4.0,
    )

    # v5.2 (2026-05-13): EE descent pre-grasp shaping. v5.1 fixed the rim wall
    # but ``grasp_from_scratch`` remained pinned at 0 for the full run, just
    # like v3/v4/v4.1/v5 before it. The 3-D ``reach_block`` reward's gradient
    # is xy-dominated and didn't reliably pull EE down to cube-grasping
    # height. ``ee_descent_to_cube`` provides the missing z-descent gradient:
    # near_xy × descended × (cube on table) — pays for lowering the gripper
    # to cube level, the geometric prerequisite for jaws-around-cube grasp.
    # Turns off once the policy succeeds in lifting (gated on block_z <=
    # 0.025 per-step) so it can't create a post-grasp hover attractor.
    # Inspired by ManiSkill3 PickCube's per-link reach decomposition and
    # Robosuite PickPlace's vertical reach shaping.
    ee_descent = RewTerm(
        func=mdp.ee_descent_to_cube,
        params={
            "xy_std": 0.04,
            "z_band": 0.02,
            "cube_half_size": 0.01,
            "minimal_height": 0.025,
        },
        weight=2.0,
    )


@configclass
class PretrainedEventCfg(EventCfg):
    """EventCfg for §9 cold-start — bootstrap-grasp **DISABLED** in v5.3.

    Inherits the §7 reset-event chain (``reset_all``, ``reset_lift_latch``,
    ``reset_rim_latch``, ``reset_block_position``). The ``bootstrap_grasped``
    event is kept registered (so the curriculum can re-enable it via
    parameter overrides in future experiments) but ``p_grasped=0.0`` means
    no envs are bootstrapped — all envs are from-scratch.

    **v5.3 rationale (2026-05-14).** Across v3–v5.2 (~6000 cumulative PPO
    iters), every variant followed the same pattern: bootstrap envs do
    ~80 % of the work (release_bootstrap, grasp_bootstrap, success_rate)
    while ``grasp_from_scratch`` stayed pinned at exactly 0. Init-state
    bootstrap teaches the value function and the *post-grasp* action
    sequence, but PPO's policy gradient is dominated by bootstrap envs
    (they have higher rewards) — so the policy optimizes their behavior,
    and from-scratch envs inherit a policy that doesn't include
    "approach + descend + close-jaws + lift". The descent and
    closed_grasp shaping terms exist but their gradient is statistically
    drowned out.

    v5.3 removes the bootstrap subsidy entirely. PPO must learn the full
    pick-place-release sequence from random initialization. This is the
    cleanest test of the supervisor's claim: "frozen ResNet + correct
    reward = cold-start works". If `grasp_from_scratch` leaves 0 within
    1–2 k iters here, the supervisor was right and bootstrap was the
    optimization-dynamics blocker. If it stays at 0, the cold-start
    problem isn't solvable by reward shaping alone and we need either
    a teacher (§7) or Jump-Start RL.
    """

    bootstrap_grasped = EventTerm(
        func=mdp.init_block_in_gripper,
        mode="reset",
        params={"p_grasped": 0.0},  # v5.3: disabled
    )


@configclass
class PretrainedColdStartCurriculumCfg(CurriculumCfg):
    """CurriculumCfg with §9 cold-start-adapted block-xy expand schedule +
    bootstrap-grasp decay.

    Inherits ``action_rate``, ``joint_vel``, ``log_success`` from the parent
    unchanged.
    """

    # v7 (2026-05-15): revert to v5.4's 15k+60k schedule. v6's slow
    # schedule (50k+200k) backfired — σ collapsed to 0.34 in narrow
    # workspace before grasp could be discovered, because PPO had too
    # much time to over-converge on the easy pre-grasp shaping basin. The
    # faster v5.4 schedule keeps the workspace varying enough to force
    # the policy to keep exploring.
    block_range_expand = CurrTerm(
        func=mdp.expand_block_xy_range,
        params={
            "initial_xy": (0.03, 0.03),
            "final_xy":   (0.07, 0.12),
            "warmup_steps":  15_000,
            "expand_steps":  60_000,
            "event_term_name": "reset_block_position",
        },
    )

    # v5.3 (2026-05-14): decay curriculum kept on the cfg but reduced to a
    # no-op (initial=final=0.0). The bootstrap event is disabled outright
    # in PretrainedEventCfg (p_grasped=0.0); keeping this term in the
    # curriculum costs nothing (it just rewrites the same 0.0 each call)
    # and preserves the TB column `Curriculum/p_grasped_decay/p_grasped`
    # so trajectory plots remain comparable to v3–v5.2 runs.
    p_grasped_decay = CurrTerm(
        func=mdp.decay_p_grasped,
        params={
            "initial":       0.0,
            "final":         0.0,
            "warmup_steps":  5_000,
            "decay_steps":   50_000,
            "event_term_name": "bootstrap_grasped",
        },
    )

    # Bootstrap diagnostic metrics. Splits lift / release success rates by
    # bootstrap status so we can see whether the from-scratch policy is
    # learning grasp, or just riding the bootstrap subsidy. Emits:
    #   Curriculum/log_metrics/p_bootstrapped         — fraction in bootstrap regime
    #   Curriculum/log_metrics/grasp_bootstrap        — lift on bootstrap envs (should be ~1 fast)
    #   Curriculum/log_metrics/grasp_from_scratch     — KEY METRIC; should rise toward 1
    #   Curriculum/log_metrics/release_bootstrap      — release on bootstrap envs
    #   Curriculum/log_metrics/release_from_scratch   — release on from-scratch envs
    log_metrics = CurrTerm(func=mdp.log_bootstrap_metrics, params={})


@configclass
class SoArm101PickPlaceBowlPretrainedEnvCfg(SoArm101PickPlaceBowlEnvCfg):
    """SO-ARM101 pick-and-place env, §9 cold-start variant.

    Three deliberate divergences from the §7 env, all confined to this
    subclass (so §7's verified pipeline runs identically on its own task
    IDs):

    1. Curriculum — see :class:`PretrainedColdStartCurriculumCfg`.
    2. Rewards   — see :class:`PretrainedRewardsCfg`.
    3. Events    — see :class:`PretrainedEventCfg`.
    """

    curriculum: PretrainedColdStartCurriculumCfg = PretrainedColdStartCurriculumCfg()
    rewards: PretrainedRewardsCfg = PretrainedRewardsCfg()
    events: PretrainedEventCfg = PretrainedEventCfg()


@configclass
class SoArm101PickPlaceBowlPretrainedEnvCfg_PLAY(SoArm101PickPlaceBowlEnvCfg_PLAY):
    """Play-mode variant (low-DR, fewer envs) of the §9 env.

    **Play mode disables the bootstrap-grasp event** — we want to visualize
    what the policy does on the real task distribution, not on the
    artificially-easy subset. The lift-threshold and curriculum overrides
    stay (they shape what the policy was trained against; eval should match).
    """

    curriculum: PretrainedColdStartCurriculumCfg = PretrainedColdStartCurriculumCfg()
    rewards: PretrainedRewardsCfg = PretrainedRewardsCfg()
    # Inherit the parent EventCfg — no bootstrap_grasped — so play mode
    # tests the pure task (block randomly placed on the table).
