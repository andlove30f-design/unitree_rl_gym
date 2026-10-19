"""Headless end-to-end YBT check: physics, PPO, checkpoint, and policy export.

Works for any registered YBT task:

    python legged_gym/scripts/check_ybt.py --task=ybt --num_envs=32 --max_iterations=2
    python legged_gym/scripts/check_ybt.py --task=ybt_arm --num_envs=64 --max_iterations=3
    python legged_gym/scripts/check_ybt.py --task=ybt_arm_stand --num_envs=64 --max_iterations=3
    python legged_gym/scripts/check_ybt.py --task=ybt_arm_walk --num_envs=64 --max_iterations=3

On the scripted-arm tasks (`ybt_arm_stand`, `ybt_arm_walk`) the arm's joints are also at
action scale 0, but their PD reference is moving, so "does it hold its default pose" is
the wrong question there: the scripted joints are instead checked against `arm_target`,
the reference the script is asking for, and the run doubles as the measurement of how
hard the swing actually pushes the robot (the trunk tilt reported at the end).  The
policy is not exercised -- the settle loop steps with zero actions -- so the walk task
is measured standing still with the arm working, not walking.

Artifacts go to logs/<task>_smoke, never to the production flat_ybt experiment.
"""
import copy
import math
import sys
from pathlib import Path

import isaacgym  # must precede torch
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.utils import export_policy_as_jit, get_args

# How far a frozen joint may drift from its default angle, and how fast it may move,
# while the environment steps with zero actions. Both failure modes seen so far are
# well clear of these: a joint that has lost its PD gains goes limp and swings to its
# limit (0.6 rad and 11 rad/s), and a joint with no inertia rings instead of holding
# (3.9 rad/s). A healthy hold measures 0.04 rad and 0.4 rad/s.
FROZEN_ARM_DRIFT_LIMIT = 0.2   # rad
FROZEN_ARM_SPEED_LIMIT = 2.0   # rad/s

# The scripted-arm task drives those joints instead of holding them, so it is judged on
# how well they follow the reference and on the swing actually happening.
#
# The bar is set by what the arm can do, measured, not by what would be nice. The
# reference is re-aimed every 20 ms under a swing whose peak rate is 3.9 rad/s, so the
# joints chase a staircase: a settled run tracks it to 0.016 rad on average and 0.04 rad
# at its worst, and the landing impact at the start of the run kicks one joint to 0.1 for
# a few steps. Nothing tracks better than that at any gain -- four times the stiffness
# halves the error and doubles the effort, a quarter of the damping removes the arm's
# 19 Hz ring but lets the error grow. What the limit is for is catching the arm *losing*
# its reference (>0.5 rad, what the additive-bias bug produced: 4.2 rad).
TRACKING_LIMIT = 0.15     # rad, scripted joint vs its reference
TRAVEL_FRACTION = 0.3     # of the commanded swing a joint has to actually cover

# Physics steps to step with zero actions. The robot spawns above its stance height
# and bounces when it lands (a peak around step 30, quiet by about step 70), so the
# effort is reported over the last 20 of these -- 100 steps was not enough to be
# settled, and reported the tail of the bounce as if it were the resting state.
SETTLE_STEPS = 150
SETTLED_WINDOW = 20


def check():
    if not any(a == "--task" or a.startswith("--task=") for a in sys.argv[1:]):
        sys.argv.append("--task=ybt")
    args = get_args()
    if args.task not in task_registry.task_classes:
        raise ValueError("Unknown task {!r}; registered: {}".format(
            args.task, sorted(task_registry.task_classes)))
    if args.resume:
        raise ValueError("Run the checker without --resume; it creates its own test checkpoint")
    args.headless = True
    if args.num_envs is None:
        args.num_envs = 64
    if args.max_iterations is None:
        args.max_iterations = 3
    if args.num_envs < 1 or args.max_iterations < 1:
        raise ValueError("num_envs and max_iterations must be positive")

    # Deep copies: the registry hands out the live config objects.
    cfg = copy.deepcopy(task_registry.get_cfgs(args.task)[0])
    cfg.seed = 1
    num_obs, num_actions = cfg.env.num_observations, cfg.env.num_actions
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    try:
        # A scripted-arm task starts its script at amplitude 0 (the folded pose) and
        # only the training loop ever grows it, so pin it at full amplitude here --
        # otherwise this checks a motionless arm that nobody will ever see in play.
        if hasattr(env, "set_arm_scale"):
            env.set_arm_scale(cfg.arm_trajectory.curriculum_max)
            print("Scripted arm: amplitude scale {:.2f}, {:.2f} Hz cycle".format(
                env.arm_scale, cfg.arm_trajectory.frequency))
        ids = torch.arange(env.num_envs, device=env.device)
        env.reset_idx(ids)
        if not torch.all((env.dof_pos >= env.dof_pos_limits[:, 0])
                         & (env.dof_pos <= env.dof_pos_limits[:, 1])):
            raise AssertionError("Reset positions violate joint limits")
        obs, _ = env.reset()
        if obs.shape != (env.num_envs, num_obs) or env.feet_indices.numel() != 4:
            raise AssertionError("Unexpected observation shape or foot count")
        print("Loaded DOF/action order:", env.dof_names)
        print("Feet:", env.feet_indices.tolist(), "observation shape:", tuple(obs.shape))

        # Joints the task froze (action scale 0): they must hold their pose no matter
        # what the policy outputs -- this is the only place that checks it.  On the
        # scripted-arm tasks the same joints are being driven, so what they have
        # to hold is the script's reference, not the default pose, and they belong in
        # the scripted check below instead.
        frozen = [i for i, name in enumerate(env.dof_names)
                  if env.action_scale[i].item() == 0.0]
        scripted = sorted(set(frozen) & set(getattr(env, "arm_idx", [])))
        held = [i for i in frozen if i not in scripted]
        arm_cols = [env.arm_idx.index(i) for i in scripted] if scripted else []
        if frozen:
            print("Frozen joints:", [env.dof_names[i] for i in frozen])
        if scripted:
            print("  of which script-driven:",
                  [env.dof_names[i] for i in scripted])
        torque_peak = torch.zeros(env.num_dof, device=env.device)
        torque_settled = torch.zeros(env.num_dof, device=env.device)
        arm_low = env.dof_pos[:, scripted].clone() if scripted else None
        arm_high = arm_low.clone() if scripted else None
        settled_tilt = 0.0
        steps = SETTLE_STEPS
        for step in range(steps):
            obs, _, rewards, _, _ = env.step(torch.zeros(env.num_envs, num_actions, device=env.device))
            for tensor in (obs, rewards, env.root_states, env.dof_state, env.contact_forces):
                if not torch.isfinite(tensor).all():
                    raise AssertionError("Physics produced non-finite state/rewards")
            if not torch.all(env.torques.abs() <= env.torque_limits + 1e-5):
                raise AssertionError("Applied torque exceeds the URDF effort limit")
            torque_peak = torch.maximum(torque_peak, env.torques.abs().max(dim=0).values)
            if scripted:
                arm_low = torch.minimum(arm_low, env.dof_pos[:, scripted])
                arm_high = torch.maximum(arm_high, env.dof_pos[:, scripted])
            # the robot is dropped from its spawn height and lands; that impact spikes
            # every joint, so report the settled effort separately from the peak
            if step >= steps - SETTLED_WINDOW:
                torque_settled = torch.maximum(
                    torque_settled, env.torques.abs().max(dim=0).values)
                # the disturbance measurement: how far the swing actually tilts the
                # trunk, in the same projected-gravity units the orientation reward uses
                settled_tilt += env.projected_gravity[:, :2].pow(2).sum(dim=1).mean().item()
        if held:
            drift = (env.dof_pos[:, held] - env.default_dof_pos[:, held]).abs().max().item()
            speed = env.dof_vel[:, held].abs().max().item()
            print("Frozen-joint hold: max drift {:.4f} rad, max speed {:.4f} rad/s".format(
                drift, speed))
            failures = []
            if drift > FROZEN_ARM_DRIFT_LIMIT:
                failures.append("drifted {:.3f} rad off the default pose".format(drift))
            if speed > FROZEN_ARM_SPEED_LIMIT:
                failures.append("kept moving at {:.3f} rad/s".format(speed))
            if failures:
                raise AssertionError(
                    "Frozen joints {} {} -- are their PD gains defined, and do their "
                    "DOFs have any inertia?".format(
                        [env.dof_names[i] for i in held], " and ".join(failures)))
        if scripted:
            error = (env.dof_pos[:, scripted] - env.arm_target[:, arm_cols]).abs().max().item()
            swing = env.arm_swing[arm_cols].abs()
            # per joint, the range any environment reached: the envs sit at different
            # phases, so no single one of them has to cover the whole cycle
            travel = arm_high.amax(dim=0) - arm_low.amin(dim=0)
            effort = (torque_settled[scripted] / env.torque_limits[scripted]).max().item()
            # Joints the script barely moves (j1, j4) have no travel to check; the ones
            # that do must have moved.  A script whose phase never advances would
            # otherwise pass a tracking check perfectly while doing nothing at all.
            moved = swing > 0.05
            covered = (travel[moved] / swing[moved]).min().item() if torch.any(moved) else 1.0
            print("Script-driven tracking: max error {:.4f} rad, travel {:.0%} of the "
                  "commanded swing, heaviest scripted joint {:.0%} of its effort limit".format(
                      error, covered, effort))
            print("Swing disturbance: trunk tilt {:.2f} deg RMS, standing with zero "
                  "actions".format(math.degrees(math.sqrt(
                      settled_tilt / min(SETTLED_WINDOW, steps)))))
            if covered < TRAVEL_FRACTION:
                raise AssertionError(
                    "the scripted joints covered only {:.0%} of the commanded swing; the "
                    "arm is not following its reference at all".format(covered))
            if error > TRACKING_LIMIT:
                raise AssertionError(
                    "the scripted joints lag their reference by {:.3f} rad; raise the arm "
                    "stiffness or lower cfg.arm_trajectory.frequency".format(error))
        use = torque_settled / env.torque_limits
        worst = torch.argsort(use, descending=True)[:4]
        print("Idle stand: trunk height {:.3f} m (reward target {:.3f}), heaviest joints "
              "(settled, last {} steps): {}".format(
                  env.root_states[:, 2].mean().item(), cfg.rewards.base_height_target,
                  SETTLED_WINDOW,
                  ", ".join("{} {:.0%}".format(env.dof_names[i], use[i]) for i in worst)))
        peak = torque_peak / env.torque_limits
        print("Landing impact: peaks at {:.0%} ({}) in the {} steps after reset".format(
            peak.max(), env.dof_names[int(peak.argmax())], steps))

        train_cfg = copy.deepcopy(task_registry.get_cfgs(args.task)[1])
        experiment = "{}_smoke".format(args.task)
        train_cfg.runner.experiment_name = experiment
        train_cfg.runner.run_name = "check{}".format(env.num_envs)
        # Ignore experiment/resume CLI overrides: never load/write a production run.
        args.experiment_name = experiment
        args.run_name = train_cfg.runner.run_name
        runner, _ = task_registry.make_alg_runner(
            env, args=args, train_cfg=train_cfg,
            log_root=str(Path(LEGGED_GYM_ROOT_DIR) / "logs" / experiment))
        before = {key: tensor.detach().clone() for key, tensor in runner.alg.actor_critic.actor.state_dict().items()}
        runner.learn(args.max_iterations, init_at_random_ep_len=True)
        actor = runner.alg.actor_critic.actor
        if not any(not torch.equal(before[key], tensor) for key, tensor in actor.state_dict().items()):
            raise AssertionError("PPO did not update actor weights")
        if not all(torch.isfinite(tensor).all() for tensor in runner.alg.actor_critic.parameters()):
            raise AssertionError("PPO produced non-finite weights")
        checkpoint = Path(runner.log_dir) / "model_{}.pt".format(runner.current_learning_iteration)
        runner.load(str(checkpoint))
        # The runner steps the environment under inference_mode. Continue
        # in that mode because its action buffers are inference tensors.
        with torch.inference_mode():
            obs, _ = env.reset()
            actions = runner.get_inference_policy(device=env.device)(obs)
        if actions.shape != (env.num_envs, num_actions) or not torch.isfinite(actions).all():
            raise AssertionError("Reloaded policy produced invalid actions")
        export_dir = Path(runner.log_dir) / "exported" / "policies"
        export_policy_as_jit(runner.alg.actor_critic, str(export_dir))
        exported = torch.jit.load(str(export_dir / "policy_1.pt"), map_location="cpu")
        with torch.no_grad():
            exported_actions = exported(obs[:4].cpu())
        if not torch.allclose(exported_actions, actions[:4].cpu(), atol=1e-4, rtol=1e-4):
            raise AssertionError("Exported actor differs from checkpoint inference")
        for _ in range(50):
            with torch.inference_mode():
                actions = runner.get_inference_policy(device=env.device)(obs)
                obs, _, rewards, _, _ = env.step(actions)
            if not torch.isfinite(obs).all() or not torch.isfinite(rewards).all():
                raise AssertionError("Reloaded policy rollout produced non-finite values")
        print("PASS: task={} {} environments; physics + {} PPO iterations + save/reload + TorchScript rollout".format(
            args.task, env.num_envs, args.max_iterations))
        print("Checkpoint:", checkpoint)
        print("Export:", export_dir / "policy_1.pt")
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    check()
