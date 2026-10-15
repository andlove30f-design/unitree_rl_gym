"""Headless end-to-end YBT check: physics, PPO, checkpoint, and policy export.

Works for any registered YBT task:

    python legged_gym/scripts/check_ybt.py --task=ybt --num_envs=32 --max_iterations=2
    python legged_gym/scripts/check_ybt.py --task=ybt_arm --num_envs=64 --max_iterations=3

Artifacts go to logs/<task>_smoke, never to the production flat_ybt experiment.
"""
import copy
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

        # Joints the task froze (action scale 0): they must hold their default pose
        # no matter what the policy outputs -- this is the only place that checks it.
        frozen = [i for i, name in enumerate(env.dof_names)
                  if env.action_scale[i].item() == 0.0]
        if frozen:
            print("Frozen joints:", [env.dof_names[i] for i in frozen])
        torque_peak = torch.zeros(env.num_dof, device=env.device)
        torque_settled = torch.zeros(env.num_dof, device=env.device)
        steps = SETTLE_STEPS
        for step in range(steps):
            obs, _, rewards, _, _ = env.step(torch.zeros(env.num_envs, num_actions, device=env.device))
            for tensor in (obs, rewards, env.root_states, env.dof_state, env.contact_forces):
                if not torch.isfinite(tensor).all():
                    raise AssertionError("Physics produced non-finite state/rewards")
            if not torch.all(env.torques.abs() <= env.torque_limits + 1e-5):
                raise AssertionError("Applied torque exceeds the URDF effort limit")
            torque_peak = torch.maximum(torque_peak, env.torques.abs().max(dim=0).values)
            # the robot is dropped from its spawn height and lands; that impact spikes
            # every joint, so report the settled effort separately from the peak
            if step >= steps - SETTLED_WINDOW:
                torque_settled = torch.maximum(
                    torque_settled, env.torques.abs().max(dim=0).values)
        if frozen:
            drift = (env.dof_pos[:, frozen] - env.default_dof_pos[:, frozen]).abs().max().item()
            speed = env.dof_vel[:, frozen].abs().max().item()
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
                        [env.dof_names[i] for i in frozen], " and ".join(failures)))
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
