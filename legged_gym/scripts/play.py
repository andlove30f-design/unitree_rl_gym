import math
import os
import sys

from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def apply_fixed_command(env, args):
    """Pin the velocity commands instead of re-randomizing them every 10 s.

    The environment re-randomizes commands on a timer and on every reset, so the
    one place that writes them is overridden rather than setting ``env.commands``
    once -- that would be undone a few hundred steps later.

    There are two ways to ask for a turn and they need different plumbing,
    because a *target heading* is turned into a yaw rate by the environment
    itself (see ``LeggedRobot._post_physics_step_callback``):

    ``--cmd_wz``      a fixed yaw rate. The heading loop rewrites ``commands[:,2]``
                      from the heading error every single step, so it has to be
                      switched off or the value never survives.
    ``--cmd_heading``  a fixed target heading, in degrees. The heading loop is
                      what implements this, so it is forced on and the yaw rate
                      is left to the environment.
    """
    if args.cmd_heading is not None and args.cmd_wz is not None:
        sys.exit("--cmd_heading and --cmd_wz cannot be combined: the environment "
                 "derives the yaw rate from the heading error, so a fixed yaw rate "
                 "would be overwritten every step. Pass one or the other.")

    if not any(c is not None for c in
               (args.cmd_vx, args.cmd_vy, args.cmd_wz, args.cmd_heading)):
        return

    fixed = torch.tensor(
        [0.0 if c is None else c for c in (args.cmd_vx, args.cmd_vy, args.cmd_wz)],
        device=env.device)
    # ``commands[:,3]`` is the *target heading*, and it starts out randomized over
    # +-pi. It has to be pinned as well: the heading loop reads it every step to
    # derive the yaw rate, so leaving it alone makes the robot walk off toward
    # whatever direction it was seeded with. Holding 0 means "keep facing the way
    # you started", which is what a bare --cmd_vx is asking for.
    heading = 0.0 if args.cmd_heading is None else math.radians(args.cmd_heading)

    def _resample_commands(env_ids, _cmd=fixed, _heading=heading):
        if len(env_ids):
            env.commands[env_ids, :3] = _cmd
            env.commands[env_ids, 3] = _heading

    env._resample_commands = _resample_commands
    env.commands[:, :3] = fixed
    env.commands[:, 3] = heading

    if args.cmd_wz is not None:
        # a fixed yaw *rate* only survives with the heading loop out of the way
        env.cfg.commands.heading_command = False
    else:
        # a fixed target heading is implemented *by* the heading loop
        env.cfg.commands.heading_command = True

    print("Holding the command fixed at vx=%+.2f vy=%+.2f wz=%+.2f"
          % tuple(fixed.tolist()))
    if args.cmd_heading is not None:
        print("  target heading %+.1f deg (%+.3f rad): the robot turns to it and holds"
              % (args.cmd_heading, heading))
    elif args.cmd_wz is None:
        print("  target heading 0.0 deg: the robot holds the direction it started in")
    if args.cmd_wz is not None:
        print("  heading_command switched off, otherwise it would overwrite wz "
              "on every step")


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # How fast the scripted arm works.  The environment reads this every policy step,
    # so overriding it here is enough -- no need to change the task.  It is a stress
    # knob, not a faithful one: at the shipped 0.5 Hz the swing already peaks around
    # 3.2 rad/s, past the RM65-B's 3.14 rad/s joint limit, and the inertial reaction
    # grows with frequency squared.
    if getattr(args, "arm_freq", None) is not None:
        if not hasattr(env_cfg, "arm_trajectory"):
            sys.exit("--arm_freq only applies to a scripted-arm task (ybt_arm_stand)")
        if args.arm_freq <= 0.0:
            sys.exit("--arm_freq must be positive")
        print("Arm cycle: {:g} Hz (the task ships {:.2f} Hz)".format(
            args.arm_freq, env_cfg.arm_trajectory.frequency))
        if args.arm_freq > 0.5:
            print("  note: past the arm's 3.14 rad/s joint limit -- a stress test for "
                  "the legs, not a motion the real arm could make")
        env_cfg.arm_trajectory.frequency = args.arm_freq

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Optionally hold the velocity command fixed instead of leaving it random.
    apply_fixed_command(env, args)

    # A scripted-arm task starts its script at amplitude 0 and only the training loop
    # ever grows it, so without this the demo would show a motionless arm. The trained
    # amplitude is the curriculum's ceiling; nothing persists it in the checkpoint.
    if hasattr(env, "set_arm_scale"):
        env.set_arm_scale(env_cfg.arm_trajectory.curriculum_max)
        print("Scripted arm: amplitude scale {:.2f}".format(env.arm_scale))

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
