#!/usr/bin/env python3
"""
Train a pure RL policy for compliant behavior.

No force estimator, no velocity commands from admittance.
The robot directly receives the applied force as part of observation and learns
to track a velocity proportional to that force.

Reward: Track velocity = admittance_gain * applied_force

This is the simplest approach - let RL figure out the compliance behavior end-to-end.

Usage:
    python train_pure_rl_compliance.py \
        --admittance-gains 0.1,0.1 \
        --num-timesteps 500_000_000 \
        --num-envs 4096
"""

import argparse
import functools
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

# Set EGL for headless rendering BEFORE importing mujoco
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
from jax import numpy as jp
import numpy as np
from ml_collections import config_dict

from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import mjcf
from orbax import checkpoint as ocp

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from pupperv3_mjx import domain_randomization, utils
from pupperv3_mjx.environment import PupperV3Env
from brax.envs.base import State
from brax import math


class PupperV3EnvPureCompliance(PupperV3Env):
    """
    Pure RL environment for compliance training.
    
    NO force in observation - robot must learn compliance from physical effects only.
    Velocity command slot filled with NOISE (to match previous policy's observation size).
    
    Observation (matches joint training actor observation):
    - IMU data (angular velocity, gravity) - 6 dims
    - Velocity command (NOISE - masked out) - 3 dims
    - Motor angles - 12 dims  
    - Last action - 12 dims
    Total: 33 dims per frame × 20 frames = 660 dims
    
    Reward: Move at velocity proportional to applied force
    - desired_velocity = admittance_gain * force (computed internally for reward only)
    - Robot never sees the real velocity command (just noise)
    - Robot must learn to "feel" the push from IMU/joint changes and respond
    """
    
    def __init__(
        self,
        admittance_gains: Tuple[float, float] = (0.1, 0.1),
        **kwargs
    ):
        super().__init__(**kwargs)
        self._admittance_gains = jp.array(admittance_gains)
        
        # Observation dim: 33 (matches joint training for checkpoint compatibility)
        # IMU (6) + command_noise (3) + motor angles (12) + last action (12) = 33
        self.observation_dim = 33
        
        print(f"Pure RL Compliance Environment")
        print(f"  Admittance gains: x={admittance_gains[0]}, y={admittance_gains[1]} m/s per N")
        print(f"  Observation dim: {self.observation_dim} (command slot = NOISE)")
        print(f"  Robot learns compliance purely from physical feedback!")
    
    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        # Command is used internally for reward computation only
        # Robot never sees it in observation
        new_info = {**state.info}
        new_info['command'] = jp.zeros(3)
        return state.replace(info=new_info)
    
    def step(self, state: State, action: jax.Array) -> State:
        """Step with force-based reward (but noise in command observation)."""
        
        # Get the current force (world frame)
        force_world = state.info['force_current_vector']
        
        # Convert to body frame
        body_rotation = state.pipeline_state.x.rot[self._torso_idx]
        force_body = math.rotate(force_world, math.quat_inv(body_rotation))
        
        # Compute TARGET velocity from force via admittance
        # This is used ONLY for reward computation - robot sees NOISE instead!
        target_vel_x = self._admittance_gains[0] * force_body[0]
        target_vel_y = self._admittance_gains[1] * force_body[1]
        target_velocity = jp.array([target_vel_x, target_vel_y, 0.0])
        
        # Clip to reasonable range
        target_velocity = jp.clip(
            target_velocity,
            jp.array([self._linear_velocity_x_range[0], self._linear_velocity_y_range[0], -2.0]),
            jp.array([self._linear_velocity_x_range[1], self._linear_velocity_y_range[1], 2.0])
        )
        
        # Set as "command" for reward computation (tracking_lin_vel uses this)
        # But robot sees NOISE in observation, not this!
        new_info = {**state.info}
        new_info['command'] = target_velocity
        state = state.replace(info=new_info)
        
        # Call parent step (computes tracking_lin_vel reward)
        return super().step(state, action)
    
    def _get_obs(
        self,
        pipeline_state,
        state_info: dict,
        obs_history: jax.Array,
    ) -> jax.Array:
        """Get observation with NOISE in command slot (to match joint training obs size).
        
        Observation (33 dims per frame):
        - IMU data (angular velocity, gravity) - 6 dims
        - Command slot (PURE NOISE) - 3 dims  <-- Robot must ignore this!
        - Motor angles - 12 dims
        - Last action - 12 dims
        """
        if self._use_imu:
            inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
            local_body_angular_velocity = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)
        else:
            inv_torso_rot = jp.array([1, 0, 0, 0])
            local_body_angular_velocity = jp.zeros(3)

        # Noise keys
        (
            state_info["rng"],
            ang_key,
            gravity_key,
            motor_angle_key,
            last_action_key,
            imu_sample_key,
            cmd_noise_key,  # New key for command noise
        ) = jax.random.split(state_info["rng"], 7)

        ang_vel_noise = (
            jax.random.uniform(ang_key, (3,), minval=-1, maxval=1) * self._angular_velocity_noise
        )
        gravity_noise = (
            jax.random.uniform(gravity_key, (3,), minval=-1, maxval=1) * self._gravity_noise
        )
        motor_ang_noise = (
            jax.random.uniform(motor_angle_key, (12,), minval=-1, maxval=1)
            * self._motor_angle_noise
        )
        last_action_noise = (
            jax.random.uniform(last_action_key, (12,), minval=-1, maxval=1)
            * self._last_action_noise
        )
        
        # PURE NOISE for command slot - robot must learn to ignore this
        # Range matches typical velocity commands: [-0.75, 0.75] for x, [-0.5, 0.5] for y
        command_noise = jax.random.uniform(
            cmd_noise_key, (3,), 
            minval=jp.array([-0.75, -0.5, -2.0]),
            maxval=jp.array([0.75, 0.5, 2.0])
        )

        noised_gravity = math.rotate(jp.array([0, 0, -1]), inv_torso_rot) + gravity_noise
        noised_gravity = noised_gravity / (jp.linalg.norm(noised_gravity) + 1e-6)
        noised_ang_vel = local_body_angular_velocity + ang_vel_noise
        noised_imu_data = jp.concatenate([noised_ang_vel, noised_gravity])

        lagged_imu_data, state_info["imu_buffer"] = utils.sample_lagged_value(
            imu_sample_key,
            state_info["imu_buffer"],
            noised_imu_data,
            self._imu_latency_distribution,
        )

        # Construct observation with NOISE in command slot
        obs = jp.concatenate(
            [
                lagged_imu_data,  # 6 dims: angular velocity + gravity
                command_noise,  # 3 dims: PURE NOISE (robot must ignore!)
                pipeline_state.q[7:] - self._default_pose + motor_ang_noise,  # 12 dims: motor angles
                state_info["last_act"] + last_action_noise,  # 12 dims: last action
            ]
        )
        # Total: 6 + 3 + 12 + 12 = 33 dims per frame

        obs = jp.clip(obs, -100.0, 100.0)

        # Stack observations through time
        new_obs_history = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        return new_obs_history


# ============================================================================
# Configs
# ============================================================================

def get_reward_config():
    config = config_dict.ConfigDict()
    config.rewards = config_dict.ConfigDict()
    config.rewards.scales = config_dict.ConfigDict()
    
    # High weight on velocity tracking - this is the main objective!
    config.rewards.scales.tracking_lin_vel = 3.0
    config.rewards.scales.tracking_ang_vel = 0.0  # No yaw tracking
    
    # Standard locomotion rewards
    config.rewards.scales.tracking_orientation = 1.0
    config.rewards.scales.lin_vel_z = -2.0
    config.rewards.scales.ang_vel_xy = -0.05
    config.rewards.scales.orientation = -5.0
    config.rewards.scales.torques = -0.0002
    config.rewards.scales.joint_acceleration = -1e-6
    config.rewards.scales.mechanical_work = 0.0
    config.rewards.scales.action_rate = -0.01
    config.rewards.scales.feet_air_time = 0.05
    config.rewards.scales.stand_still = -0.5
    config.rewards.scales.stand_still_joint_velocity = -0.1
    config.rewards.scales.abduction_angle = -0.1
    config.rewards.scales.termination = -100.0
    config.rewards.scales.foot_slip = -0.1
    config.rewards.scales.knee_collision = -1.0
    config.rewards.scales.body_collision = -1.0
    config.rewards.scales.force_following = 0.0  # Not using this
    config.rewards.tracking_sigma = 0.25
    
    return config


def get_simulation_config(model_path: str):
    config = config_dict.ConfigDict()
    config.model_path = model_path
    
    sys = mjcf.load(config.model_path)
    config.joint_upper_limits = np.array(sys.mj_model.jnt_range[1:, 1]).tolist()
    config.joint_lower_limits = np.array(sys.mj_model.jnt_range[1:, 0]).tolist()
    
    config.foot_site_names = [
        "leg_front_r_3_foot_site",
        "leg_front_l_3_foot_site",
        "leg_back_r_3_foot_site",
        "leg_back_l_3_foot_site",
    ]
    config.torso_name = "base_link"
    config.upper_leg_body_names = ["leg_front_r_2", "leg_front_l_2", "leg_back_r_2", "leg_back_l_2"]
    config.lower_leg_body_names = ["leg_front_r_3", "leg_front_l_3", "leg_back_r_3", "leg_back_l_3"]
    config.foot_radius = 0.02
    config.physics_dt = 0.004
    
    return config


def get_training_config(args):
    config = config_dict.ConfigDict()
    config.environment_dt = 0.02
    
    config.ppo = config_dict.ConfigDict()
    config.ppo.num_timesteps = args.num_timesteps
    config.ppo.episode_length = 1000
    config.ppo.num_evals = 3  # Eval every 3 times
    config.ppo.reward_scaling = 1
    config.ppo.normalize_observations = True
    config.ppo.action_repeat = 1
    config.ppo.unroll_length = 20
    config.ppo.num_minibatches = 32
    config.ppo.num_updates_per_batch = 4
    config.ppo.discounting = 0.97
    config.ppo.learning_rate = 3.0e-4
    config.ppo.entropy_cost = 1e-2
    config.ppo.num_envs = args.num_envs
    config.ppo.batch_size = 256
    
    config.lin_vel_x_range = [-0.75, 0.75]
    config.lin_vel_y_range = [-0.5, 0.5]
    config.ang_vel_yaw_range = [-2.0, 2.0]
    config.zero_command_probability = 0.0
    config.stand_still_command_threshold = 0.05
    
    config.policy = config_dict.ConfigDict()
    config.policy.hidden_layer_sizes = (256, 128, 128, 128)
    config.policy.activation = "elu"
    
    return config


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train pure RL compliance policy")
    
    # Paths
    parser.add_argument("--model-path", type=str, 
                        default="../pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.no_body.self_collision.two_iterations.xml",
                        help="Path to MuJoCo model")
    parser.add_argument("--output-dir", type=str, default="output_pure_rl_compliance",
                        help="Output directory prefix")
    
    # Training
    parser.add_argument("--num-timesteps", type=int, default=500_000_000,
                        help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=4096,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    # Compliance
    parser.add_argument("--admittance-gains", type=str, default="0.1,0.1",
                        help="Admittance gains (x,y) in m/s per N")
    
    # Force parameters
    parser.add_argument("--force-magnitude-min", type=float, default=2.0,
                        help="Minimum force magnitude (N)")
    parser.add_argument("--force-magnitude-max", type=float, default=6.0,
                        help="Maximum force magnitude (N)")
    parser.add_argument("--force-probability", type=float, default=0.8,
                        help="Probability of force being active")
    
    # Checkpoint
    parser.add_argument("--restore-checkpoint", type=str, default=None,
                        help="Path to checkpoint to restore from")
    
    # Logging
    parser.add_argument("--wandb-key", type=str, default=None, help="WandB API key")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")
    parser.add_argument("--wandb-project", type=str, default="pupperv3-pure-rl-compliance",
                        help="WandB project name")
    parser.add_argument("--no-video", action="store_true", help="Disable video rendering")
    
    args = parser.parse_args()
    
    # Parse admittance gains
    admittance_gains = tuple(float(x) for x in args.admittance_gains.split(","))
    
    # Setup output directory
    train_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = Path(f"{args.output_dir}_{train_datetime}").resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Pure RL Compliance Training")
    print("=" * 70)
    print(f"Admittance gains: x={admittance_gains[0]}, y={admittance_gains[1]} m/s per N")
    print(f"Force range: {args.force_magnitude_min} - {args.force_magnitude_max} N")
    print(f"Num timesteps: {args.num_timesteps:,}")
    print(f"Num envs: {args.num_envs}")
    print(f"Output: {output_folder}")
    print("=" * 70)
    
    # Initialize WandB
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        try:
            wandb.init(
                project=args.wandb_project,
                config={
                    "admittance_gains": admittance_gains,
                    "force_magnitude_range": [args.force_magnitude_min, args.force_magnitude_max],
                    "num_timesteps": args.num_timesteps,
                    "num_envs": args.num_envs,
                }
            )
            print("WandB initialized")
        except Exception as e:
            print(f"WandB init failed: {e}")
            use_wandb = False
    else:
        print("Training without WandB logging")
    
    # Load configs
    reward_config = get_reward_config()
    sim_config = get_simulation_config(args.model_path)
    training_config = get_training_config(args)
    
    # Create environment factory
    def create_env(**kwargs):
        return PupperV3EnvPureCompliance(
            path=args.model_path,
            reward_config=reward_config,
            action_scale=0.3,
            observation_history=20,  # 20 frames × 33 dims = 660 dim observation
            joint_lower_limits=sim_config.joint_lower_limits,
            joint_upper_limits=sim_config.joint_upper_limits,
            torso_name=sim_config.torso_name,
            foot_site_names=sim_config.foot_site_names,
            upper_leg_body_names=sim_config.upper_leg_body_names,
            lower_leg_body_names=sim_config.lower_leg_body_names,
            linear_velocity_x_range=training_config.lin_vel_x_range,
            linear_velocity_y_range=training_config.lin_vel_y_range,
            angular_velocity_range=training_config.ang_vel_yaw_range,
            zero_command_probability=training_config.zero_command_probability,
            stand_still_command_threshold=training_config.stand_still_command_threshold,
            environment_timestep=training_config.environment_dt,
            physics_timestep=sim_config.physics_dt,
            foot_radius=sim_config.foot_radius,
            force_probability=args.force_probability,
            force_magnitude_range=jp.array([args.force_magnitude_min, args.force_magnitude_max]),
            admittance_gains=admittance_gains,
            **kwargs
        )
    
    env = create_env()
    eval_env = create_env()
    
    print(f"\nEnvironment created:")
    print(f"  Observation size: {env.observation_size} (33 dims × 20 frames = 660)")
    print(f"  Action size: {env.action_size}")
    print(f"  Command slot: PURE NOISE (robot must learn to ignore)")
    print(f"  Robot learns compliance from: IMU, joint angles, action history")
    
    # PPO setup
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=training_config.policy.hidden_layer_sizes,
        activation=utils.activation_fn_map(training_config.policy.activation)
    )
    
    train_fn = functools.partial(
        ppo.train,
        **training_config.ppo.to_dict(),
        network_factory=make_networks_factory,
        randomization_fn=functools.partial(
            domain_randomization.domain_randomize,
            friction_range=(0.6, 1.4),
            body_mass_scale_range=(0.8, 1.2),
            kp_multiplier_range=(0.85, 1.15),
            kd_multiplier_range=(0.85, 1.15),
        ),
        seed=args.seed,
    )
    
    # Progress function
    def progress_fn(num_steps, metrics):
        if 'eval/episode_reward' in metrics:
            reward = metrics['eval/episode_reward']
            reward_std = metrics.get('eval/episode_reward_std', 0)
            print(f"  Step {num_steps:,} | Reward: {reward:.2f} ± {reward_std:.2f}")
            
            if use_wandb:
                try:
                    wandb.log({
                        "reward": reward,
                        "reward_std": reward_std,
                        "steps": num_steps,
                    }, step=num_steps)
                except:
                    pass
    
    # Track full params for checkpoint saving
    latest_full_params = [None]
    
    # Video rendering callback
    def policy_params_fn(current_step, make_policy, params):
        latest_full_params[0] = params
        
        if args.no_video:
            return
        try:
            video_env = create_env()
            video_jit_reset = jax.jit(video_env.reset)
            video_jit_step = jax.jit(video_env.step)
            
            utils.visualize_policy(
                current_step=current_step,
                make_policy=make_policy,
                params=params,
                eval_env=video_env,
                jit_step=video_jit_step,
                jit_reset=video_jit_reset,
                output_folder=str(output_folder)
            )
        except Exception as e:
            print(f"  Video rendering failed: {e}")
    
    # Checkpoint loading
    checkpoint_kwargs = {}
    if args.restore_checkpoint:
        checkpoint_path = Path(args.restore_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if checkpoint_path.exists():
            print(f"Restoring from checkpoint: {checkpoint_path}")
            checkpoint_kwargs["restore_checkpoint_path"] = checkpoint_path
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
    
    # Train!
    print("\nStarting training...")
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        progress_fn=progress_fn,
        eval_env=eval_env,
        policy_params_fn=policy_params_fn,
        **checkpoint_kwargs
    )
    
    # Save final checkpoint
    checkpoint_path = (output_folder / "final_checkpoint").resolve()
    if latest_full_params[0] is not None:
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        orbax_checkpointer.save(
            str(checkpoint_path),
            latest_full_params[0],
            force=True
        )
        print(f"\nSaved checkpoint to: {checkpoint_path}")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

