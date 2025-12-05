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
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, List

# Set EGL for headless rendering BEFORE importing mujoco
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
from jax import numpy as jp
import numpy as np
from ml_collections import config_dict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt

from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import mjcf
from orbax import checkpoint as ocp
from flax.training import orbax_utils
import mediapy as media

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
    
    NO force/velocity command in observation - robot must learn compliance from physical effects only.
    Velocity command slot filled with ZEROS, orientation uses actual desired orientation [0,0,1].
    
    Observation (matches joint training with estimator):
    - IMU data (angular velocity, gravity) - 6 dims
    - Velocity command (ZERO) - 3 dims  <-- No velocity info!
    - Desired orientation [0,0,1] - 3 dims  <-- Same as joint training
    - Motor angles - 12 dims  
    - Last action - 12 dims
    Total: 36 dims per frame × 20 frames = 720 dims
    
    Reward: Move at velocity proportional to applied force
    - desired_velocity = admittance_gain * force (computed internally for reward only)
    - Robot never sees any velocity command (always zero)
    - Robot must learn to "feel" the push from IMU/joint changes and respond
    """
    
    def __init__(
        self,
        admittance_gains: Tuple[float, float] = (0.1, 0.1),
        **kwargs
    ):
        super().__init__(**kwargs)
        self._admittance_gains = jp.array(admittance_gains)
        
        # Observation dim: 36 (matches original pupperv3-mjx for checkpoint compatibility)
        # IMU (6) + command_noise (3) + orientation_noise (3) + motor angles (12) + last action (12) = 36
        self.observation_dim = 36
        
        print(f"Pure RL Compliance Environment")
        print(f"  Admittance gains: x={admittance_gains[0]}, y={admittance_gains[1]} m/s per N")
        print(f"  Observation dim: {self.observation_dim} (command=ZERO, orientation=[0,0,1])")
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
        """Get observation with ZERO command but actual orientation (matches joint training).
        
        Observation (36 dims per frame, matches joint training with estimator):
        - IMU data (angular velocity, gravity) - 6 dims
        - Command slot (ZERO) - 3 dims  <-- No velocity command given
        - Desired orientation [0,0,1] - 3 dims  <-- Same as joint training
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
        ) = jax.random.split(state_info["rng"], 6)

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
        
        # ZERO for command slot - robot gets no velocity command
        command_zero = jp.zeros(3)
        
        # Use actual desired orientation (same as joint training with estimator)
        # This is typically [0, 0, 1] for "stay level"
        desired_orientation = state_info.get('desired_world_z_in_body_frame', jp.array([0.0, 0.0, 1.0]))

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

        # Construct observation with ZERO command but actual orientation (matches joint training)
        obs = jp.concatenate(
            [
                lagged_imu_data,  # 6 dims: angular velocity + gravity
                command_zero,  # 3 dims: ZERO (no velocity command)
                desired_orientation,  # 3 dims: desired orientation [0,0,1] = stay level
                pipeline_state.q[7:] - self._default_pose + motor_ang_noise,  # 12 dims: motor angles
                state_info["last_act"] + last_action_noise,  # 12 dims: last action
            ]
        )
        # Total: 6 + 3 + 3 + 12 + 12 = 36 dims per frame

        obs = jp.clip(obs, -100.0, 100.0)

        # Stack observations through time
        new_obs_history = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        return new_obs_history
    
    def render(
        self,
        trajectory,
        camera=None,
        force_vis_scale: float = 0.05,
    ):
        """Render trajectory - handles both State objects and pipeline_state objects.
        
        This override fixes compatibility with utils.visualize_policy which passes
        full State objects, while the base PupperV3Env.render expects pipeline_states.
        """
        if not trajectory:
            return []
        
        # Check if we got full State objects or pipeline_state objects
        if hasattr(trajectory[0], 'pipeline_state'):
            # Got full State objects - extract pipeline_states for base render
            pipeline_states = [s.pipeline_state for s in trajectory]
            return super().render(pipeline_states, camera=camera, force_vis_scale=force_vis_scale)
        else:
            # Got pipeline_state objects - pass directly to base render
            return super().render(trajectory, camera=camera, force_vis_scale=force_vis_scale)


# ============================================================================
# Configs
# ============================================================================

def get_reward_config():
    config = config_dict.ConfigDict()
    config.rewards = config_dict.ConfigDict()
    config.rewards.scales = config_dict.ConfigDict()
    
    # High weight on velocity tracking - this is the main objective!
    config.rewards.scales.tracking_lin_vel = 5
    config.rewards.scales.tracking_ang_vel = 0.0  # No yaw tracking
    
    # Standard locomotion rewards
    config.rewards.scales.tracking_orientation = 1.0
    config.rewards.scales.lin_vel_z = -2.0
    config.rewards.scales.ang_vel_xy = -0.05
    config.rewards.scales.orientation = -5.0
    config.rewards.scales.torques = -0.01
    config.rewards.scales.joint_acceleration = -1e-3
    config.rewards.scales.mechanical_work = 0.0
    config.rewards.scales.action_rate = -0.1
    config.rewards.scales.feet_air_time = 0.05
    config.rewards.scales.stand_still = -0.5
    config.rewards.scales.stand_still_joint_velocity = -0.1
    config.rewards.scales.abduction_angle = -0.1
    config.rewards.scales.zero_force_motion = -5.0  # Heavy penalty when moving with no applied force
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
    config.ppo.learning_rate = 3.0e-5  # Match joint training
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
    parser.add_argument("--learning-rate", type=float, default=3.0e-5,
                        help="Learning rate for PPO (default matches joint training)")
    
    # Compliance
    parser.add_argument("--admittance-gains", type=str, default="0.2,0.2",
                        help="Admittance gains (x,y) in m/s per N")
    
    # Force parameters
    parser.add_argument("--force-magnitude-min", type=float, default=2.0,
                        help="Minimum force magnitude (N)")
    parser.add_argument("--force-magnitude-max", type=float, default=6.0,
                        help="Maximum force magnitude (N)")
    parser.add_argument("--force-probability", type=float, default=0.8,
                        help="Probability of force being active")
    parser.add_argument("--force-duration-min", type=int, default=40,
                        help="Minimum force duration in environment steps")
    parser.add_argument("--force-duration-max", type=int, default=120,
                        help="Maximum force duration in environment steps")
    
    # Checkpoint
    parser.add_argument("--restore-checkpoint", type=str, default=None,
                        help="Path to checkpoint to restore from")
    
    # Logging
    parser.add_argument("--wandb-key", type=str, default=None, help="WandB API key")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")
    parser.add_argument("--wandb-project", type=str, default="pupperv3-pure-rl-compliance",
                        help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity (team/user)")
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
    print(f"Force duration: {args.force_duration_min} - {args.force_duration_max} steps")
    print(f"Num timesteps: {args.num_timesteps:,}")
    print(f"Num envs: {args.num_envs}")
    print(f"Output: {output_folder}")
    print("=" * 70)
    
    # Load configs first (needed for wandb config)
    reward_config = get_reward_config()
    sim_config = get_simulation_config(args.model_path)
    training_config = get_training_config(args)
    
    # Override learning rate if provided
    training_config.ppo.learning_rate = args.learning_rate
    
    # Build comprehensive wandb config
    full_config = {
        # Compliance-specific
        "compliance": {
            "admittance_gains": admittance_gains,
            "force_magnitude_range": [args.force_magnitude_min, args.force_magnitude_max],
            "force_probability": args.force_probability,
            "force_duration_range": [args.force_duration_min, args.force_duration_max],
        },
        # PPO params
        "ppo": training_config.ppo.to_dict(),
        # Reward scales
        "reward_scales": reward_config.rewards.scales.to_dict(),
        "tracking_sigma": reward_config.rewards.tracking_sigma,
        # Policy architecture
        "policy": {
            "hidden_layer_sizes": training_config.policy.hidden_layer_sizes,
            "activation": training_config.policy.activation,
            "action_scale": 0.3,
            "observation_history": 20,
        },
        # Domain randomization
        "domain_randomization": {
            "friction_range": [0.6, 1.4],
            "body_mass_scale_range": [0.8, 1.2],
            "kp_multiplier_range": [0.85, 1.15],
            "kd_multiplier_range": [0.85, 1.15],
        },
        # Training
        "seed": args.seed,
        "restore_checkpoint": args.restore_checkpoint,
        # Simulation
        "simulation": {
            "model_path": args.model_path,
            "physics_dt": sim_config.physics_dt,
            "environment_dt": training_config.environment_dt,
        },
    }
    
    # Initialize WandB with comprehensive config
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        try:
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                config=full_config,
                save_code=True,
                settings=wandb.Settings(
                    _service_wait=90,
                    init_timeout=90,
                )
            )
            print(f"WandB initialized: {wandb.run.name}")
            # Update output folder to include wandb run name
            output_folder = Path(f"output_{wandb.run.name}").resolve()
            output_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"WandB init failed: {e}")
            use_wandb = False
    else:
        print("Training without WandB logging")
    
    # Create environment factory
    def create_env(**kwargs):
        return PupperV3EnvPureCompliance(
            path=args.model_path,
            reward_config=reward_config,
            action_scale=0.3,
            observation_history=20,  # 20 frames × 36 dims = 720 dim observation
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
            force_duration_range=jp.array([args.force_duration_min, args.force_duration_max]),
            admittance_gains=admittance_gains,
            **kwargs
        )
    
    env = create_env()
    eval_env = create_env()
    
    print(f"\nEnvironment created:")
    print(f"  Observation size: {env.observation_size} (36 dims × 20 frames = 720)")
    print(f"  Action size: {env.action_size}")
    print(f"  Command slot: ZERO (no velocity command)")
    print(f"  Orientation slot: [0,0,1] (stay level, same as joint training)")
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
    
    # =========================================================================
    # Progress tracking (like Nathan's notebook)
    # =========================================================================
    x_data: List[int] = []
    y_data: List[float] = []
    ydataerr: List[float] = []
    times: List[datetime] = [datetime.now()]
    
    def progress_fn(num_steps: int, metrics: dict):
        """
        Comprehensive progress logging with plots and wandb integration.
        Logs ALL metrics from Brax PPO, not just reward.
        """
        times.append(datetime.now())
        
        if 'eval/episode_reward' in metrics:
            reward = metrics['eval/episode_reward']
            reward_std = metrics.get('eval/episode_reward_std', 0)
            
            x_data.append(num_steps)
            y_data.append(reward)
            ydataerr.append(reward_std)
            
            # Calculate training speed
            if len(times) >= 2:
                elapsed = (times[-1] - times[-2]).total_seconds()
                if elapsed > 0 and len(x_data) >= 2:
                    steps_since_last = x_data[-1] - x_data[-2] if len(x_data) >= 2 else num_steps
                    steps_per_sec = steps_since_last / elapsed
                else:
                    steps_per_sec = 0
            else:
                steps_per_sec = 0
            
            # Print progress
            print(f"  Step {num_steps:,} | Reward: {reward:.2f} ± {reward_std:.2f} | Speed: {steps_per_sec:,.0f} steps/s")
            
            # Save progress plot
            try:
                plt.figure(figsize=(10, 6))
                plt.xlim([0, args.num_timesteps * 1.25])
                plt.ylim([0, 50])  # Adjust based on expected reward range
                plt.xlabel("# environment steps")
                plt.ylabel("reward per episode")
                plt.title(f"Pure RL Compliance Training - Latest: {reward:.2f}")
                plt.errorbar(x_data, y_data, yerr=ydataerr, capsize=3, marker='o', markersize=4)
                plt.grid(True, alpha=0.3)
                plot_path = output_folder / "training_progress.png"
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  Warning: Could not save progress plot: {e}")
        
        # Log ALL metrics to wandb (like Nathan's notebook)
        if use_wandb:
            try:
                # Log the full metrics dict (includes all PPO internals)
                wandb.log(metrics, step=num_steps)
                
                # Also log computed values
                if 'eval/episode_reward' in metrics:
                    wandb.log({
                        "training/steps_per_second": steps_per_sec,
                        "training/total_time_minutes": (times[-1] - times[0]).total_seconds() / 60,
                    }, step=num_steps)
            except Exception as e:
                print(f"  Warning: wandb logging failed: {e}")
    
    # =========================================================================
    # Track params and save checkpoints
    # =========================================================================
    latest_full_params = [None]
    
    def policy_params_fn(current_step: int, make_policy, params):
        """
        Callback for saving checkpoints and rendering videos.
        Logs checkpoints and videos to wandb.
        """
        latest_full_params[0] = params
        
        # Save checkpoint
        checkpoint_step_path = output_folder / str(current_step)
        try:
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(params)
            orbax_checkpointer.save(
                str(checkpoint_step_path.resolve()),
                params,
                force=True,
                save_args=save_args
            )
            print(f"  Saved checkpoint: {checkpoint_step_path}")
            
            # Log checkpoint to wandb
            if use_wandb:
                try:
                    wandb.log_model(
                        path=str(checkpoint_step_path),
                        name=f"checkpoint_{wandb.run.name}_{current_step}"
                    )
                except Exception as e:
                    print(f"  Warning: wandb checkpoint logging failed: {e}")
        except Exception as e:
            print(f"  Warning: checkpoint saving failed: {e}")
        
        # Render video using the proven utils.visualize_policy function
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
                output_folder=str(output_folder),
                vx=0.5,  # Test forward/back
                vy=0.3,  # Test sideways
                wz=1.0,  # Test rotation
            )
            
            # Note: utils.visualize_policy handles wandb logging internally
                    
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
    
    # =========================================================================
    # Train with timing!
    # =========================================================================
    print("\nStarting training...")
    print("  (First step includes JIT compilation time)")
    
    training_start_time = datetime.now()
    
    make_inference_fn, params, final_metrics = train_fn(
        environment=env,
        progress_fn=progress_fn,
        eval_env=eval_env,
        policy_params_fn=policy_params_fn,
        **checkpoint_kwargs
    )
    
    training_end_time = datetime.now()
    
    # Calculate timing metrics
    total_training_time = (training_end_time - training_start_time).total_seconds()
    time_to_jit = (times[1] - times[0]).total_seconds() if len(times) > 1 else 0
    time_to_train = (times[-1] - times[1]).total_seconds() if len(times) > 1 else total_training_time
    
    print(f"\n  Time to JIT: {time_to_jit:.1f}s")
    print(f"  Time to train: {time_to_train:.1f}s ({time_to_train/60:.1f} min)")
    print(f"  Total time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
    
    if args.num_timesteps > 0 and time_to_train > 0:
        avg_steps_per_sec = args.num_timesteps / time_to_train
        print(f"  Average speed: {avg_steps_per_sec:,.0f} steps/s")
    
    # Log timing to wandb
    if use_wandb:
        try:
            wandb.run.summary["time_to_jit"] = time_to_jit
            wandb.run.summary["time_to_train"] = time_to_train
            wandb.run.summary["total_training_time"] = total_training_time
            if len(y_data) > 0:
                wandb.run.summary["final_reward"] = y_data[-1]
                wandb.run.summary["best_reward"] = max(y_data)
        except Exception as e:
            print(f"  Warning: wandb summary logging failed: {e}")
    
    # =========================================================================
    # Save final checkpoint
    # =========================================================================
    final_checkpoint_path = (output_folder / "final_checkpoint").resolve()
    if latest_full_params[0] is not None:
        try:
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(latest_full_params[0])
            orbax_checkpointer.save(
                str(final_checkpoint_path),
                latest_full_params[0],
                force=True,
                save_args=save_args
            )
            print(f"\nSaved final checkpoint to: {final_checkpoint_path}")
            
            # Log final checkpoint to wandb
            if use_wandb:
                try:
                    wandb.log_model(
                        path=str(final_checkpoint_path),
                        name=f"final_checkpoint_{wandb.run.name}"
                    )
                except:
                    pass
        except Exception as e:
            print(f"  Warning: Final checkpoint saving failed: {e}")
    
    # Save final progress plot
    try:
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 1, 1)
        plt.fill_between(x_data, 
                         [y - e for y, e in zip(y_data, ydataerr)],
                         [y + e for y, e in zip(y_data, ydataerr)],
                         alpha=0.3, color='blue')
        plt.plot(x_data, y_data, 'b-', linewidth=2, marker='o', markersize=5)
        plt.xlabel("Environment Steps", fontsize=12)
        plt.ylabel("Episode Reward", fontsize=12)
        plt.title(f"Pure RL Compliance Training - Final: {y_data[-1]:.2f}" if y_data else "Training Progress", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        final_plot_path = output_folder / "final_training_progress.png"
        plt.savefig(final_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved final progress plot to: {final_plot_path}")
        
        # Log final plot to wandb
        if use_wandb:
            try:
                wandb.log({"training/final_progress_plot": wandb.Image(str(final_plot_path))})
            except:
                pass
    except Exception as e:
        print(f"  Warning: Could not save final plot: {e}")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    if y_data:
        print(f"  Final reward: {y_data[-1]:.2f}")
        print(f"  Best reward: {max(y_data):.2f}")
    print(f"  Output folder: {output_folder}")
    print("=" * 70)
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

