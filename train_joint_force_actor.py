#!/usr/bin/env python3
"""
Joint training script for Force Estimator + Actor Policy.

This script:
1. Loads a pretrained force estimator from JSON
2. Loads a pretrained actor checkpoint (Brax PPO)
3. Creates PupperV3EnvWithEstimator that integrates force estimator + admittance
4. Fine-tunes the actor to track velocity commands from admittance

The actor learns to be robust to force estimator noise by training with
velocity commands derived from (potentially noisy) force estimates.

Usage:
    python train_joint_force_actor.py \
        --force-estimator-path force_estimator_training/force_estimator.json \
        --actor-checkpoint-path output_morning-jazz-49/501350400 \
        --admittance-gains 0.5,0.5 \
        --num-timesteps 100000000
"""

import argparse
import functools
import os
from datetime import datetime
from pathlib import Path

import jax
from jax import numpy as jp
import numpy as np
from ml_collections import config_dict

from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model

from pupperv3_mjx import domain_randomization, utils
from pupperv3_mjx.environment_with_estimator import PupperV3EnvWithEstimator

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed, training without logging")


def get_reward_config():
    """Get reward configuration for compliance training.
    
    Key rewards:
    - tracking_lin_vel: Main reward for tracking velocity command from admittance
    - Other regularization rewards for smooth, stable behavior
    """
    config = config_dict.ConfigDict()
    config.rewards = config_dict.ConfigDict()
    config.rewards.scales = config_dict.ConfigDict()
    
    # Primary tracking reward - this is what the actor optimizes for
    config.rewards.scales.tracking_lin_vel = 1.5
    config.rewards.scales.tracking_ang_vel = 0.0  # Not used with force-based commands
    config.rewards.scales.tracking_orientation = 1.0
    
    # Regularization rewards
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
    config.rewards.scales.force_following = 0.0  # Not needed - we use velocity tracking
    
    config.rewards.tracking_sigma = 0.25
    
    return config


def get_simulation_config(model_path: str):
    """Get simulation configuration."""
    from etils import epath
    from brax.io import mjcf
    
    config = config_dict.ConfigDict()
    config.model_path = model_path
    
    # Load model to get joint limits
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
    """Get training configuration."""
    config = config_dict.ConfigDict()
    
    # Environment timestep
    config.environment_dt = 0.02
    
    # PPO params
    config.ppo = config_dict.ConfigDict()
    config.ppo.num_timesteps = args.num_timesteps
    config.ppo.episode_length = 1000
    config.ppo.num_evals = 10
    config.ppo.reward_scaling = 1
    config.ppo.normalize_observations = True
    config.ppo.action_repeat = 1
    config.ppo.unroll_length = 20
    config.ppo.num_minibatches = 32
    config.ppo.num_updates_per_batch = 4
    config.ppo.discounting = 0.97
    config.ppo.learning_rate = 3.0e-5  # Lower LR for fine-tuning
    config.ppo.entropy_cost = 1e-2
    config.ppo.num_envs = args.num_envs
    config.ppo.batch_size = 256
    
    # Command sampling (will be overridden by admittance)
    config.resample_velocity_step = config.ppo.episode_length // 2
    config.lin_vel_x_range = [-0.75, 0.75]
    config.lin_vel_y_range = [-0.5, 0.5]
    config.ang_vel_yaw_range = [-2.0, 2.0]
    config.zero_command_probability = 0.0  # Admittance controls command
    config.stand_still_command_threshold = 0.05
    
    # Orientation
    config.maximum_pitch_command = 0.0
    config.maximum_roll_command = 0.0
    config.desired_world_z_in_body_frame = (0.0, 0.0, 1.0)
    
    # Termination
    config.terminal_body_z = 0.1
    config.terminal_body_angle = 0.52
    config.early_termination_step_threshold = config.ppo.episode_length // 2
    
    # Joint PD
    config.dof_damping = 0.25
    config.position_control_kp = 5.5
    
    # Default pose
    config.default_pose = jp.array(
        [0.26, 0.0, -0.52, -0.26, 0.0, 0.52, 0.26, 0.0, -0.52, -0.26, 0.0, 0.52]
    )
    config.desired_abduction_angles = jp.array([0.0, 0.0, 0.0, 0.0])
    
    # Domain randomization
    config.kick_probability = 0.1
    config.kick_vel = 0.2
    config.angular_velocity_noise = 0.1
    config.gravity_noise = 0.05
    config.motor_angle_noise = 0.05
    config.last_action_noise = 0.01
    
    # Force perturbation (still active for robustness)
    config.force_probability = 0.8
    config.force_magnitude_range = jp.array([3, 6])
    config.force_duration_range = jp.array([40, 120])
    config.force_point_noise_sd = 0.05
    
    # Starting position
    config.start_position_config = domain_randomization.StartPositionRandomization(
        x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, z_min=0.15, z_max=0.20
    )
    
    # Latency
    config.latency_distribution = jp.array([0.2, 0.8])
    config.imu_latency_distribution = jp.array([0.5, 0.5])
    
    # Domain randomization ranges
    config.friction_range = (0.6, 1.4)
    config.position_control_kp_multiplier_range = (0.6, 1.1)
    config.position_control_kd_multiplier_range = (0.8, 1.5)
    config.body_com_x_shift_range = (-0.02, 0.02)
    config.body_com_y_shift_range = (-0.005, 0.005)
    config.body_com_z_shift_range = (-0.005, 0.005)
    config.body_mass_scale_range = (0.9, 1.3)
    config.body_inertia_scale_range = (0.9, 1.3)
    
    return config


def get_policy_config():
    """Get policy network configuration."""
    config = config_dict.ConfigDict()
    config.use_imu = True
    config.observation_history = 20
    config.action_scale = 0.75
    config.hidden_layer_sizes = (256, 128, 128, 128)
    config.activation = "elu"
    return config


def create_env(sim_config, train_config, policy_config, reward_config, 
               force_estimator_path, admittance_gains, use_ground_truth_force=False):
    """Create PupperV3EnvWithEstimator."""
    
    env_kwargs = dict(
        path=sim_config.model_path,
        force_estimator_path=force_estimator_path,
        admittance_gains=admittance_gains,
        use_ground_truth_force=use_ground_truth_force,
        action_scale=policy_config.action_scale,
        observation_history=policy_config.observation_history,
        joint_lower_limits=sim_config.joint_lower_limits,
        joint_upper_limits=sim_config.joint_upper_limits,
        dof_damping=train_config.dof_damping,
        position_control_kp=train_config.position_control_kp,
        foot_site_names=sim_config.foot_site_names,
        torso_name=sim_config.torso_name,
        upper_leg_body_names=sim_config.upper_leg_body_names,
        lower_leg_body_names=sim_config.lower_leg_body_names,
        resample_velocity_step=train_config.resample_velocity_step,
        linear_velocity_x_range=train_config.lin_vel_x_range,
        linear_velocity_y_range=train_config.lin_vel_y_range,
        angular_velocity_range=train_config.ang_vel_yaw_range,
        zero_command_probability=train_config.zero_command_probability,
        stand_still_command_threshold=train_config.stand_still_command_threshold,
        maximum_pitch_command=train_config.maximum_pitch_command,
        maximum_roll_command=train_config.maximum_roll_command,
        start_position_config=train_config.start_position_config,
        default_pose=train_config.default_pose,
        desired_abduction_angles=train_config.desired_abduction_angles,
        reward_config=reward_config,
        angular_velocity_noise=train_config.angular_velocity_noise,
        gravity_noise=train_config.gravity_noise,
        motor_angle_noise=train_config.motor_angle_noise,
        last_action_noise=train_config.last_action_noise,
        kick_vel=train_config.kick_vel,
        kick_probability=train_config.kick_probability,
        force_probability=train_config.force_probability,
        force_duration_range=train_config.force_duration_range,
        force_magnitude_range=train_config.force_magnitude_range,
        force_point_noise_sd=train_config.force_point_noise_sd,
        terminal_body_z=train_config.terminal_body_z,
        early_termination_step_threshold=train_config.early_termination_step_threshold,
        terminal_body_angle=train_config.terminal_body_angle,
        foot_radius=sim_config.foot_radius,
        environment_timestep=train_config.environment_dt,
        physics_timestep=sim_config.physics_dt,
        latency_distribution=train_config.latency_distribution,
        imu_latency_distribution=train_config.imu_latency_distribution,
        desired_world_z_in_body_frame=jp.array(train_config.desired_world_z_in_body_frame),
        use_imu=policy_config.use_imu,
    )
    
    return PupperV3EnvWithEstimator(**env_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Joint Force Estimator + Actor Training")
    parser.add_argument(
        "--force-estimator-path",
        type=str,
        required=True,
        help="Path to force_estimator.json"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="../pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.no_body.self_collision.two_iterations.xml",
        help="Path to MuJoCo XML model file (relative to pupperv3-compliance folder)"
    )
    parser.add_argument(
        "--actor-checkpoint-path",
        type=str,
        default=None,
        help="Path to pretrained actor checkpoint (Brax format). If not provided, trains from scratch."
    )
    parser.add_argument(
        "--admittance-gains",
        type=str,
        default="0.5,0.5",
        help="Comma-separated admittance gains (x,y) in m/s per N"
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=100_000_000,
        help="Number of training timesteps"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4096,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_compliance",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--use-ground-truth-force",
        action="store_true",
        help="Use ground truth force instead of estimated (for debugging)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--wandb-key",
        type=str,
        default=None,
        help="Weights & Biases API key (optional)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    
    args = parser.parse_args()
    
    # Parse admittance gains
    admittance_gains = tuple(float(x) for x in args.admittance_gains.split(","))
    assert len(admittance_gains) == 2, "Admittance gains must be 2 values (x,y)"
    
    print("=" * 60)
    print("Joint Force Estimator + Actor Training")
    print("=" * 60)
    print(f"Force estimator: {args.force_estimator_path}")
    print(f"Actor checkpoint: {args.actor_checkpoint_path or 'None (training from scratch)'}")
    print(f"Admittance gains: x={admittance_gains[0]}, y={admittance_gains[1]} m/s per N")
    print(f"Num timesteps: {args.num_timesteps:,}")
    print(f"Num envs: {args.num_envs}")
    print(f"Use ground truth force: {args.use_ground_truth_force}")
    print("=" * 60)
    
    # Initialize wandb if available and not disabled
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        try:
            wandb.init(
                project="pupperv3-compliance",
                config={
                    "force_estimator_path": args.force_estimator_path,
                    "actor_checkpoint_path": args.actor_checkpoint_path,
                    "admittance_gains": admittance_gains,
                    "num_timesteps": args.num_timesteps,
                    "num_envs": args.num_envs,
                },
                settings={"_service_wait": 90, "init_timeout": 90}
            )
            print("Wandb initialized successfully")
        except Exception as e:
            print(f"Wandb init failed: {e}")
            use_wandb = False
    else:
        print("Training without wandb logging")
    
    # Get configs
    sim_config = get_simulation_config(args.model_path)
    train_config = get_training_config(args)
    policy_config = get_policy_config()
    reward_config = get_reward_config()
    
    # Create output directory
    train_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = f"{args.output_dir}_{train_datetime}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output directory: {output_folder}")
    
    # Create environments
    print("\nCreating environments...")
    env = create_env(
        sim_config, train_config, policy_config, reward_config,
        args.force_estimator_path, admittance_gains, args.use_ground_truth_force
    )
    eval_env = create_env(
        sim_config, train_config, policy_config, reward_config,
        args.force_estimator_path, admittance_gains, args.use_ground_truth_force
    )
    
    # Setup network factory
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_config.hidden_layer_sizes,
        activation=utils.activation_fn_map(policy_config.activation)
    )
    
    # Setup training function
    train_fn = functools.partial(
        ppo.train,
        **(train_config.ppo.to_dict()),
        network_factory=make_networks_factory,
        randomization_fn=functools.partial(
            domain_randomization.domain_randomize,
            friction_range=train_config.friction_range,
            kp_multiplier_range=train_config.position_control_kp_multiplier_range,
            kd_multiplier_range=train_config.position_control_kd_multiplier_range,
            body_com_x_shift_range=train_config.body_com_x_shift_range,
            body_com_y_shift_range=train_config.body_com_y_shift_range,
            body_com_z_shift_range=train_config.body_com_z_shift_range,
            body_mass_scale_range=train_config.body_mass_scale_range,
            body_inertia_scale_range=train_config.body_inertia_scale_range,
        ),
        seed=args.seed,
    )
    
    # Progress tracking
    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]
    
    # Custom progress function that doesn't require wandb
    def progress_fn(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics.get('eval/episode_reward', 0))
        ydataerr.append(metrics.get('eval/episode_reward_std', 0))
        
        reward = metrics.get('eval/episode_reward', 0)
        reward_std = metrics.get('eval/episode_reward_std', 0)
        
        elapsed = times[-1] - times[0]
        steps_per_sec = num_steps / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        
        print(f"Step {num_steps:,} | Reward: {reward:.2f} ± {reward_std:.2f} | Steps/sec: {steps_per_sec:,.0f}")
        
        if use_wandb:
            try:
                wandb.log(metrics, step=num_steps)
            except Exception as e:
                pass  # Silently ignore wandb errors
    
    # JIT step and reset for visualization
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    def policy_params_fn(current_step, make_policy, params):
        """Save checkpoints and visualize policy."""
        try:
            utils.visualize_policy(
                current_step=current_step,
                make_policy=make_policy,
                params=params,
                eval_env=eval_env,
                jit_step=jit_step,
                jit_reset=jit_reset,
                output_folder=output_folder
            )
        except Exception as e:
            print(f"  Warning: Video rendering failed: {e}")
        
        utils.save_checkpoint(
            current_step=current_step,
            make_policy=make_policy,
            params=params,
            checkpoint_path=output_folder
        )
        print(f"  Checkpoint saved at step {current_step}")
    
    # Setup checkpoint restoration
    checkpoint_kwargs = {}
    if args.actor_checkpoint_path:
        # Ensure absolute path
        checkpoint_path = Path(args.actor_checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        
        if checkpoint_path.exists():
            print(f"\nRestoring actor from checkpoint: {checkpoint_path}")
            checkpoint_kwargs["restore_checkpoint_path"] = checkpoint_path
        else:
            print(f"\nWARNING: Checkpoint path does not exist: {checkpoint_path}")
            print("Training from scratch...")
    
    # Train!
    print("\nStarting training...")
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress_fn,
        eval_env=eval_env,
        policy_params_fn=policy_params_fn,
        **checkpoint_kwargs
    )
    
    # Save final model
    print("\nSaving final model...")
    model_path = os.path.join(output_folder, f'mjx_params_{train_datetime}')
    model.save_params(model_path, params)
    
    # Export to JSON for deployment
    from pupperv3_mjx import export
    params_rtneural = export.convert_params(
        jax.block_until_ready(params),
        activation=policy_config.activation,
        action_scale=policy_config.action_scale,
        kp=train_config.position_control_kp,
        kd=train_config.dof_damping,
        default_pose=train_config.default_pose,
        joint_upper_limits=sim_config.joint_upper_limits,
        joint_lower_limits=sim_config.joint_lower_limits,
        use_imu=policy_config.use_imu,
        observation_history=policy_config.observation_history,
        final_activation="tanh",
    )
    
    policy_json_path = os.path.join(output_folder, "policy.json")
    import json
    with open(policy_json_path, "w") as f:
        json.dump(params_rtneural, f)
    
    print(f"\nTraining complete!")
    print(f"Time to JIT: {times[1] - times[0]}")
    print(f"Total training time: {times[-1] - times[1]}")
    print(f"Saved model to: {model_path}")
    print(f"Saved policy JSON to: {policy_json_path}")


if __name__ == "__main__":
    main()

