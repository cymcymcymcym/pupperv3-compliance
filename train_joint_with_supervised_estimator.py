#!/usr/bin/env python3
"""
Joint training: Force Estimator (supervised) + Actor (PPO).

This script implements FULLY AUTOMATED alternating training:
1. Train actor with PPO for N steps (force estimator frozen)
2. Collect (obs, ground_truth_force) pairs during actor rollouts
3. Update force estimator with supervised learning on collected data
4. Repeat for multiple rounds

Features:
- ✅ Load pretrained actor checkpoint
- ✅ Load pretrained force estimator
- ✅ WandB logging support
- ✅ Video rendering during training
- ✅ Checkpoint saving

Usage:
    python train_joint_with_supervised_estimator.py \
        --force-estimator-path force_estimator_training/force_estimator.json \
        --actor-checkpoint-path output_morning-jazz-49/501350400 \
        --admittance-gains 0.25,0.25 \
        --num-rounds 5 \
        --actor-steps-per-round 50000000
"""

import argparse
import functools
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, List

# Set EGL for headless rendering BEFORE importing mujoco
os.environ.setdefault("MUJOCO_GL", "egl")

import flax.linen as nn
import jax
from jax import numpy as jp
import numpy as np
import optax
from ml_collections import config_dict

from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model, mjcf
from orbax import checkpoint as ocp

from pupperv3_mjx import domain_randomization, utils
from pupperv3_mjx.environment_with_estimator import PupperV3EnvWithEstimator

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed, training without logging")


# ============================================================================
# Force Estimator Network (matches the pretrained architecture)
# ============================================================================

class ForceEstimatorNetwork(nn.Module):
    """Flax MLP for force estimation - matches ForceEstimatorLarge."""
    hidden_sizes: Tuple[int, ...] = (512, 512, 256, 128)
    
    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = False) -> jp.ndarray:
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)
        x = nn.Dense(3)(x)
        return x


# ============================================================================
# Load Force Estimator from JSON
# ============================================================================

def load_estimator_from_json(json_path: str) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Load force estimator params from JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    
    input_mean = np.array(data["input_mean"])
    input_std = np.array(data["input_std"])
    
    # Reconstruct params dict
    params = {}
    layers = data["layers"]
    
    dense_idx = 0
    ln_idx = 0
    for layer in layers:
        layer_type = layer.get("type", "dense")
        weights = layer["weights"]
        
        if layer_type == "dense":
            params[f"Dense_{dense_idx}"] = {
                "kernel": np.array(weights[0]),
                "bias": np.array(weights[1]),
            }
            dense_idx += 1
        elif layer_type == "layer_norm":
            params[f"LayerNorm_{ln_idx}"] = {
                "scale": np.array(weights[0]),
                "bias": np.array(weights[1]),
            }
            ln_idx += 1
    
    return params, input_mean, input_std


def export_force_estimator(params: Dict, input_mean: np.ndarray, input_std: np.ndarray, export_path: str):
    """Export force estimator to JSON."""
    layers = []
    
    # Count layers
    num_dense = len([k for k in params.keys() if k.startswith("Dense_")])
    num_ln = len([k for k in params.keys() if k.startswith("LayerNorm_")])
    num_hidden = num_ln  # LayerNorm count = hidden layer count
    
    for i in range(num_hidden):
        # Dense
        kernel = np.array(params[f"Dense_{i}"]["kernel"])
        bias = np.array(params[f"Dense_{i}"]["bias"])
        layers.append({
            "type": "dense",
            "activation": "identity",
            "shape": [None, int(bias.shape[0])],
            "weights": [kernel.tolist(), bias.tolist()],
        })
        
        # LayerNorm
        scale = np.array(params[f"LayerNorm_{i}"]["scale"])
        ln_bias = np.array(params[f"LayerNorm_{i}"]["bias"])
        layers.append({
            "type": "layer_norm",
            "activation": "elu",
            "shape": [None, int(scale.shape[0])],
            "weights": [scale.tolist(), ln_bias.tolist()],
        })
    
    # Output layer
    kernel = np.array(params[f"Dense_{num_hidden}"]["kernel"])
    bias = np.array(params[f"Dense_{num_hidden}"]["bias"])
    layers.append({
        "type": "dense",
        "activation": "identity",
        "shape": [None, 3],
        "weights": [kernel.tolist(), bias.tolist()],
    })
    
    # Handle zero std
    input_std_safe = np.where(np.array(input_std) < 1e-6, 1.0, np.array(input_std))
    
    export_dict = {
        "input_mean": np.array(input_mean).tolist(),
        "input_std": input_std_safe.tolist(),
        "layers": layers,
    }
    
    with open(export_path, "w") as f:
        json.dump(export_dict, f)
    
    print(f"  Exported force estimator to {export_path}")


# ============================================================================
# Supervised Training for Force Estimator
# ============================================================================

def train_force_estimator_supervised(
    params: Dict,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    obs_data: np.ndarray,
    force_data: np.ndarray,
    num_epochs: int = 50,
    batch_size: int = 2048,
    learning_rate: float = 1e-4,
    direction_weight: float = 0.8,
    magnitude_weight: float = 0.2,
) -> Dict:
    """Train force estimator with supervised learning."""
    print(f"  Training force estimator on {len(obs_data)} samples...")
    
    # Handle zero std
    input_std_safe = np.where(input_std < 1e-6, 1.0, input_std)
    
    # Normalize observations
    obs_norm = (obs_data - input_mean) / input_std_safe
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Infer hidden sizes from params
    hidden_sizes = []
    num_ln = len([k for k in params.keys() if k.startswith("LayerNorm_")])
    for i in range(num_ln):
        hidden_sizes.append(params[f"Dense_{i}"]["bias"].shape[0])
    
    model = ForceEstimatorNetwork(hidden_sizes=tuple(hidden_sizes))
    
    @jax.jit
    def loss_fn(params, obs_batch, force_batch):
        pred = model.apply({'params': params}, obs_batch, train=True)
        
        # Direction loss (cosine similarity)
        pred_norm = jp.linalg.norm(pred, axis=-1, keepdims=True) + 1e-6
        target_norm = jp.linalg.norm(force_batch, axis=-1, keepdims=True) + 1e-6
        pred_unit = pred / pred_norm
        target_unit = force_batch / target_norm
        cosine_sim = jp.sum(pred_unit * target_unit, axis=-1)
        target_mag = target_norm.squeeze()
        has_force = target_mag > 0.1
        direction_loss = jp.mean(jp.where(has_force, 1.0 - cosine_sim, 0.0))
        
        # Magnitude loss
        pred_mag = pred_norm.squeeze()
        magnitude_loss = jp.mean((pred_mag - target_mag) ** 2)
        
        return direction_weight * direction_loss + magnitude_weight * magnitude_loss
    
    @jax.jit
    def train_step(params, opt_state, obs_batch, force_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, obs_batch, force_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Training loop - SIMPLE AND FAST (matching train_force_estimator.py approach)
    num_samples = len(obs_norm)
    num_batches = num_samples // batch_size
    
    # Pre-transfer ALL data to GPU once (avoid per-batch transfers)
    obs_gpu = jp.array(obs_norm)
    force_gpu = jp.array(force_data)
    
    best_loss = float('inf')
    best_params = params
    
    # Use JAX random key for GPU-native permutations
    rng = jax.random.PRNGKey(42)
    
    for epoch in range(num_epochs):
        # Generate permutation ON GPU (like original train_force_estimator.py)
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, num_samples)
        
        total_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_idx = perm[start:end]  # GPU indexing
            
            # GPU→GPU indexing (fast, no transfer)
            obs_batch = obs_gpu[batch_idx]
            force_batch = force_gpu[batch_idx]
            
            params, opt_state, loss = train_step(params, opt_state, obs_batch, force_batch)
            total_loss += float(loss)  # Small sync per batch, but simple
        
        avg_loss = total_loss / max(1, num_batches)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Only copy params when we have a new best (not every epoch)
            best_params = jax.tree.map(lambda x: np.array(x), params)
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"    Epoch {epoch+1}/{num_epochs}: loss = {avg_loss:.6f}")
    
    print(f"  Best loss: {best_loss:.6f}")
    return best_params


# ============================================================================
# Data Collection
# ============================================================================

class DataCollector:
    """Collects (obs, force) pairs."""
    
    def __init__(self, max_size: int = 500000):
        self.max_size = max_size
        self.obs_list: List[np.ndarray] = []
        self.force_list: List[np.ndarray] = []
    
    def add(self, obs: np.ndarray, force: np.ndarray):
        """Add samples (including zero-force samples)."""
        self.obs_list.append(obs)
        self.force_list.append(force)
        
        # Trim if too large
        total = sum(len(o) for o in self.obs_list)
        while total > self.max_size and len(self.obs_list) > 1:
            self.obs_list.pop(0)
            self.force_list.pop(0)
            total = sum(len(o) for o in self.obs_list)
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.obs_list:
            return np.array([]), np.array([])
        return np.concatenate(self.obs_list), np.concatenate(self.force_list)
    
    def clear(self):
        self.obs_list = []
        self.force_list = []
    
    def __len__(self):
        return sum(len(o) for o in self.obs_list)


# ============================================================================
# Configs
# ============================================================================

def get_reward_config():
    config = config_dict.ConfigDict()
    config.rewards = config_dict.ConfigDict()
    config.rewards.scales = config_dict.ConfigDict()
    
    config.rewards.scales.tracking_lin_vel = 1.5
    config.rewards.scales.tracking_ang_vel = 0.0
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
    config.rewards.scales.force_following = 0.0
    config.rewards.tracking_sigma = 0.25
    
    return config


def get_simulation_config(model_path: str):
    from etils import epath
    
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


def get_training_config(args, actor_steps: int):
    config = config_dict.ConfigDict()
    config.environment_dt = 0.02
    
    config.ppo = config_dict.ConfigDict()
    config.ppo.num_timesteps = actor_steps
    config.ppo.episode_length = 1000
    config.ppo.num_evals = 2
    config.ppo.reward_scaling = 1
    config.ppo.normalize_observations = True
    config.ppo.action_repeat = 1
    config.ppo.unroll_length = 20
    config.ppo.num_minibatches = 32
    config.ppo.num_updates_per_batch = 4
    config.ppo.discounting = 0.97
    config.ppo.learning_rate = 3.0e-5
    config.ppo.entropy_cost = 1e-2
    config.ppo.num_envs = args.num_envs
    config.ppo.batch_size = 256
    
    config.resample_velocity_step = config.ppo.episode_length // 2
    config.lin_vel_x_range = [-0.75, 0.75]
    config.lin_vel_y_range = [-0.5, 0.5]
    config.ang_vel_yaw_range = [-2.0, 2.0]
    config.zero_command_probability = 0.0
    config.stand_still_command_threshold = 0.05
    
    config.maximum_pitch_command = 0.0
    config.maximum_roll_command = 0.0
    
    config.policy = config_dict.ConfigDict()
    config.policy.hidden_layer_sizes = (256, 128, 128, 128)  # Must match pretrained checkpoint
    config.policy.activation = "elu"
    
    return config


# ============================================================================
# Collect Data with Trained Policy (PARALLELIZED)
# ============================================================================

def collect_data_with_policy(
    env,
    make_inference_fn,
    params,
    num_steps: int = 100000,
    seed: int = 42,
    num_parallel_envs: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect (obs, force) data using trained policy with parallel environments.
    
    Uses vmap to run multiple environments in parallel for much faster collection.
    OPTIMIZED: Batch transfers, minimize GPU→CPU syncs.
    """
    print(f"  Collecting {num_steps} steps of data with {num_parallel_envs} parallel envs...")
    
    inference_fn = make_inference_fn(params)
    
    # Vectorize the environment functions
    def batched_reset(rng):
        rngs = jax.random.split(rng, num_parallel_envs)
        return jax.vmap(env.reset)(rngs)
    
    def batched_step(state, action):
        return jax.vmap(env.step)(state, action)
    
    def batched_inference(obs, rng):
        rngs = jax.random.split(rng, num_parallel_envs)
        return jax.vmap(lambda o, r: inference_fn(o, r))(obs, rngs)
    
    jit_reset = jax.jit(batched_reset)
    jit_step = jax.jit(batched_step)
    jit_inference = jax.jit(batched_inference)
    
    # JIT-compiled function to compute force stats (no filtering - keep all samples)
    @jax.jit
    def compute_force_stats(force):
        """Compute force magnitudes for logging."""
        force_mags = jp.linalg.norm(force, axis=1)
        has_force = force_mags > 0.1
        return force_mags, has_force
    
    rng = jax.random.PRNGKey(seed)
    
    # Pre-allocate arrays for batch collection (reduce list appends)
    collection_batch_size = 1000  # Collect this many steps before transferring to CPU
    obs_buffer_list = []
    force_buffer_list = []
    
    # Reset all envs
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)
    
    steps_collected = 0
    total_steps = 0
    batch_obs_gpu = []
    batch_force_gpu = []
    
    print(f"    Starting parallel collection...")
    
    while steps_collected < num_steps:
        total_steps += num_parallel_envs
        
        # Get actions for all envs
        rng, act_rng = jax.random.split(rng)
        actions, _ = jit_inference(state.obs, act_rng)
        
        # Step all envs
        state = jit_step(state, actions)
        
        # Get force vector (stay on GPU)
        force_gpu = state.info.get('force_current_vector', jp.zeros((num_parallel_envs, 3)))
        
        # Compute stats for logging (no filtering - keep ALL samples including zero force)
        force_mags, has_force = compute_force_stats(force_gpu)
        
        # Accumulate ALL samples on GPU (no filtering)
        batch_obs_gpu.append(state.obs)
        batch_force_gpu.append(force_gpu)
        
        # Transfer to CPU in batches (not every step)
        if len(batch_obs_gpu) >= collection_batch_size // num_parallel_envs:
            # Concatenate on GPU then transfer ALL samples (no filtering)
            all_obs = jp.concatenate(batch_obs_gpu, axis=0)
            all_force = jp.concatenate(batch_force_gpu, axis=0)
            
            # Single GPU→CPU transfer (ALL samples, including zero force)
            obs_cpu = np.array(all_obs)
            force_cpu = np.array(all_force)
            
            obs_buffer_list.append(obs_cpu)
            force_buffer_list.append(force_cpu)
            steps_collected += len(obs_cpu)
            
            batch_obs_gpu = []
            batch_force_gpu = []
        
        # Reset done envs (check less frequently)
        if total_steps % (num_parallel_envs * 10) == 0:
            done = np.array(state.done)
            if np.any(done):
                rng, reset_rng = jax.random.split(rng)
                state = jit_reset(reset_rng)
        
        # Print progress every ~50k steps
        if total_steps % (50000 // num_parallel_envs * num_parallel_envs) < num_parallel_envs:
            avg_force = float(jp.mean(force_mags))
            has_force_pct = float(jp.mean(has_force.astype(jp.float32))) * 100
            print(f"    Collected {steps_collected}/{num_steps} samples "
                  f"(total steps: {total_steps}, avg_force: {avg_force:.3f}, has_force: {has_force_pct:.1f}%)")
        
        # Safety check (shouldn't trigger now since we keep all samples)
        if total_steps > num_steps * 2:
            break  # Collected enough
    
    # Flush remaining GPU buffer (keep ALL samples including zero force)
    if batch_obs_gpu:
        all_obs = jp.concatenate(batch_obs_gpu, axis=0)
        all_force = jp.concatenate(batch_force_gpu, axis=0)
        obs_cpu = np.array(all_obs)
        force_cpu = np.array(all_force)
        obs_buffer_list.append(obs_cpu)
        force_buffer_list.append(force_cpu)
    
    if len(obs_buffer_list) == 0:
        print("    ERROR: No samples collected! Forces may all be zero.")
        return np.array([]), np.array([])
    
    # Concatenate all collected data
    all_obs = np.concatenate(obs_buffer_list, axis=0)
    all_forces = np.concatenate(force_buffer_list, axis=0)
    
    # Trim to exact number requested
    if len(all_obs) > num_steps:
        all_obs = all_obs[:num_steps]
        all_forces = all_forces[:num_steps]
    
    print(f"    Collection complete: {len(all_obs)} samples")
    return all_obs, all_forces


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Joint training: Force Estimator (supervised) + Actor (PPO)")
    
    # Paths
    parser.add_argument("--force-estimator-path", type=str, required=True,
                        help="Path to pretrained force estimator JSON")
    parser.add_argument("--actor-checkpoint-path", type=str, default=None,
                        help="Path to pretrained actor checkpoint (Brax format)")
    parser.add_argument("--model-path", type=str,
                        default="../pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.no_body.self_collision.two_iterations.xml",
                        help="Path to MuJoCo model XML")
    parser.add_argument("--output-dir", type=str, default="output_joint_supervised",
                        help="Output directory")
    
    # Training params
    parser.add_argument("--num-rounds", type=int, default=5,
                        help="Number of alternating training rounds")
    parser.add_argument("--actor-steps-per-round", type=int, default=50_000_000,
                        help="PPO training steps per round")
    parser.add_argument("--estimator-epochs-per-round", type=int, default=100,
                        help="Supervised epochs for force estimator per round")
    parser.add_argument("--data-collection-steps", type=int, default=200_000,
                        help="Steps to collect for estimator training per round")
    parser.add_argument("--data-collection-envs", type=int, default=256,
                        help="Number of parallel envs for data collection (higher = faster)")
    parser.add_argument("--estimator-lr", type=float, default=1e-4,
                        help="Learning rate for force estimator")
    parser.add_argument("--estimator-batch-size", type=int, default=2048,
                        help="Batch size for force estimator training")
    
    # Admittance
    parser.add_argument("--admittance-gains", type=str, default="0.25,0.25",
                        help="Admittance gains (x,y) in m/s per N")
    
    # Environment
    parser.add_argument("--num-envs", type=int, default=4096,
                        help="Number of parallel environments. Note: PPO has minimum batch requirements, "
                             "so very small num_envs (like 2) won't actually reduce training time much.")
    parser.add_argument("--seed", type=int, default=42)
    
    # Logging
    parser.add_argument("--wandb-key", type=str, default=None,
                        help="WandB API key")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="pupperv3-compliance-joint",
                        help="WandB project name")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable video rendering (useful for headless servers without EGL)")
    
    args = parser.parse_args()
    
    admittance_gains = tuple(float(x) for x in args.admittance_gains.split(","))
    
    # Setup output (use absolute path - required by Orbax)
    train_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = Path(f"{args.output_dir}_{train_datetime}").resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Joint Training: Force Estimator (Supervised) + Actor (PPO)")
    print("=" * 70)
    print(f"Force estimator: {args.force_estimator_path}")
    print(f"Actor checkpoint: {args.actor_checkpoint_path or 'None (training from scratch)'}")
    print(f"Admittance gains: x={admittance_gains[0]}, y={admittance_gains[1]} m/s per N")
    print(f"Num rounds: {args.num_rounds}")
    print(f"Actor steps per round: {args.actor_steps_per_round:,}")
    print(f"Estimator epochs per round: {args.estimator_epochs_per_round}")
    print(f"Data collection steps per round: {args.data_collection_steps:,}")
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
                    "force_estimator_path": args.force_estimator_path,
                    "actor_checkpoint_path": args.actor_checkpoint_path,
                    "admittance_gains": admittance_gains,
                    "num_rounds": args.num_rounds,
                    "actor_steps_per_round": args.actor_steps_per_round,
                    "estimator_epochs_per_round": args.estimator_epochs_per_round,
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
    
    # Load initial force estimator
    print("\nLoading force estimator...")
    fe_params, input_mean, input_std = load_estimator_from_json(args.force_estimator_path)
    print(f"  Input dim: {len(input_mean)}")
    
    # Track actor checkpoint path
    # Note: We only restore from the INITIAL checkpoint (morning-jazz-49) on round 0
    # For subsequent rounds, we train from scratch with improved force estimator
    # This is because our Orbax saves don't match PPO's expected checkpoint structure
    initial_actor_checkpoint = args.actor_checkpoint_path
    
    # ========================================================================
    # Training Rounds
    # ========================================================================
    
    for round_idx in range(args.num_rounds):
        print(f"\n{'='*70}")
        print(f"ROUND {round_idx + 1}/{args.num_rounds}")
        print(f"{'='*70}")
        
        round_folder = output_folder / f"round_{round_idx}"
        round_folder.mkdir(exist_ok=True)
        
        # Export current force estimator
        fe_path = round_folder / "force_estimator.json"
        export_force_estimator(fe_params, input_mean, input_std, str(fe_path))
        
        # ====================================================================
        # Phase 1: Train Actor with PPO
        # ====================================================================
        print(f"\n--- Phase 1: Train Actor (PPO, {args.actor_steps_per_round:,} steps) ---")
        
        training_config = get_training_config(args, args.actor_steps_per_round)
        
        # Create environment with current force estimator
        def create_env(**kwargs):
            return PupperV3EnvWithEstimator(
                path=args.model_path,
                reward_config=reward_config,
                action_scale=0.3,
                observation_history=20,  # Must match pretrained checkpoint (morning-jazz-49 uses 20)
                joint_lower_limits=sim_config.joint_lower_limits,
                joint_upper_limits=sim_config.joint_upper_limits,
                torso_name=sim_config.torso_name,
                foot_site_names=sim_config.foot_site_names,
                upper_leg_body_names=sim_config.upper_leg_body_names,
                lower_leg_body_names=sim_config.lower_leg_body_names,
                resample_velocity_step=training_config.resample_velocity_step,
                linear_velocity_x_range=training_config.lin_vel_x_range,
                linear_velocity_y_range=training_config.lin_vel_y_range,
                angular_velocity_range=training_config.ang_vel_yaw_range,
                zero_command_probability=training_config.zero_command_probability,
                stand_still_command_threshold=training_config.stand_still_command_threshold,
                environment_timestep=training_config.environment_dt,
                physics_timestep=sim_config.physics_dt,
                foot_radius=sim_config.foot_radius,
                force_estimator_path=str(fe_path),
                admittance_gains=admittance_gains,
                **kwargs
            )
        
        env = create_env()
        eval_env = create_env()
        
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
            seed=args.seed + round_idx,
        )
        
        # Progress function
        def progress_fn(num_steps, metrics):
            reward = metrics.get("eval/episode_reward", 0)
            reward_std = metrics.get("eval/episode_reward_std", 0)
            print(f"  Step {num_steps:,} | Reward: {reward:.2f} ± {reward_std:.2f}")
            
            if use_wandb:
                try:
                    wandb.log({
                        f"round_{round_idx}/reward": reward,
                        f"round_{round_idx}/reward_std": reward_std,
                        "round": round_idx,
                        "total_steps": round_idx * args.actor_steps_per_round + num_steps,
                    }, step=round_idx * args.actor_steps_per_round + num_steps)
                except:
                    pass
        
        # Video rendering - create fresh env inside callback to avoid tracer leak
        def policy_params_fn(current_step, make_policy, params):
            if args.no_video:
                return  # Skip video rendering
            try:
                # Create FRESH environment for video rendering (avoid tracer leak)
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
                    output_folder=str(round_folder)
                )
            except Exception as e:
                print(f"  Video rendering failed: {e}")
        
        # Checkpoint loading - only restore from initial checkpoint on round 0
        # For subsequent rounds, train from scratch (but with improved force estimator)
        checkpoint_kwargs = {}
        if round_idx == 0 and initial_actor_checkpoint:
            checkpoint_path = Path(initial_actor_checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = Path.cwd() / checkpoint_path
            if checkpoint_path.exists():
                print(f"  Restoring actor from: {checkpoint_path}")
                checkpoint_kwargs["restore_checkpoint_path"] = checkpoint_path
            else:
                print(f"  Warning: Checkpoint not found: {checkpoint_path}")
        else:
            print(f"  Training actor from scratch (round {round_idx + 1}, with updated force estimator)")
        
        # Train
        make_inference_fn, actor_params, _ = train_fn(
            environment=env,
            progress_fn=progress_fn,
            eval_env=eval_env,
            policy_params_fn=policy_params_fn,
            **checkpoint_kwargs
        )
        
        # Save actor checkpoint using Orbax (same format as PPO uses internally)
        # actor_params is (normalizer_params, policy_params) tuple from train_fn
        # NOTE: Orbax requires ABSOLUTE paths
        actor_checkpoint_path = (round_folder / "actor_checkpoint").resolve()
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        orbax_checkpointer.save(
            str(actor_checkpoint_path), 
            actor_params,
            force=True  # Overwrite if exists
        )
        print(f"  Saved actor checkpoint to {actor_checkpoint_path}")
        
        # Note: Policy JSON export skipped (use checkpoint for inference)
        # The Brax checkpoint can be loaded directly for inference
        # We still save the checkpoint for potential manual use, but don't restore from it
        
        # ====================================================================
        # Phase 2: Collect Data
        # ====================================================================
        print(f"\n--- Phase 2: Collect Data ({args.data_collection_steps:,} steps) ---")
        
        # Create a FRESH environment for data collection (avoid tracer leak from PPO)
        data_collection_env = create_env()
        
        obs_data, force_data = collect_data_with_policy(
            env=data_collection_env,
            make_inference_fn=make_inference_fn,
            params=actor_params,
            num_steps=args.data_collection_steps,
            seed=args.seed + round_idx + 1000,
            num_parallel_envs=args.data_collection_envs,
        )
        
        print(f"  Collected {len(obs_data)} samples")
        if len(force_data) > 0 and force_data.ndim == 2:
            print(f"  Force magnitude: mean={np.linalg.norm(force_data, axis=1).mean():.3f}, "
                  f"max={np.linalg.norm(force_data, axis=1).max():.3f}")
        else:
            print(f"  Warning: No valid force data collected!")
        
        # Save collected data
        data_path = round_folder / "collected_data.npz"
        np.savez(data_path, observations=obs_data, forces=force_data)
        print(f"  Saved data to {data_path}")
        
        # ====================================================================
        # Phase 3: Train Force Estimator (Supervised)
        # ====================================================================
        print(f"\n--- Phase 3: Train Force Estimator (Supervised, {args.estimator_epochs_per_round} epochs) ---")
        
        if len(obs_data) < 1000:
            print("  Warning: Not enough data collected, skipping estimator update")
        else:
            fe_params = train_force_estimator_supervised(
                params=fe_params,
                input_mean=input_mean,
                input_std=input_std,
                obs_data=obs_data,
                force_data=force_data,
                num_epochs=args.estimator_epochs_per_round,
                batch_size=args.estimator_batch_size,
                learning_rate=args.estimator_lr,
            )
            
            # Save updated estimator
            fe_updated_path = round_folder / "force_estimator_updated.json"
            export_force_estimator(fe_params, input_mean, input_std, str(fe_updated_path))
        
        if use_wandb:
            try:
                force_mean = np.linalg.norm(force_data, axis=1).mean() if (len(force_data) > 0 and force_data.ndim == 2) else 0
                wandb.log({
                    f"round_{round_idx}/data_collected": len(obs_data),
                    f"round_{round_idx}/force_mean": force_mean,
                }, step=(round_idx + 1) * args.actor_steps_per_round)
            except:
                pass
        
        print(f"\n  Round {round_idx + 1} complete!")
    
    # ========================================================================
    # Final Export
    # ========================================================================
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    
    # Export final models
    final_fe_path = output_folder / "force_estimator_final.json"
    export_force_estimator(fe_params, input_mean, input_std, str(final_fe_path))
    
    print(f"\nFinal outputs:")
    print(f"  Force estimator: {final_fe_path}")
    print(f"  Actor checkpoint: {output_folder}/round_{args.num_rounds - 1}/actor_checkpoint")
    print(f"  Output folder: {output_folder}")
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
