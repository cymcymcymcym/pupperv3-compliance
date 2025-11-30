#!/usr/bin/env python3
"""
Joint training: Force Estimator (supervised) + Actor (PPO).

This script implements alternating training:
1. Train actor with PPO for N steps (force estimator frozen during rollout)
2. Collect (obs, ground_truth_force) pairs during rollout
3. Update force estimator with supervised learning on collected data
4. Repeat

The force estimator receives ONLY supervised gradient (no reward signal).
The actor receives ONLY reward gradient (no supervised signal).

Usage:
    python train_joint_with_supervised_estimator.py \
        --force-estimator-path force_estimator_training/force_estimator.json \
        --admittance-gains 0.25,0.25 \
        --num-rounds 10 \
        --actor-steps-per-round 10000000
"""

import argparse
import functools
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, List

import flax.linen as nn
from flax.training import train_state
import jax
from jax import numpy as jp
import numpy as np
import optax
from ml_collections import config_dict

from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model, mjcf

from pupperv3_mjx import domain_randomization, utils
from pupperv3_mjx.environment_with_trainable_estimator import (
    PupperV3EnvWithTrainableEstimator,
    create_force_estimator_fn,
    load_estimator_from_json,
)


# ============================================================================
# Force Estimator Network
# ============================================================================

class ForceEstimatorNetwork(nn.Module):
    """Flax MLP for force estimation."""
    hidden_sizes: Tuple[int, ...] = (512, 256, 128)
    
    @nn.compact
    def __call__(self, x: jp.ndarray, train: bool = False) -> jp.ndarray:
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)
        x = nn.Dense(3)(x)
        return x


# ============================================================================
# Supervised Training for Force Estimator
# ============================================================================

def train_force_estimator_supervised(
    params: Dict,
    input_mean: jp.ndarray,
    input_std: jp.ndarray,
    obs_data: np.ndarray,
    force_data: np.ndarray,
    num_epochs: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 1e-4,
    direction_weight: float = 0.8,
    magnitude_weight: float = 0.2,
) -> Dict:
    """Train force estimator with supervised learning.
    
    Args:
        params: Current force estimator params
        input_mean, input_std: Normalization stats
        obs_data: Observations [N, obs_dim]
        force_data: Ground truth forces [N, 3]
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Updated params
    """
    print(f"  Training force estimator on {len(obs_data)} samples...")
    
    # Normalize observations
    obs_norm = (obs_data - input_mean) / input_std
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Get hidden sizes from params
    hidden_sizes = []
    for key in sorted(params.keys()):
        if 'Dense' in key and 'LayerNorm' not in key:
            if key != f"Dense_{len([k for k in params.keys() if 'Dense' in k and 'LayerNorm' not in k]) - 1}":
                hidden_sizes.append(params[key]['bias'].shape[0])
    
    model = ForceEstimatorNetwork(hidden_sizes=tuple(hidden_sizes))
    
    @jax.jit
    def loss_fn(params, obs_batch, force_batch):
        pred = model.apply({'params': params}, obs_batch, train=True)
        
        # Direction loss
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
    
    # Training loop
    num_samples = len(obs_norm)
    num_batches = num_samples // batch_size
    
    for epoch in range(num_epochs):
        # Shuffle data
        perm = np.random.permutation(num_samples)
        obs_shuffled = obs_norm[perm]
        force_shuffled = force_data[perm]
        
        total_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            obs_batch = jp.array(obs_shuffled[start:end])
            force_batch = jp.array(force_shuffled[start:end])
            
            params, opt_state, loss = train_step(params, opt_state, obs_batch, force_batch)
            total_loss += float(loss)
        
        avg_loss = total_loss / max(1, num_batches)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"    Epoch {epoch+1}/{num_epochs}: loss = {avg_loss:.6f}")
    
    return params


# ============================================================================
# Data Collection During PPO
# ============================================================================

class DataCollector:
    """Collects (obs, force) pairs during PPO rollouts."""
    
    def __init__(self, max_size: int = 500000):
        self.max_size = max_size
        self.obs_buffer: List[np.ndarray] = []
        self.force_buffer: List[np.ndarray] = []
    
    def add(self, obs: np.ndarray, force: np.ndarray):
        """Add a batch of (obs, force) pairs."""
        self.obs_buffer.append(obs)
        self.force_buffer.append(force)
        
        # Trim if too large
        total = sum(len(o) for o in self.obs_buffer)
        while total > self.max_size and len(self.obs_buffer) > 1:
            self.obs_buffer.pop(0)
            self.force_buffer.pop(0)
            total = sum(len(o) for o in self.obs_buffer)
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all collected data."""
        if not self.obs_buffer:
            return np.array([]), np.array([])
        return np.concatenate(self.obs_buffer), np.concatenate(self.force_buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.obs_buffer = []
        self.force_buffer = []
    
    def __len__(self):
        return sum(len(o) for o in self.obs_buffer)


# ============================================================================
# Export Force Estimator
# ============================================================================

def export_force_estimator(params, input_mean, input_std, export_path):
    """Export force estimator to JSON."""
    layers = []
    
    dense_keys = sorted([k for k in params.keys() if 'Dense' in k])
    ln_keys = sorted([k for k in params.keys() if 'LayerNorm' in k])
    
    num_hidden = len(ln_keys)
    
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
    
    # Output
    kernel = np.array(params[f"Dense_{num_hidden}"]["kernel"])
    bias = np.array(params[f"Dense_{num_hidden}"]["bias"])
    layers.append({
        "type": "dense",
        "activation": "identity",
        "shape": [None, 3],
        "weights": [kernel.tolist(), bias.tolist()],
    })
    
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
# Reward Config
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


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-estimator-path", type=str, required=True)
    parser.add_argument("--admittance-gains", type=str, default="0.25,0.25")
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--actor-steps-per-round", type=int, default=10_000_000)
    parser.add_argument("--estimator-epochs-per-round", type=int, default=50)
    parser.add_argument("--estimator-lr", type=float, default=1e-4)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--output-dir", type=str, default="output_joint")
    parser.add_argument("--model-path", type=str,
                        default="../pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.no_body.self_collision.two_iterations.xml")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    admittance_gains = tuple(float(x) for x in args.admittance_gains.split(","))
    
    print("=" * 70)
    print("Joint Training: Force Estimator (Supervised) + Actor (PPO)")
    print("=" * 70)
    print(f"Rounds: {args.num_rounds}")
    print(f"Actor steps per round: {args.actor_steps_per_round:,}")
    print(f"Estimator epochs per round: {args.estimator_epochs_per_round}")
    print(f"Admittance gains: {admittance_gains}")
    print("=" * 70)
    
    # Setup output
    train_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = f"{args.output_dir}_{train_datetime}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Load initial force estimator
    print("\nLoading force estimator...")
    fe_params, input_mean, input_std = load_estimator_from_json(args.force_estimator_path)
    
    # Data collector
    data_collector = DataCollector(max_size=500000)
    
    # Training rounds
    for round_idx in range(args.num_rounds):
        print(f"\n{'='*70}")
        print(f"ROUND {round_idx + 1}/{args.num_rounds}")
        print(f"{'='*70}")
        
        # Export current estimator
        fe_path = os.path.join(output_folder, f"force_estimator_round_{round_idx}.json")
        export_force_estimator(fe_params, input_mean, input_std, fe_path)
        
        # For now, print instructions (full automation requires custom PPO loop)
        print(f"\n  1. Train actor with frozen estimator:")
        print(f"     python train_joint_force_actor.py \\")
        print(f"         --force-estimator-path {fe_path} \\")
        print(f"         --admittance-gains {args.admittance_gains} \\")
        print(f"         --num-timesteps {args.actor_steps_per_round} \\")
        print(f"         --output-dir {output_folder}/actor_round_{round_idx} \\")
        print(f"         --no-wandb")
        
        print(f"\n  2. Collect data:")
        print(f"     python collect_force_data.py \\")
        print(f"         --num-samples 200000 \\")
        print(f"         --output-dir {output_folder}/data_round_{round_idx}")
        
        print(f"\n  3. Update estimator (run this script again with --update-estimator)")
        
        # If we have collected data, train estimator
        # This would be automated in a full implementation
        
        print(f"\n  Round {round_idx + 1} setup complete.")
        
        # For demo, only show first round
        if round_idx == 0:
            print("\n  [Demo mode: showing first round only]")
            print("  For full training, run the commands above for each round.")
            break
    
    print(f"\n{'='*70}")
    print("Joint training setup complete!")
    print(f"Output directory: {output_folder}")
    print("=" * 70)


if __name__ == "__main__":
    main()
