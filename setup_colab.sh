#!/bin/bash
# ============================================================
# Google Colab Environment Setup Script for pupperv3-compliance
# ============================================================
# Usage: 
#   1. Upload this script to Colab
#   2. Run: !bash setup_colab.sh
#   3. Then run your training script
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "Setting up pupperv3-compliance environment for Google Colab"
echo "============================================================"

# Step 1: Install base dependencies
echo ""
echo "[1/7] Installing base MuJoCo and Brax packages..."
pip install mujoco==3.2.7 mujoco-mjx==3.2.7 brax==0.10.5 flax==0.10.2

# Step 2: Clean up incompatible JAX versions (Colab may have newer versions)
echo ""
echo "[2/7] Cleaning up incompatible JAX versions..."
pip uninstall -y jax jaxlib optax orbax-checkpoint 2>/dev/null || true

# Step 3: Install specific compatible JAX versions for CUDA
echo ""
echo "[3/7] Installing compatible JAX versions with CUDA support..."
pip install optax==0.2.2 orbax-checkpoint==0.11.10
pip install "jax[cuda12]==0.5.0" "jax-cuda12-plugin==0.5.0" "jax-triton==0.2.0"

# Step 4: Install media and plotting dependencies
echo ""
echo "[4/7] Installing media and plotting dependencies..."
apt-get update && apt-get install -y ffmpeg > /dev/null 2>&1 || true
pip install mediapy matplotlib plotly wandb

# # Step 5: Clone pupper_v3_description (robot model)
# echo ""
# echo "[5/7] Cloning pupper_v3_description (robot model)..."
# rm -rf pupper_v3_description
# git clone -q https://github.com/g-levine/pupper_v3_description -b master

# # Step 6: Clone and install pupperv3-mjx
# echo ""
# echo "[6/7] Cloning and installing pupperv3-mjx..."
# rm -rf pupperv3_mjx
# git clone -q -b apply_constant_force --single-branch https://github.com/cymcymcymcym/pupperv3-mjx.git pupperv3_mjx
# pip install -q ./pupperv3_mjx

# Step 7: Setup OSMesa for headless CPU rendering (more compatible than EGL)
echo ""
echo "[7/7] Installing OSMesa for headless rendering..."
apt-get install -y libosmesa6-dev > /dev/null 2>&1 || true

# Set environment variables
# Using OSMesa (software rendering) instead of EGL for better compatibility
# EGL requires specific GPU driver support that may not be available
export MUJOCO_GL=osmesa
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_triton_gemm_any=True"

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "Environment variables set:"
echo "  MUJOCO_GL=osmesa (software rendering, always works)"
echo "  XLA_FLAGS includes --xla_gpu_triton_gemm_any=True"
echo ""
echo "To verify installation, run:"
echo "  python -c \"import mujoco; import jax; print('JAX version:', jax.__version__); print('MuJoCo OK')\""
echo ""
echo "To train, run:"
echo "  python train_joint_force_actor.py --force-estimator-path /path/to/force_estimator.json --no-wandb"
echo "============================================================"


