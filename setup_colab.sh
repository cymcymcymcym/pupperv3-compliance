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
pip install -q mujoco==3.2.7 mujoco-mjx==3.2.7 brax==0.10.5 flax==0.10.2

# Step 2: Clean up incompatible JAX versions (Colab may have newer versions)
echo ""
echo "[2/7] Cleaning up incompatible JAX versions..."
pip uninstall -y jax jaxlib optax orbax-checkpoint 2>/dev/null || true

# Step 3: Install specific compatible JAX versions for CUDA
echo ""
echo "[3/7] Installing compatible JAX versions with CUDA support..."
pip install -q optax==0.2.2 orbax-checkpoint==0.11.10
pip install -q "jax[cuda12]==0.5.0" "jax-cuda12-plugin==0.5.0" "jax-triton==0.2.0"

# Step 4: Install media and plotting dependencies
echo ""
echo "[4/7] Installing media and plotting dependencies..."
apt-get update -qq && apt-get install -qq -y ffmpeg > /dev/null 2>&1 || true
pip install -q mediapy matplotlib plotly wandb

# Step 5: Clone pupper_v3_description (robot model)
echo ""
echo "[5/7] Cloning pupper_v3_description (robot model)..."
rm -rf pupper_v3_description
git clone -q https://github.com/g-levine/pupper_v3_description -b master

# Step 6: Clone and install pupperv3-mjx
echo ""
echo "[6/7] Cloning and installing pupperv3-mjx..."
rm -rf pupperv3_mjx
git clone -q -b apply_constant_force --single-branch https://github.com/cymcymcymcym/pupperv3-mjx.git pupperv3_mjx
pip install -q ./pupperv3_mjx

# Step 7: Setup EGL for headless GPU rendering
echo ""
echo "[7/7] Configuring EGL for GPU rendering..."
NVIDIA_ICD_CONFIG_PATH='/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
if [ ! -f "$NVIDIA_ICD_CONFIG_PATH" ]; then
    mkdir -p /usr/share/glvnd/egl_vendor.d/
    cat > "$NVIDIA_ICD_CONFIG_PATH" << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
fi

# Set environment variables
export MUJOCO_GL=egl
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_triton_gemm_any=True"

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "Environment variables set:"
echo "  MUJOCO_GL=egl"
echo "  XLA_FLAGS includes --xla_gpu_triton_gemm_any=True"
echo ""
echo "To verify installation, run:"
echo "  python -c \"import mujoco; import jax; print('JAX version:', jax.__version__); print('MuJoCo OK')\""
echo ""
echo "To train, run:"
echo "  python train_joint_force_actor.py --force-estimator-path /path/to/force_estimator.json --no-wandb"
echo "============================================================"

