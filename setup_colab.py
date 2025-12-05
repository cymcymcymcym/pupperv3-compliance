"""
Google Colab Environment Setup for pupperv3-compliance

Usage in Colab:
    1. Upload this file to Colab or copy/paste into a cell
    2. Run: exec(open('setup_colab.py').read())
    
Or just copy the cell below into your notebook.
"""

# ============================================================
# COPY THIS ENTIRE CELL INTO YOUR COLAB NOTEBOOK
# ============================================================

import subprocess
import os
import sys

def run_cmd(cmd, check=True):
    """Run shell command and print output."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if check and result.returncode != 0:
        print(f"Warning: Command returned {result.returncode}")
    return result.returncode

print("=" * 60)
print("Setting up pupperv3-compliance environment for Google Colab")
print("=" * 60)

# Step 1: Check GPU
print("\n[1/8] Checking GPU...")
run_cmd("nvidia-smi", check=False)

# Step 2: Install base dependencies
print("\n[2/8] Installing base MuJoCo and Brax packages...")
run_cmd("pip install -q mujoco==3.2.7 mujoco-mjx==3.2.7 brax==0.10.5 flax==0.10.2")

# Step 3: Clean up incompatible JAX versions
print("\n[3/8] Cleaning up incompatible JAX versions...")
run_cmd("pip uninstall -y jax jaxlib optax orbax-checkpoint 2>/dev/null || true", check=False)

# Step 4: Install specific compatible JAX versions
print("\n[4/8] Installing compatible JAX versions with CUDA support...")
run_cmd("pip install -q optax==0.2.2 orbax-checkpoint==0.11.10")
run_cmd('pip install -q "jax[cuda12]==0.5.0" "jax-cuda12-plugin==0.5.0" "jax-triton==0.2.0"')

# Step 5: Install media and plotting dependencies
print("\n[5/8] Installing media and plotting dependencies...")
run_cmd("apt-get update -qq && apt-get install -qq -y ffmpeg > /dev/null 2>&1 || true", check=False)
run_cmd("pip install -q mediapy matplotlib plotly wandb")

# Step 6: Clone pupper_v3_description (robot model)
print("\n[6/8] Cloning pupper_v3_description (robot model)...")
run_cmd("rm -rf pupper_v3_description")
run_cmd("git clone -q https://github.com/g-levine/pupper_v3_description -b master")

# Step 7: Clone and install pupperv3-mjx
print("\n[7/8] Cloning and installing pupperv3-mjx...")
run_cmd("rm -rf pupperv3_mjx")
run_cmd("git clone -q -b apply_constant_force --single-branch https://github.com/cymcymcymcym/pupperv3-mjx.git pupperv3_mjx")
run_cmd("pip install -q ./pupperv3_mjx")

# Step 8: Setup EGL for headless GPU rendering
print("\n[8/8] Configuring EGL for GPU rendering...")
NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
    os.makedirs(os.path.dirname(NVIDIA_ICD_CONFIG_PATH), exist_ok=True)
    with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
        f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")
    print(f"Created {NVIDIA_ICD_CONFIG_PATH}")
else:
    print(f"{NVIDIA_ICD_CONFIG_PATH} already exists")

# Set environment variables
os.environ['MUJOCO_GL'] = 'egl'
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

print("\n" + "=" * 60)
print("Setup complete!")
print("=" * 60)

# Verify installation
print("\nVerifying installation...")
try:
    import mujoco
    mujoco.MjModel.from_xml_string('<mujoco/>')
    print("✓ MuJoCo OK")
except Exception as e:
    print(f"✗ MuJoCo failed: {e}")

try:
    import jax
    print(f"✓ JAX version: {jax.__version__}")
    print(f"✓ JAX devices: {jax.devices()}")
except Exception as e:
    print(f"✗ JAX failed: {e}")

try:
    from pupperv3_mjx import environment
    print("✓ pupperv3_mjx imported successfully")
except Exception as e:
    print(f"✗ pupperv3_mjx import failed: {e}")

print("\n" + "=" * 60)
print("Environment variables:")
print(f"  MUJOCO_GL = {os.environ.get('MUJOCO_GL', 'not set')}")
print("=" * 60)
print("\nYou can now run training scripts!")
print("Example:")
print("  !python train_joint_force_actor.py \\")
print("      --force-estimator-path force_estimator.json \\")
print("      --model-path pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.no_body.self_collision.two_iterations.xml \\")
print("      --admittance-gains 0.25,0.25 \\")
print("      --no-wandb")












