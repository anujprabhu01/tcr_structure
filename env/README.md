# TCRdock Environment Setup (Sol HPC)

## 1. Load mamba
Run this on ASU's Sol HPC:
```bash
module load mamba/latest
```

## 2. Create the environment
```bash
mamba env create -f tcrdock_env.yaml
```

> **Note**: By default, this creates the environment in `/home/{USERID}/.conda/envs/tcrdock_env2`. Using home directory is recommended over scratch for conda environments.

## 3. Install GPU-enabled jaxlib

**CRITICAL STEP**: The YAML installs CPU-only jaxlib. You must upgrade to GPU version:

```bash
source activate tcrdock_env2

# Uninstall CPU-only jaxlib
pip uninstall jaxlib -y

# Install GPU-enabled jaxlib
pip install "jaxlib==0.4.14+cuda12.cudnn89" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## 4. Apply CUDA library fix
Create a file named `activate.sh` inside your environment directory:
```bash
nano /home/{USERID}/.conda/envs/tcrdock_env2/activate.sh
```

Paste this inside:
```bash
#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/{USERID}/.conda/envs/tcrdock_env2/lib
```

Make it executable:
```bash
chmod +x /home/{USERID}/.conda/envs/tcrdock_env2/activate.sh
```

## 5. Using in SLURM scripts

For GPU jobs (e.g., AlphaFold prediction), use this **exact** activation order:

```bash
# Load modules in this order (mamba first, then CUDA)
module load mamba/latest
module load cuda-12.4.1-gcc-12.1.0

# Activate environment
source /packages/apps/mamba/0.23.3/etc/profile.d/conda.sh
source activate tcrdock_env2

# Set GPU environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```

For CPU jobs (e.g., setup, relabeling):

```bash
module load mamba/latest
source /packages/apps/mamba/0.23.3/etc/profile.d/conda.sh
source activate tcrdock_env2
source /home/{USERID}/.conda/envs/tcrdock_env2/activate.sh
```

## 6. Verify installation

**Basic check (CPU node):**
```bash
source activate tcrdock_env2

python << 'EOF'
import jax
print("JAX version:", jax.__version__)
print("JAXlib version:", jax.__version__)

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import pandas
print("Pandas: OK")

from Bio import SeqIO
print("BioPython: OK")
EOF
```

**GPU check (requires GPU node):**
```bash
# Request interactive GPU session
salloc --partition=htc --qos=public --gres=gpu:1 --time=00:10:00

# Once on GPU node:
module load mamba/latest
module load cuda-12.4.1-gcc-12.1.0
source activate tcrdock_env2

python << 'EOF'
import jax
print("JAX version:", jax.__version__)
print("Available devices:", jax.devices())
print("GPU count:", len(jax.devices("gpu")))
EOF

exit  # Exit interactive session
```

Expected output should show at least one GPU device.

## Troubleshooting

### GPU not detected
- Verify GPU-enabled jaxlib is installed: `pip show jaxlib | grep Version`
  - Should show: `0.4.14+cuda12.cudnn89`
  - If just `0.4.14`, repeat step 3
- Make sure you're on a GPU node: `nvidia-smi` should show GPU
- Check module load order: mamba → cuda → activate

### Missing dependencies
```bash
mamba activate tcrdock_env2
mamba install python-dateutil pytz -y
```

### "Could not find cuda drivers"
- Load CUDA module **before** activating conda
- Use `--export=ALL` in sbatch (not `--export=NONE`)

## Complete YAML

Your `tcrdock_env.yaml`:

```yaml
name: tcrdock_env2
channels:
  - conda-forge
  - nvidia
  - defaults
dependencies:
  - python=3.10
  - numpy=1.*
  - scipy
  - pandas
  - python-dateutil
  - pytz
  - tensorflow-cpu=2.12
  - cudnn=8.9
  - cuda-nvcc=12.0
  - cuda-runtime=12.0
  - pip
  - pip:
      - jax==0.4.14
      - jaxlib==0.4.14  # Upgraded to GPU version in step 3
      - biopython==1.79
      - chex==0.1.7
      - dm-haiku==0.0.10
      - dm-tree==0.1.8
      - immutabledict==2.0.0
      - matplotlib==3.10.3
      - ml-collections==0.1.0
      - docker==5.0.0
      - optax==0.2.1
      - optree==0.15.0
      - orbax-checkpoint==0.3.4
      - jmp==0.0.4
      - humanize==4.12.3
      - pillow==11.2.1
      - rich==14.0.0
      - protobuf==4.21.12
```
