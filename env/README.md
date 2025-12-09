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

This will create the environment at `/home/{USERID}/.conda/envs/tcrdock_env2`.

## 3. Activate the environment
```bash
source activate tcrdock_env2
```

## 4. Install GPU-enabled jaxlib

**CRITICAL**: The YAML file installs CPU-only jaxlib. You must upgrade it to the GPU version:

```bash
pip uninstall jaxlib -y
pip install "jaxlib==0.4.14+cuda12.cudnn89" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## 5. Apply CUDA library fix
Create a file named `activate.sh` inside your environment directory:
```bash
nano /home/{USERID}/.conda/envs/tcrdock_env2/activate.sh
```

Paste this inside:
```bash
#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/{USERID}/.conda/envs/tcrdock_env2/lib
```

Save and exit (Ctrl+O, Enter, Ctrl+X), then make it executable:
```bash
chmod +x /home/{USERID}/.conda/envs/tcrdock_env2/activate.sh
```

## 6. Activate the environment in SLURM scripts

For GPU jobs (prediction), use this exact order:
```bash
module load mamba/latest
module load cuda-12.4.1-gcc-12.1.0
source /packages/apps/mamba/0.23.3/etc/profile.d/conda.sh
source activate tcrdock_env2

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```

For CPU jobs (setup, relabeling), use:
```bash
module load mamba/latest
source /packages/apps/mamba/0.23.3/etc/profile.d/conda.sh
source activate tcrdock_env2
source /home/{USERID}/.conda/envs/tcrdock_env2/activate.sh
```

## 7. Verify GPU setup (optional but recommended)

Request an interactive GPU session:
```bash
salloc --partition=htc --qos=public --gres=gpu:1 --time=00:10:00
```

Once on the GPU node, test:
```bash
module load mamba/latest
module load cuda-12.4.1-gcc-12.1.0
source activate tcrdock_env2

python << 'EOF'
import jax
print("JAX version:", jax.__version__)
print("Available devices:", jax.devices())
print("GPU available:", len(jax.devices("gpu")) > 0)
EOF
```

Expected output should show `GPU available: True` and list GPU device(s).

Exit the interactive session:
```bash
exit
```

## Done!

Your environment is now ready for TCRdock pipeline.
