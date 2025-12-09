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

## 3. Activate the environment
```bash
source activate tcrdock_env
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
nano /path/to/tcrdock_env/activate.sh
```

Paste this inside:
```bash
#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/tcrdock_env/lib
```

Save and exit, then make it executable:
```bash
chmod +x /path/to/tcrdock_env/activate.sh
```
