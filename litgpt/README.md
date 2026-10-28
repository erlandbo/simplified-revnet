For more details on LitGPT, see `README_original.md` or visit the official repository at https://github.com/Lightning-AI/litgpt. All experiments were run on a single NVIDIA H100 GPU (80 GB VRAM) on SLURM.

# Setup
Create an environment with python version 3.12 (see `README_original.md` for the supported Python versions)

Install LitGPT:
```
pip install -e ".[extra,compiler,test]"
```
Download the tokenizer corresponding to TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T:
```
litgpt download \
   TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
   --tokenizer_only true
```

# Main results
Downloading the OpenWebText datasets may take some time the first time.

Train models on SLURM:
```
sbatch run_srevnet.slurm
```
or train locally:
```
./run_srevnet.sh > run_srevnet_log.log
```
# Additional MResNet momentum rate ablations
To train MResNet with a momentum value other than the default (0.9), submit the following SLURM job:
```
sbatch run_srevnet_mresnet.slurm
```
or train locally:
```
./run_srevnet_mresnet.sh > run_srevnet_mresnet_log.log
```