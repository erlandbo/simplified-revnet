#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=<YOUR-ACCOUNT-HERE>
#SBATCH --time=12-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10     # 10 CPU cores
#SBATCH --mem=32G              # 32 gigabytes of system memory (not GPU memory)
#SBATCH --gres=gpu:1  
#SBATCH --job-name="litgpt-srevnet-mresnet"

#SBATCH --gres=gpu:h100:1

#SBATCH --constraint=gpu80g

module purge
module load Python/3.12.3-GCCcore-13.3.0

source venv/bin/activate

nvidia-smi

litgpt pretrain --config config_hub/pretrain/microllama.yaml --data OpenWebText --out_dir out-owt-microllama --tokenizer_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train.micro_batch_size 16 --recompute_op reversible-momentum --gamma 0.99 --use_backprop False --train.max_tokens 4_000_000_000 --precision 32-true --train.global_batch_size 512
litgpt pretrain --config config_hub/pretrain/microllama.yaml --data OpenWebText --out_dir out-owt-microllama --tokenizer_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train.micro_batch_size 16 --recompute_op reversible-momentum --gamma 0.95 --use_backprop False --train.max_tokens 4_000_000_000 --precision 32-true --train.global_batch_size 512
litgpt pretrain --config config_hub/pretrain/microllama.yaml --data OpenWebText --out_dir out-owt-microllama --tokenizer_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train.micro_batch_size 16 --recompute_op reversible-momentum --gamma 0.75 --use_backprop False --train.max_tokens 4_000_000_000 --precision 32-true --train.global_batch_size 512
litgpt pretrain --config config_hub/pretrain/microllama.yaml --data OpenWebText --out_dir out-owt-microllama --tokenizer_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train.micro_batch_size 16 --recompute_op reversible-momentum --gamma 0.5 --use_backprop False --train.max_tokens 4_000_000_000 --precision 32-true --train.global_batch_size 512
litgpt pretrain --config config_hub/pretrain/microllama.yaml --data OpenWebText --out_dir out-owt-microllama --tokenizer_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train.micro_batch_size 16 --recompute_op reversible-momentum --gamma 0.25 --use_backprop False --train.max_tokens 4_000_000_000 --precision 32-true --train.global_batch_size 512
litgpt pretrain --config config_hub/pretrain/microllama.yaml --data OpenWebText --out_dir out-owt-microllama --tokenizer_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --train.micro_batch_size 16 --recompute_op reversible-momentum --gamma 0.1 --use_backprop False --train.max_tokens 4_000_000_000 --precision 32-true --train.global_batch_size 512
