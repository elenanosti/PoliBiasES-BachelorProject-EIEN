#!/bin/bash
#SBATCH --job-name=main 
#SBATCH --account=eei440                
#SBATCH --nodes=1                                        
#SBATCH --time=0-00:15:00

# === NEW: Set cache directory to scratch space ===
# === This is necessary to avoid running out of disk space on the home directory ===
# Set up local storage on the compute node
export LOCAL_SCRATCH="/local/$USER"
mkdir -p "$LOCAL_SCRATCH"

# Download model or data
# wget -O "$LOCAL_SCRATCH/model.tar.gz" "https://your-direct-download-link"
# tar -xzvf "$LOCAL_SCRATCH/model.tar.gz" -C "$LOCAL_SCRATCH"

# Point your code to use the model/data from $LOCAL_SCRATCH
export HF_HOME="$LOCAL_SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$LOCAL_SCRATCH/hf_cache/transformers"
export HF_DATASETS_CACHE="$LOCAL_SCRATCH/hf_cache/datasets"
export TMPDIR="$LOCAL_SCRATCH/tmp"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"
# === END: Set cache directory to scratch space ===E


# -------------------- MODULE SETUP --------------------
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/

# module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
# module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8
# module load nlpl-tokenizers/0.21.0-foss-2022b-Python-3.10.8
# module load nlpl-accelerate/0.34.2-foss-2022b-Python-3.10.8
# module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8
# module load nlpl-llmtools/06-foss-2022b-Python-3.10.8
# module load JupyterLab/4.0.3-GCCcore-12.2.0

# -------------------- PARAMETERS --------------------
MODELS=("Llama-3-70B-Instruct") # ("Llama-3-8B-instruct", , "Mistral-7B-instruct", "Falcon3-7B-instruct", "Gemma-2-9B-instruct", "deepseek-llm-7b-chat")
PROMPTS=(1)
PROMPT_TEMPLATES=(0)
#REPLACES=(2)

# -------------------- EXPERIMENT LOOP --------------------
for model in "${MODELS[@]}"; do
  for prompt in "${PROMPTS[@]}"; do
    for prompt_template in "${PROMPT_TEMPLATES[@]}"; do
      #for replace in "${REPLACES[@]}"; do
        echo "Running with model=$model, prompt=$prompt, prompt_template=$prompt_template" #, replace=$replace"
        python3 -u running_experiments.py \
          --exp=ide \
          --model="$model" \
          --prompt="$prompt" \
          --debug=1 \
          --datasize=200
          #--replace="$replace" \
      #done
    done
  done
done
