#!/bin/bash
#SBATCH --job-name=llm_array
#SBATCH --account=eei440                
#SBATCH --nodes=1                                        
#SBATCH --time=0-06:00:00
#SBATCH --array=0-3

# === NEW: Set cache directory to scratch space ===
export HF_HOME=/var/scratch/eei440/hf_cache
mkdir -p "$HF_HOME"
# === END: Set cache directory to scratch space ===

# -------------------- MODULE SETUP --------------------
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/

# Uncomment and load your modules as needed
# module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
# module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8
# module load nlpl-tokenizers/0.21.0-foss-2022b-Python-3.10.8
# module load nlpl-accelerate/0.34.2-foss-2022b-Python-3.10.8
# module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8
# module load nlpl-llmtools/06-foss-2022b-Python-3.10.8
# module load JupyterLab/4.0.3-GCCcore-12.2.0

# -------------------- PARAMETERS --------------------
MODELS=("Llama-3-8B-instruct" "Mistral-7B-instruct" "Gemma-2-9B-instruct" "Falcon3-7B-instruct")
PROMPTS=(1)
PROMPT_TEMPLATES=(0)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# -------------------- EXPERIMENT --------------------
for prompt in "${PROMPTS[@]}"; do
  for prompt_template in "${PROMPT_TEMPLATES[@]}"; do
    echo "Running with model=$MODEL, prompt=$prompt, prompt_template=$prompt_template"
    python3 -u running_experiments.py \
      --exp=ide \
      --model="$MODEL" \
      --prompt="$prompt" \
      --debug=1 \
      --datasize=200 \
      --prompt_template="$prompt_template"
  done
done