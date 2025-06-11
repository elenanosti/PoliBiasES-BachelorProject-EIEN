#!/bin/bash
#SBATCH --job-name=main 
#SBATCH --account=eei440                
#SBATCH --nodes=1                                        
#SBATCH --time=0-00:15:00

# === NEW: Set cache directory to scratch space ===
export HF_HOME=/var/scratch/eei440/hf_cache
mkdir -p "$HF_HOME"
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
MODELS=("Falcon3-7B-instruct") # ("Llama-3-8B-instruct", "Llama-3-70B-Instruct" "Mistral-7B-instruct" "Gemma-2-9B-instruct" "Falcon3-7B-instruct" "deepseek-llm-7b-chat")
PROMPTS=(1)
PROMPT_TEMPLATES=(0)
#REPLACES=(2)

# -------------------- EXPERIMENT LOOP --------------------
for model in "${MODELS[@]}"; do
  for prompt in "${PROMPTS[@]}"; do
    for prompt_template in "${PROMPT_TEMPLATES[@]}"; do
      #for replace in "${REPLACES[@]}"; do
        echo "Running with model=$model, prompt=$prompt, prompt_template=$prompt_template, replace=$replace"
        python3 -u running_experiments.py \
          --exp=ide \
          --model="$model" \
          --prompt="$prompt" \
          #--replace="$replace" \
          --debug=1 \
          --datasize=200
      #done
    done
  done
done
