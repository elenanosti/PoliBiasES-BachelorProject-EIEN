#!/bin/bash
#SBATCH --job-name=main 
#SBATCH --account=eei440                
#SBATCH --nodes=1                                        
#SBATCH --time=24:00:00
#SBATCH --array=0

# === NEW: Set cache directory to scratch space ===
export LOCAL_SCRATCH="/local/$USER"
mkdir -p "$LOCAL_SCRATCH"

export HF_HOME="$LOCAL_SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$LOCAL_SCRATCH/hf_cache/transformers"
export HF_DATASETS_CACHE="$LOCAL_SCRATCH/hf_cache/datasets"
export TMPDIR="$LOCAL_SCRATCH/tmp"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"
# === END: Set cache directory to scratch space ===

# -------------------- MODULE SETUP --------------------
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/

# -------------------- PARAMETERS --------------------
MODELS=("Gemma-2-9B-instruct") # ("Falcon3-7B-instruct""Llama-2-7b""Aguila-7B-instruct" "Llama-3-70B-Instruct" "Llama-3-8B-instruct"  "Mistral-7B-instruct""deepseek-llm-7b-chat")

PROMPTS=(1)
PROMPT_TEMPLATES=(0)
PARTY="Ciudadanos" #['Ciudadanos','Más País','Euzko Alderdi Jeltzalea - Partido Nacionalista Vasco',
                        # 'Partido Popular','Partido Socialista Obrero Español','Candidatura d\'Unitat Popular',
                        # 'Esquerra Republicana de Catalunya','Vox','Euskal Herria Bildu', 'Junts per Catalunya']
STANCE="de izquierdas" # ["de izquierdas", "de derechas", "central"]

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# -------------------- EXPERIMENT LOOP --------------------
for prompt in "${PROMPTS[@]}"; do
  for prompt_template in "${PROMPT_TEMPLATES[@]}"; do
    #for replace in "${REPLACES[@]}"; do
      echo "Running with model=$MODEL, prompt=$prompt, prompt_template=$prompt_template" #, replace=$replace"
      python3 -u running_experiments.py \
        --exp=ent \
        --model="$MODEL" \
        --prompt="$prompt" \
        --debug=1 \
        --datasize=200 \
        --party="$PARTY" \
        --stance="$STANCE"
  done
done
# -------------------- PARAMETERS --------------------