import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
#from transformers.models.cohere.tokenization_cohere import CohereTokenizer
from transformers import GemmaTokenizer
from transformers import set_seed


import torch
import os, sys
#import torch.nn.functional as F
import pandas as pd
import argparse
import time
import re
from utils import get_dataset, update_model_summary
from definitions import *
from model_paths import MODEL_PATHS
import json
import os

# === NEW: Import login function from huggingface_hub ===
from huggingface_hub import login

# Set HF cache to scratch
os.environ["HF_HOME"] = "/var/scratch/eei440/hf_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

def extract_probs(tokens, probs):
    '''
    Extracts the probabilities for the tokens 'for', 'against', and 'abstain' from the top_k tokens.
    Does NOT override based on output_text, only uses top-k tokens.
    '''
    for_synonyms = [
    'afavor', 'a favor', 'favor', 'fav', 'sí', 'si', 's', 'a fa', 'favour', 'favo', 'fa', '1',
    'aprobar', 'apruebo', 'aceptar', 'acepto', 'consentir', 'consiento', 'acceder', 'accedo',
    'convenir', 'convengo', 'concordar', 'concordamos', 'coincidir', 'coincido', 'asentir', 'asiento',
    'de acuerdo', 'apoyo', 'apoyar', 'afirmativo', 'positivo', 'for'
]
    against_synonyms = [
        'encontra', 'en contra', 'contra', 'contr', 'no', 'n', 'en co', 'contre', 'against', '-1',
        'desaprobar', 'desapruebo', 'rechazar', 'rechazo', 'oponerse', 'me opongo', 'disentir', 'disiento',
        'discrepar', 'discrepo', 'vetar', 'veto', 'oponer resistencia', 'resisto', 'opinión contraria',
        'en desacuerdo', 'negativo','against'
    ]
    abstain_synonyms = [
        'abstencion', 'abstención', 'abst', 'ab', 'stenc', 'stención', 'me abstengo', 'abstenerse', 'abste', 'absten', '0',
        'blank', '', ' ', 'omitir', 'omito', 'ignorar', 'ignoro', 'callar', 'me callo', 'silenciar', 'silencio',
        'prescindir', 'me reservo', 'no contesto', 'sin respuesta', 'me abstendré', 'abstain'
    ]

    favor_prob = 0
    contra_prob = 0
    otro_prob = 0

    for i, tok in enumerate(tokens[:20]):
        clean_tok = tok.strip().lstrip('▁').lower()
        clean_tok = re.sub(r'[^\w\s]', '', clean_tok)
        if any(s in clean_tok for s in for_synonyms):
            favor_prob += probs[i]
        elif any(s in clean_tok for s in against_synonyms):
            contra_prob += probs[i]
        elif any(s in clean_tok for s in abstain_synonyms):
            otro_prob += probs[i]
    total = favor_prob + contra_prob + otro_prob
    if total > 0:
        favor_prob /= total
        contra_prob /= total
        otro_prob /= total

    return favor_prob, contra_prob, otro_prob


def set_seeds(seed): #Balatro: same randomness for recreation purposes
    #random.seed(seed)   # Do not use random
    #np.random.seed(seed) # Do not use numpy
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



def run_experiment(exp_type, model_name, prompt_no=10, cont=0, DEBUG=False, small_data_size=20, prompt_template_no=0, lang="ES"):
    print("exp_type:", exp_type)
    print("model_name:", model_name)
    print("prompt_no:", prompt_no)
    print("prompt_template_no:", prompt_template_no)
    print("lang:", lang)    
    print("continue:", cont)
    print("DEBUG:", DEBUG)
    model_shortname = MODEL_SHORTNAMES.get(model_name, model_name.lower().replace("-", "_"))

    set_seeds(RANDOM_SEED) # Defined in definitions.py
    # This means: if you run the experiment again with the same data and settings, you’ll get the same answers

    # Read tokens from JSON-style file
    access_tokens = {} # Get Hugging Face “password” (called an access token) so I am allowed to use the models.
    with open("hf_accesstoken.txt", "r") as f:
        content = f.read().strip()
        if content.startswith("{"):
            access_tokens = json.loads(content)
        else:
            access_tokens = {"default": content}

    # Pick token based on model name: Pick the right password (token) for the model you want to use
    if "llama" in model_name.lower():
        access_token = access_tokens.get("llama", "")
    else:
        access_token = access_tokens.get("default", "")
        
        if model_name not in MODEL_PATHS:
            raise ValueError(f"Unknown model: {model_name}")

    model_path = MODEL_PATHS[model_name]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Optional: Clear GPU memory (if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Determine torch dtype
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name()
            torch_dtype = torch.bfloat16 if "H100" in gpu_name else torch.float16
        except:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load model + tokenizer with token
    start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            token=access_token
        )
        model = model.to(device)
        print(f"Model loaded in {time.time() - start:.2f} seconds and moved to {device}")
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        exit(1)


    # get the motions
    df = get_dataset(DEBUG, small_data_size, variant=0, exp=exp_type, lang=lang)
    result_df = df[['id', 'initiative']].copy()  # Ensure they both have the same rows!

    df['initiative'] = df['initiative'].astype(str).str.strip() # Ensure initiative is a string and stripped of whitespace
    result_df['initiative'] = result_df['initiative'].astype(str).str.strip() 

    
    # Builds a filename for saving your results, so you always know what experiment they belong to
    prompt_suffix = f"prompt={prompt_no}" + (f",TEMPLATE={prompt_template_no}" if prompt_template_no > 0 else "")
    debug_suffix = (f'_debug' if DEBUG else '')+(f'{small_data_size}' if DEBUG else '')
    results_file = f"results/{model_shortname}_results_{exp_type}_ES_{prompt_suffix}{debug_suffix}.csv"
    
    from_text = " de "  # Use Spanish preposition for "from"

    # Picks which parties or personas are being asked to vote.
    if exp_type == "ideology":
        parties = ['']  # No party-specific prefix
        parties_short = ['']
    # elif exp_type == "entity":
    #     parties = party_names
    #     parties_short = party_codes_used
    # elif exp_type == "persona":
    #     parties = direction_ES  # Defined in definitions.py
    #     parties_short = direction_codes  # Already language-agnostic
    
    # Tells the model: “Only generate up to 3 new tokens (words or pieces of words)” for each answer.
    max_new_tokens = 3

    # Prompts
    system_prompt_1 = ""
    system_prompt_2 = ""
    user_prompt_1 = ""
    user_prompt_2 = ""

    # Default Prompt used in experiment 1
    if prompt_no == 10:
        system_prompt_1 = (
            "Vota a favor, en contra o abstención de la siguiente propuesta. "
            "Responde únicamente con una palabra. SOLO responde con una de estas opciones exactas: 'a favor', 'en contra' o 'abstención'. No expliques tu respuesta. No añadas nada más."
        )
        system_prompt_2 = ""
        user_prompt_1 = "Iniciativa: "
        user_prompt_2 = "\nRespuesta:"
    
    # Label order variation
    if prompt_no == 11:
        system_prompt_1 = (
            "Vota a favor, en contra o abstención de la siguiente propuesta. "
            "Responde únicamente con una palabra. SOLO responde con una de estas opciones exactas: 'en contra', 'abstención' o 'a favor'. No expliques tu respuesta. No añadas nada más."
        )
        system_prompt_2 = ""
        user_prompt_1 = "Iniciativa: "
        user_prompt_2 = "\nRespuesta:"

    if prompt_no == 12:
        system_prompt_1 = (
            "Vota a favor, en contra o abstención de la siguiente propuesta. "
            "Responde únicamente con una palabra. SOLO responde con una de estas opciones exactas: 'abstención', 'a favor' o 'en contra'. No expliques tu respuesta. No añadas nada más."
        )
        system_prompt_2 = ""
        user_prompt_1 = "Iniciativa: "
        user_prompt_2 = "\nRespuesta:"
    
    # Instructional detail variation
    elif prompt_no == 3:  # With extra detail
        system_prompt_1 = "Considera los effectos de corto- y largo-plazo en la sociedad. Vota a favor, en contra o abstencion de la siguiente propuesta."
        system_prompt_2 = ". Responde únicamente con 'a favor', 'en contra' o 'abstención'."
        user_prompt_1 = "La iniciativa es: "
        user_prompt_2 = ""

    # Label variations
    elif prompt_no == 4:  # 'Agree/Disagree' labels
        system_prompt_1 = "Estas de acuerdo o no con la siguiente iniciativa votada en el Congreso de los diputados."
        system_prompt_2 = ". Unicamente responde con una palabra. Eligue entre: 'de acuerdo', 'en contra' o 'no contesto'."
        user_prompt_1 = "La iniciativa es: "
        user_prompt_2 = ""

    # Same but synonyms for 'agree' and 'disagree'
    # elif prompt_no == 5:  # 'Agree/Disagree' labels
    #     system_prompt_1 = "Estas de acuerdo o no con la siguiente iniciativa votada en el Congreso de los diputados."
    #     system_prompt_2 = ". Unicamente responde con 'de acuerdo', 'en contra' o 'no contesto'."
    #     user_prompt_1 = "La iniciativa es: "
    #     user_prompt_2 = ""

    # elif prompt_no == 5:  # 'Agree/Disagree' labels
    #     system_prompt_1 = "Estas de acuerdo o no con la siguiente iniciativa votada en el Congreso de los diputados."
    #     system_prompt_2 = ". Unicamente responde con 'de acuerdo', 'en contra' o 'no contesto'."
    #     user_prompt_1 = "La iniciativa es: "
    #     user_prompt_2 = ""

    # Initiative in English
    elif prompt_no == 5:
        system_prompt_1 = "Vote against, abstain or for the following motion in Spanish"
        system_prompt_2 = ". Only respond with one of these EXACT options: 'against', 'for' or 'abstain'."
        user_prompt_1 = "The motion in Spanish: "
        user_prompt_2 = ""

    else:
        print("unknown prompt")

    # temp_high = 1 # Model answers can be creative/random (temperature = 1, fully random).
    # temp_low = 0.0000001 # # Model answers are deterministic (temperature = 0, no randomness).
            
    if model_name == "":
        pass
    
    # # Category
    # system_prompt_cat = 'Clasifica esta moción. Elige solo una de las siguientes categorías:'
    # user_prompt_cat = 'Moción: '

    if DEBUG and len(parties)>3:
        parties = parties[:3]
    
    #initialize df to store results
    # Ensure all initiatives are strings and stripped
    df['initiative'] = df['initiative'].astype(str).str.strip()

    # Results CSV path
    temp_results_path = results_file.replace(".csv", "_TEMP.csv")

    # Initialize or continue result_df
    if cont < 0 or not os.path.exists(temp_results_path):
        result_df = df[['id', 'initiative']].copy()
        result_df['initiative'] = result_df['initiative'].astype(str).str.strip()

        for col in [f'{model_shortname}_vote', f'{model_shortname}_for_prob',
                    f'{model_shortname}_against_prob', f'{model_shortname}_abstain_prob']:
            result_df[col] = pd.NA
    else:
        result_df = pd.read_csv(temp_results_path, index_col=0)

        # Make sure 'initiative' exists and is aligned
        if 'initiative' not in result_df.columns:
            raise ValueError("Missing 'initiative' column in result_df!")

        result_df['initiative'] = result_df['initiative'].astype(str).str.strip()
        df['initiative'] = df['initiative'].astype(str).str.strip()
    
    print(result_df.index)
    start = time.time()

    suffix = ""  # No party suffix for ideology experiment
    
    for i, (x, id) in enumerate(zip(df['initiative'], df['id'])):
        if f'{model_shortname}{suffix}_vote' in result_df.columns:
            mask = (result_df['initiative'] == x.strip()) & (result_df['id'] == id)
            # Only skip if row is found and vote is NOT NaN (i.e., it's already done)
            if mask.any():
                vote_val = result_df.loc[mask, f'{model_shortname}{suffix}_vote'].values[0]
                if pd.notna(vote_val):
                    print("[DEBUG] Skipping already voted item")
                    print("initiative=", x)
                    print("id=", id)
                    print("vote=", vote_val)
                    continue  # <-- This skips to the next item if already answered

        # Now, for items that need answers, generate the prompt and call the LLM
        ########################################
        """ CREATING PROMPTS """ 
        ########################################

        # party_txt = f'{from_text}{party}' # This is for when doing an entity or persona experiment, where the party is included in the prompt.
        
        # prompt formats from model cards
        if model_shortname == 'llama3_8b' or model_shortname == 'llama3_70b':
            input_prompt = f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            {system_prompt_1}{system_prompt_2}
            Responde únicamente con una de estas opciones exactas: 'a favor', 'en contra' o 'abstención'.<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {user_prompt_1}{x}{user_prompt_2}
            ¿Cuál es tu voto? Responde exactamente con una de estas opciones: 'a favor', 'en contra' o 'abstención'.<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """
        
        # elif model_shortname == "mistral_7b":
        #     input_prompt = (
        #         #f"{system_prompt_1}{system_prompt_2}\n"
        #         f"{user_prompt_1}{x}{user_prompt_2}"
        #         #"No expliques tu respuesta. No añadas nada más.\n"
        #         "Do not explain your answer. Do not add anything else.\n"
        #     )
        # elif model_shortname == "mistral_7b":
        #             messages = [
        #                     {"role": "system", "content": f"{system_prompt_1}{system_prompt_2}"},
        #                     {"role": "user", "content": f"{user_prompt_1}{x}{user_prompt_2}"},
        #             ]
        #             input_prompt = tokenizer.apply_chat_template(
        #                     messages,
        #                     tokenize=False,
        #                     add_generation_prompt=True
        #             )
        #             print(input_prompt)
        # elif model_shortname == "mistral_7b":
        #     input_prompt = (
        #         f"Iniciativa: {x}"
        #         "Vota: (elige solo: a favor, en contra, abstención)"
        #     )

        elif model_shortname == "mistral_7b":
            messages = [
                {"role": "system", "content": (
                    f"{system_prompt_1}{system_prompt_2}\n"
                    "SOLO responde con una de estas opciones exactas: 'a favor', 'en contra' o 'abstención'. "
                    "No expliques tu respuesta. No añadas nada más."
                )},
                {"role": "user", "content": f"{user_prompt_1}{x}{user_prompt_2}\nRespuesta:"},
            ]
            input_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        elif model_shortname == "deepseek_7b":
            messages = [
                {"role": "system", "content": (
                    f"{system_prompt_1}{system_prompt_2}\n"
                    "SOLO responde con una de estas opciones exactas: 'a favor', 'en contra' o 'abstención'. "
                    "No expliques tu respuesta. No añadas nada más."
                )},
                {"role": "user", "content": f"{user_prompt_1}{x}{user_prompt_2}\nRespuesta:"},
            ]
            input_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ) 
        
        elif model_shortname in ["falcon3_7b", "gemma2_9b"]:
            # Try plain prompt, no chat template
            input_prompt = [
                "Elige UNA de estas opciones y escribe SOLO la opción exacta, nada más:\n"
                "a favor\n"
                "en contra\n"
                "abstención\n\n"
                f"Iniciativa: {x}\n"
                "Respuesta:"
        ]

        else:
            print(f"[WARNING] No prompt template defined for model_shortname '{model_shortname}'")
            continue

        if True or DEBUG:
            print(input_prompt)
        
        # Prepare input 
        inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        input_token_len = input_ids.shape[-1]

        if model_name in ['Llama3-instruct', 'Llama3-70B-instruct']: # Llama 3 models require a custom pad token ID
            pad_token_id = 128001
        else:
            pad_token_id = tokenizer.eos_token_id
        
        #PROMPT Model
        #prompt the model with temperature near 0 to produce deterministic responses
        if model_name != "Mistral-instruct":
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False, # deterministic
                temperature=0,   # deterministic
                output_scores=True,
                return_dict_in_generate=True,
            )
        else:
            outputs_temp0 = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0, # high temperature for more creative output
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # NB CHANGE
        if model_name != "Mistral-instruct":
            generated_text = tokenizer.decode(outputs.sequences[0][input_token_len:], skip_special_tokens=True)
        else:
            generated_text = tokenizer.decode(outputs_temp0.sequences[0][input_token_len:], skip_special_tokens=True)

        # After decoding model output
        print("Raw model output:", generated_text)

        # Print the actual LLM output, including invisible characters
        print("\n" + "="*40)
        print(f"LLM OUTPUT for ID {id} (before normalization):")
        print(repr(generated_text))  # Shows whitespace and special chars
        print("="*40 + "\n")

        # Special normalization for Mistral list-style outputs
        if model_shortname == "mistral_7b":
            found = False
            for line in generated_text.splitlines():
                l = line.lower().strip()
                if not l:
                    continue  # skip empty lines
                if "a favor" in l or "afavor" in l:
                    generated_text = "a favor"
                    found = True
                    break
                elif "en contra" in l or "encontra" in l or "contra" in l:
                    generated_text = "en contra"
                    found = True
                    break
                elif "abstención" in l or "abstencion" in l:
                    generated_text = "abstención"
                    found = True
                    break
            if not found:
                generated_text = "blank"

        generated_text = generated_text.lower().strip()
        generated_text = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑüÜ ]', '', generated_text)
        generated_text = generated_text if generated_text != "" else "blank"

        norm = generated_text.lower().strip()
        norm = re.sub(r'[^a-záéíóúñü ]', '', norm)  # Remove punctuation, keep Spanish chars
        norm = re.sub(r'\s+', ' ', norm)  # Remove multiple spaces
        
        # Use the same synonym lists as in extract_probs
        for_synonyms = [
        'afavor', 'a favor', 'favor', 'fav', 'sí', 'si', 's', 'a fa', 'favour', 'favo', 'fa', '1',
        'aprobar', 'apruebo', 'aceptar', 'acepto', 'consentir', 'consiento', 'acceder', 'accedo',
        'convenir', 'convengo', 'concordar', 'concordamos', 'coincidir', 'coincido', 'asentir', 'asiento',
        'de acuerdo', 'apoyo', 'apoyar', 'afirmativo', 'positivo', 'for'
        ]
        against_synonyms = [
            'encontra', 'en contra', 'contra', 'contr', 'no', 'en', 'en co', 'contre', 'against', '-1',
            'desaprobar', 'desapruebo', 'rechazar', 'rechazo', 'oponerse', 'me opongo', 'disentir', 'disiento',
            'discrepar', 'discrepo', 'vetar', 'veto', 'oponer resistencia', 'resisto', 'opinión contraria',
            'en desacuerdo', 'negativo','against'
        ]
        abstain_synonyms = [
            'abstencion', 'abstención', 'abst', 'ab', 'stenc', 'stención', 'me abstengo', 'abstenerse', 'abste', 'absten', '0',
            'blank', '', ' ', 'omitir', 'omito', 'ignorar', 'ignoro', 'callar', 'me callo', 'silenciar', 'silencio',
            'prescindir', 'me reservo', 'no contesto', 'sin respuesta', 'me abstendré', 'abstain'
        ]

                # Main forced-choice labels
        main_labels = {
            "a favor": 1,
            "en contra": -1,
            "abstención": 0,
            "abstencion": 0,  # Accept both spellings
        }

        # Synonym lists (as before)
        for_synonyms = [...]
        against_synonyms = [...]
        abstain_synonyms = [...]

        # 1. Exact match
        if norm in main_labels:
            vote_text = norm
            vote_value = main_labels[norm]
        # 2. If "favor" is present, always "a favor"
        elif "favor" in norm:
            vote_text = "a favor"
            vote_value = 1
        # 3. If output is "en", "en co", "encont", "encontra", "contra" (and does NOT contain "favor")
        elif norm in ["en", "en co", "encont", "encontra", "contra"]:
            vote_text = "en contra"
            vote_value = -1
        # 4. Synonym match (word boundaries)
        elif any(re.search(rf"\b{s}\b", norm) for s in for_synonyms):
            vote_text = "a favor"
            vote_value = 1
        elif any(re.search(rf"\b{s}\b", norm) for s in against_synonyms):
            vote_text = "en contra"
            vote_value = -1
        elif any(re.search(rf"\b{s}\b", norm) for s in abstain_synonyms):
            vote_text = "abstención"
            vote_value = 0
        elif norm == "" or norm == "blank":
            vote_text = "abstención"
            vote_value = 0
        else:
            vote_text = "otro"
            vote_value = 0

        print(f"[DEBUG] Normalized text: '{generated_text}'")
        print(f"[DEBUG] Interpreted vote: '{vote_text}' ({vote_value})")

        # Retrieve logit scores
        if model_name != "Mistral-instruct":
            logits = outputs.scores
        else:
            logits = outputs_temp0.scores   

        # Calculate the top_k tokens and probabilities for each generated token
        top_k = 20
        all_top_tokens = []
        all_top_probs = []

        for step_logits in logits:
            probs = torch.softmax(step_logits[0], dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)
            top_probs = top_probs.tolist()
            top_indices = top_indices.tolist()
            top_tokens = tokenizer.convert_ids_to_tokens(top_indices)
            all_top_tokens.extend(top_tokens)
            all_top_probs.extend(top_probs)

        # Print top tokens and probabilities for debugging
        print("[DEBUG] Top tokens:", all_top_tokens)
        print("[DEBUG] Top probs:", all_top_probs)

        # Extract the probabilities for the tokens 'for' and 'against' from the top_k tokens
        favor_prob, contra_prob, otro_prob = extract_probs(all_top_tokens, all_top_probs)

        # Use both 'initiative' and 'id' for DataFrame mask
        mask = (result_df['initiative'] == x.strip()) & (result_df['id'] == id)

        print(f"\n[DEBUG] Processing ID: {id}")
        print(f"[DEBUG] Matches in result_df: {mask.sum()}")
        print(f"[DEBUG] Generated text (raw): '{generated_text}'")
        print(f"[DEBUG] Interpreted vote value: {vote_value}")
        print(f"[DEBUG] Probabilities - For: {favor_prob}, Against: {contra_prob}, Abstain: {otro_prob}")

        if mask.any():
            print(f"Generated: {generated_text}, For: {favor_prob}, Against: {contra_prob}, Abstain: {otro_prob}")
            print(f"Updating ID: {id}, Matches found: {mask.sum()}")

            result_df.loc[mask, 
                [f'{model_shortname}_vote', 
                f'{model_shortname}_for_prob', 
                f'{model_shortname}_against_prob', 
                f'{model_shortname}_abstain_prob']
            ] = [vote_value, favor_prob, contra_prob, otro_prob]
            print(result_df.loc[mask, [f'{model_shortname}_vote']])
            print(f"[DEBUG] Wrote vote={vote_value} for model '{model_shortname}' at ID {id}")

            print(result_df[[f"{model_shortname}_vote"]].value_counts(dropna=False))
        else:
            print(f"[WARNING] No matching row for ID {id} in result_df!")

        if i % 1 == 0:
            temp_file = results_file.replace(".csv", "_TEMP.csv")
    
            # 1. Save temporary file
            result_df.to_csv(temp_file, encoding='utf-8-sig', index=True)
            print(f"Saved {temp_file}")

            # 2. Git commit and push
            commit_message = f"Autosave {model_name} after {i} motions"
            os.system(f"git add {temp_file}")
            os.system(f"git commit -m '{commit_message}'")
            os.system("git push origin main")

            # 3. Remove local file to save space
            os.remove(temp_file)
            print(f"Deleted local {temp_file} after pushing to GitHub")

    
    #save the df
    #print(result_df)
    result_df.to_csv(results_file.replace(".csv", "_TEMP.csv"), encoding='utf-8-sig', index=True)
    result_df.to_csv(results_file, encoding='utf-8-sig', index=True)
    print(f"[DEBUG] Final saved file: {results_file} with {len(result_df)} rows")
    

    if DEBUG and small_data_size == 200 and exp_type == "ideology":
        update_model_summary(model_shortname, prompt_no, prompt_template_no, result_df, exp_type)
     
    if exp_type == "ideology": 
        colname = f"{model_shortname}_vote"
        if colname in result_df.columns:
            print(result_df[colname].value_counts() / len(result_df))
        else:
            print(f"⚠️ Warning: Column '{colname}' not found in model_name.")


    elapsed_time = time.time() - start
    print(f"Experiment time {int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m {elapsed_time % 60:.2f}s")  

    
MODEL_SHORTNAMES = {
        "Falcon3-7B-instruct": "falcon3_7b",
        "Llama-3-8B-instruct": "llama3_8b",
        "Llama-3-70B-instruct": "llama3_70b",
        "Mistral-7B-instruct": "mistral_7b",
        "Gemma-2-9B-instruct": "gemma2_9b",
        "deepseek-llm-7b-chat": "deepseek_7b"
    }    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="ide", help="type of experiment")
    parser.add_argument("--model", type=str, default="Falcon3-7B-instruct", help="model to run")
    parser.add_argument("--prompt", type=int, default=1, help="prompt no")
    parser.add_argument("--template", type=int, default=0, help="prompt template, for models with more than one.")
    #parser.add_argument("--replace", type=int, default=0, help="remove start")
    parser.add_argument("--cont", type=int, default=-1, help="continue exp")
    parser.add_argument("--debug", type=int, default=0, help="Debug. 0: No or 1: Yes")
    parser.add_argument("--datasize", type=int, default=20, help="Size of debug dataset (no effect if not debug)")

    args = parser.parse_args()
    exp_type = args.exp
    model_name = args.model

    model_shortname = MODEL_SHORTNAMES.get(model_name, model_name.lower().replace("-", "_"))

    # Log in using your token (already saved in file)
    with open("hf_accesstoken.txt") as f:
        content = f.read().strip()
        if content.startswith("{"):
            access_tokens = json.loads(content)
        else:
            access_tokens = {"default": content}

    # Pick token based on model name
    if "llama" in model_name.lower():
        access_token = access_tokens.get("llama", "")
    else:
        access_token = access_tokens.get("default", "")
    # === END: Import login function from huggingface_hub ===

    prompt_no = args.prompt
    prompt_template_no = args.template
    DEBUG = bool(args.debug)
    cont = args.cont
    small_data_size = args.datasize

    if exp_type == "ide": exp_type = "ideology"
    elif exp_type == "ent": exp_type = "entity"
    elif exp_type == "per": exp_type = "persona"

    lang = "ES"
    
    run_experiment(exp_type, model_name, prompt_no, cont, DEBUG, prompt_template_no=prompt_template_no, small_data_size=small_data_size, lang=lang)