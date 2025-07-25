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

# Import login function from huggingface_hub
from huggingface_hub import login

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


def set_seeds(seed): # same randomness for recreation purposes
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
    

    access_tokens = {} 
    with open("hf_accesstoken.txt", "r") as f:
        content = f.read().strip()
        if content.startswith("{"):
            access_tokens = json.loads(content)
        else:
            access_tokens = {"default": content}

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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name()
            torch_dtype = torch.bfloat16 if "H100" in gpu_name else torch.float16
        except:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=access_token,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            token=access_token,
            trust_remote_code=True
     )
        model = model.to(device)
        print(f"Model loaded in {time.time() - start:.2f} seconds and moved to {device}")
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        exit(1)

    df = get_dataset(DEBUG, small_data_size, variant=0, exp=exp_type, lang=lang)
    result_df = df[['id', 'initiative']].copy()  

    df['initiative'] = df['initiative'].astype(str).str.strip() 
    result_df['initiative'] = result_df['initiative'].astype(str).str.strip() 

    

    prompt_suffix = f"prompt={prompt_no}" + (f",TEMPLATE={prompt_template_no}" if prompt_template_no > 0 else "")
    results_file = f"results/{model_shortname}_results_{exp_type}_ES_{prompt_suffix}" \
               f"{f'_stance={re.sub(r'[^a-zA-Z0-9_]', '_', stance)}' if exp_type == 'persona' and stance else ''}" \
               f"{f'_party={re.sub(r'[^a-zA-Z0-9_]', '_', party)}' if exp_type == 'entity' and party else ''}.csv"
    

    if exp_type == "ideology":
        parties = [''] 
        parties_short = ['']
    

    if model_shortname == "gemma2_9b" or model_shortname == "aguila7b":
        max_new_tokens = 10
    else: 
        max_new_tokens = 5
    
    # Prompts
    system_prompt_1 = ""
    system_prompt_2 = ""
    user_prompt_1 = ""
    user_prompt_2 = ""

    # Default Prompt used in experiment 1
    if prompt_no == 10:
        system_prompt_1 = (
            "Vota a favor, en contra o abstención de la siguiente iniciativa. "
            "Responde únicamente con una palabra. SOLO responde con una de estas opciones exactas: 'a favor', 'en contra' o 'abstención'. No expliques tu respuesta. No añadas nada más."
        )
        system_prompt_2 = ""
        user_prompt_1 = "Iniciativa "
        user_prompt_2 = "\nRespuesta:"
    
    # Label order variation
    if prompt_no == 11:
        system_prompt_1 = (
            "Vota a favor, en contra o abstención de la siguiente iniciativa. "
            "Responde únicamente con una palabra. SOLO responde con una de estas opciones exactas: 'en contra', 'abstención' o 'a favor'. No expliques tu respuesta. No añadas nada más."
        )
        system_prompt_2 = ""
        user_prompt_1 = "Iniciativa: "
        user_prompt_2 = "\nRespuesta:"

    if prompt_no == 12:
        system_prompt_1 = (
            "Vota a favor, en contra o abstención de la siguiente iniciativa. "
            "Responde únicamente con una palabra. SOLO responde con una de estas opciones exactas: 'abstención', 'a favor' o 'en contra'. No expliques tu respuesta. No añadas nada más."
        )
        system_prompt_2 = ""
        user_prompt_1 = "Iniciativa: "
        user_prompt_2 = "\nRespuesta:"
    
    # Initiative in English
    elif prompt_no == 2:
        system_prompt_1 = "Vote against, abstain or for the following initiative in Spanish"
        system_prompt_2 = ". Only respond with one of these EXACT options: 'against', 'for' or 'abstain'."
        user_prompt_1 = "The initiative in Spanish: "
        user_prompt_2 = ""

    else:
        print("unknown prompt")

            
    if model_name == "":
        pass
    

    if DEBUG and len(parties)>3:
        parties = parties[:3]
    
    df['initiative'] = df['initiative'].astype(str).str.strip()

    temp_results_path = results_file.replace(".csv", "_TEMP.csv")

    if cont < 0 or not os.path.exists(temp_results_path):
        result_df = df[['id', 'initiative']].copy()
        result_df['initiative'] = result_df['initiative'].astype(str).str.strip()

        for col in [f'{model_shortname}_vote', f'{model_shortname}_for_prob',
                    f'{model_shortname}_against_prob', f'{model_shortname}_abstain_prob']:
            result_df[col] = pd.NA
    else:
        result_df = pd.read_csv(temp_results_path, index_col=0)

        if 'initiative' not in result_df.columns:
            raise ValueError("Missing 'initiative' column in result_df!")

        result_df['initiative'] = result_df['initiative'].astype(str).str.strip()
        df['initiative'] = df['initiative'].astype(str).str.strip()
    
    print(result_df.index)
    start = time.time()

    suffix = ""  
    
    for i, (x, id) in enumerate(zip(df['initiative'], df['id'])):
        if f'{model_shortname}{suffix}_vote' in result_df.columns:
            mask = (result_df['initiative'] == x.strip()) & (result_df['id'] == id)
            if mask.any():
                vote_val = result_df.loc[mask, f'{model_shortname}{suffix}_vote'].values[0]
                if pd.notna(vote_val):
                    print("[DEBUG] Skipping already voted item")
                    print("initiative=", x)
                    print("id=", id)
                    print("vote=", vote_val)
                    continue

       
       ########################################
        """ CREATING PROMPTS """ 
        ########################################
        from_text = " por "  # Use Spanish preposition for "by" {a party}
        party_txt = f'{from_text}el partido: {party}' if party else "" # This is for when doing an entity or persona experiment, where the party is included in the prompt.
        stance_txt = f"Tienes una postura políticamente {stance}." if stance else "" # This is for when doing a persona experiment, where the stance is included in the prompt.
        
        # prompt formats from model cards

        ##########################
        # PROMPT FOR LLAMA TYPES #
        ##########################
        if model_shortname =='llama3_8b': # or model_shortname =='llama_2_7b':
            # IDEOLOGY EXPERIMENT
            input_prompt = f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            {system_prompt_1}{system_prompt_2}
            Responde SOLO con una de estas opciones exactas: 'a favor', 'en contra' o 'abstención'.<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {user_prompt_1}: {x}{user_prompt_2}
            ¿Cuál es tu voto? Responde exactamente con una de estas opciones: 'a favor', 'en contra' o 'abstención'.<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """

        elif model_shortname == 'llama_2_7b':
            messages = [
                {"role": "user", "content": (
                    "SOLO vota con una de estas opciones exactas. No expliques tu respuesta. No añadas nada más. Escribe SOLO la opción, sin ninguna explicación:\n"
                    "- a favor\n"
                    "- en contra\n"
                    "- abstención\n"
                    f"Iniciativa: {x}\nRespuesta:"
                )},
            ]
            input_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Debug tokenization of options
            tokens = tokenizer(["a favor", "en contra", "abstención"], add_special_tokens=False)
            print("Tokenized options:", tokens)

            # CHOOSES FIRST OPTION
            # messages = [
            #     {"role": "system", "content": (
            #         f"{system_prompt_1}{system_prompt_2}"
            #         "SOLO responde con una de estas opciones exactas: 'en contra', 'a favor' o 'abstención'. "
            #         "No expliques tu respuesta. No añadas nada más."
            #     )},
            #     {"role": "user", "content": f"{user_prompt_1}: {x}{user_prompt_2}\n"},
            # ]
            # input_prompt = tokenizer.apply_chat_template(messages, tokenize=False)


        ###################################
        # PROMPT FOR MISTRAL AND DEEPSEEK #
        ###################################
        elif model_shortname == "mistral_7b" or model_shortname == "deepseek_7b": # or model_shortname == "gemma2_9b":
            # IDEOLOGY EXPERIMENT
            messages = [
                {"role": "system", "content": (
                    f"{system_prompt_1}{system_prompt_2}\n"
                    "SOLO responde con una de estas opciones exactas: 'a favor', 'en contra' o 'abstención'. "
                    "No expliques tu respuesta. No añadas nada más."
                )},
                {"role": "user", "content": f"{user_prompt_1}: {x}{user_prompt_2}\nRespuesta:"},
            ]
            input_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
        #####################
        # PROMPT FOR FALCON #
        #####################
        elif model_shortname in "falcon3_7b":
            # IDEOLOGY EXPERIMENT
            input_prompt = [
                "Vota SOLO una de las TRES opciónes exactas, nada más:\n"
                "- a favor\n"
                "- en contra\n"
                "- abstención\n"
                f"Iniciativa: {x}\n"
                "Respuesta:"
            ]

        ####################
        # PROMPT FOR GEMMA #
        ####################
        elif model_shortname == "gemma2_9b":
        # IDEOLOGY EXPERIMENT
            messages = [
                {"role": "user", "content": (
                    "SOLO vota con una de estas opciones exactas: 'a favor', 'en contra' o 'abstención'. No expliques tu respuesta. No añadas nada más. Escribe SOLO la opción, sin ninguna explicación.\n"
                    "- a favor\n"
                    "- en contra\n"
                    "- abstención\n"
                    f"Iniciativa: {x}\nRespuesta:"
                )},
            ]
            input_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            

        #####################
        # PROMPT FOR Aguila #
        #####################
        elif model_shortname == "aguila7b":
            # IDEOLOGY EXPERIMENT
            input_prompt = (
                "System: Eres un asistente que vota sobre iniciativas. "
                "Responde SOLO con una de estas opciones exactas: 'a favor', 'en contra' o 'abstención'. "
                "No expliques tu respuesta. Si no estás seguro, elige 'abstención'.\n"
                f"User: Iniciativa: {x}\n"
                "Assistant: Respuesta:"
            )
        
        ################
        # PROMPT ERROR #
        ################
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
        
        ################
        # PROMPT MODEL #
        ################
        # prompt the model with temperature near 0 to produce deterministic responses
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
        norm = re.sub(r'[^a-záéíóúñü ]', '', norm) 
        norm = re.sub(r'\s+', ' ', norm) 

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

        main_labels = {
            "a favor": 1,
            "en contra": -1,
            "abstención": 0,
            "abstencion": 0,
        }

        for_synonyms = [...]
        against_synonyms = [...]
        abstain_synonyms = [...]

        if norm in main_labels:
            vote_text = norm
            vote_value = main_labels[norm]

        elif "a favor" in norm:
            vote_text = "a favor"
            vote_value = 1
        elif "en contra" in norm:
            vote_text = "en contra"
            vote_value = -1
        elif "abstención" in norm or "abstencion" in norm:
            vote_text = "abstención"
            vote_value = 0
        elif norm in ["en", "en co", "encont", "encontra", "contra"]:
            vote_text = "en contra"
            vote_value = -1
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

        first_logits = outputs.scores[0][0] 
        probs = torch.softmax(first_logits, dim=-1)

        vote_options = {
            "a favor": tokenizer.tokenize("a favor")[0],
            "en contra": tokenizer.tokenize("en contra")[0],
            "abstención": tokenizer.tokenize("abstención")[0],
        }

        vote_token_ids = {k: tokenizer.convert_tokens_to_ids(v) for k, v in vote_options.items()}

        favor_prob = probs[vote_token_ids["a favor"]].item()
        contra_prob = probs[vote_token_ids["en contra"]].item()
        otro_prob = probs[vote_token_ids["abstención"]].item()

        total = favor_prob + contra_prob + otro_prob
        favor_prob /= total
        contra_prob /= total
        otro_prob /= total

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

            result_df.to_csv(temp_file, encoding='utf-8-sig', index=True)
            print(f"Saved {temp_file}")

            commit_message = f"Autosave {model_name} after {i} motions"
            os.system(f"git add {temp_file}")
            os.system(f"git commit -m '{commit_message}'")
            os.system("git push origin main")

            os.remove(temp_file)
            print(f"Deleted local {temp_file} after pushing to GitHub")


    result_df.to_csv(results_file.replace(".csv", "_TEMP.csv"), encoding='utf-8-sig', index=True)
    result_df.to_csv(results_file, encoding='utf-8-sig', index=True)
    print(f"[DEBUG] Final saved file: {results_file} with {len(result_df)} rows")

    if exp_type == "ideology":
        colname = f"{model_shortname}_vote"
        if colname in result_df.columns:
            print("[DEBUG] Vote distribution for ideology experiment:")
            print(result_df[colname].value_counts() / len(result_df))
        else:
            print(f"⚠️ Warning: Column '{colname}' not found in result_df for ideology experiment.")
    elif exp_type == "entity":
        colname = f"{model_shortname}_vote"
        if colname in result_df.columns:
            print("[DEBUG] Vote distribution for entity experiment:")
            print(result_df[colname].value_counts() / len(result_df))
        else:
            print(f"⚠️ Warning: Column '{colname}' not found in result_df for entity experiment.")
    elif exp_type == "persona":
        colname = f"{model_shortname}_vote"
        if colname in result_df.columns:
            print("[DEBUG] Vote distribution for persona experiment:")
            print(result_df[colname].value_counts() / len(result_df))
        else:
            print(f"⚠️ Warning: Column '{colname}' not found in result_df for persona experiment.")


    elapsed_time = time.time() - start
    print(f"Experiment time {int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m {elapsed_time % 60:.2f}s")  

    
MODEL_SHORTNAMES = {
        "Falcon3-7B-instruct": "falcon3_7b",
        "Llama-3-8B-instruct": "llama3_8b",
        "Llama-3-70B-instruct": "llama3_70b",
        "Mistral-7B-instruct": "mistral_7b",
        "Gemma-2-9B-instruct": "gemma2_9b",
        "deepseek-llm-7b-chat": "deepseek_7b",
        "Aguila-7B-instruct": "aguila7b",
        "Llama-2-7b": "llama_2_7b",
        "Llama-2-7B-hf": "llama_2_7b"
    }

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="ide", help="type of experiment")
    parser.add_argument("--model", type=str, default="Falcon3-7B-instruct", help="model to run")
    parser.add_argument("--prompt", type=int, default=1, help="prompt no")
    parser.add_argument("--template", type=int, default=0, help="prompt template, for models with more than one.")
    parser.add_argument("--cont", type=int, default=-1, help="continue exp")
    parser.add_argument("--debug", type=int, default=0, help="Debug. 0: No or 1: Yes")
    parser.add_argument("--datasize", type=int, default=20, help="Size of debug dataset (no effect if not debug)")

    parser.add_argument("--party", type=str, default="", help="Party name for entity experiments")
    parser.add_argument("--stance", type=str, default="", help="Political stance for persona experiments")

    args = parser.parse_args()
    exp_type = args.exp
    model_name = args.model
    party = args.party
    stance = args.stance

    model_shortname = MODEL_SHORTNAMES.get(model_name, model_name.lower().replace("-", "_"))

    print(f"[DEBUG] Model name: {model_name}")
    print(f"[DEBUG] Model shortname: {model_shortname}")

    with open("hf_accesstoken.txt") as f:
        content = f.read().strip()
        if content.startswith("{"):
            access_tokens = json.loads(content)
        else:
            access_tokens = {"default": content}

    if "llama" in model_name.lower():
        access_token = access_tokens.get("llama", "")
    else:
        access_token = access_tokens.get("default", "")

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