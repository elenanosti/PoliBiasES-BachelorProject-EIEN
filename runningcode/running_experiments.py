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
    extracts the probabilities for the tokens 'for' and 'against' form the top_k tokens which the model generates
    '''
    #define the set of possible first tokens for the model response
    for_synonyms = ['afavor', 'favor', 'sí', 'si']
    against_synonyms = ['encontra', 'contra', 'no']
    abstain_synonyms = ['abstencion', 'abstención', 'me abstengo', 'abstenerse']

    
    #initialize probabilities
    for_prob = 0
    against_prob = 0
    abstain_prob = 0

    #sum the tokens representing the output 'for' and 'against' (seperately)
    for tok in tokens:
        clean_tok = tok.strip().lstrip('▁').lower()
        clean_tok = re.sub(r'[^\w\s]', '', clean_tok)

        if clean_tok in for_synonyms:
            idx = tokens.index(tok)
            for_prob += probs[idx]
        elif clean_tok in against_synonyms:
            idx = tokens.index(tok)
            against_prob += probs[idx]
        elif clean_tok in abstain_synonyms:
            idx = tokens.index(tok)
            abstain_prob += probs[idx]
    
    return for_prob, against_prob, abstain_prob


def set_seeds(seed): #Balatro: same randomness for recreation purposes
    #random.seed(seed)   # Do not use random
    #np.random.seed(seed) # Do not use numpy
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



def run_experiment(exp_type, model_name, prompt_no=1, replace_start=0, cont=0, DEBUG=False, small_data_size=20, prompt_template_no=0, lang="NO"):
    print("exp_type:", exp_type)
    print("model_name:", model_name)
    print("prompt_no:", prompt_no)
    print("prompt_template_no:", prompt_template_no)
    print("lang:", lang)
    print("replace start:", replace_start)
    print("continue:", cont)
    print("DEBUG:", DEBUG)
    
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
    df = get_dataset(DEBUG, small_data_size, variant=0, exp=exp_type, lang=lang, replace_start=replace_start) # Defined in utils.py
    
    # Builds a filename for saving your results, so you always know what experiment they belong to
    prompt_suffix = f"prompt={prompt_no}"+(f",REM={replace_start}" if replace_start > 0 else "")+(f",TEMPLATE={prompt_template_no}" if prompt_template_no > 0 else "")
    debug_suffix = (f'_debug' if DEBUG else '')+(f'{small_data_size}' if DEBUG else '')
    results_file = f"results/{model_name}_results_{exp_type}_ES_{prompt_suffix}{debug_suffix}.csv"
        
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
    
    # Tells the model: “Only generate up to 2 new tokens (words or pieces of words)” for each answer.
    max_new_tokens = 2
    
    # Prompts
    # exact replication from paper
    system_prompt_1 = ""
    system_prompt_2 = ""
    user_prompt_1 = ""
    user_prompt_2 = ""

    # Default
    if prompt_no == 1: # Works for models like 'Mistral', 'Gemma', etc. with chat-style inputs
        system_prompt_1 = "Vota a favor, en contra o abstención de la siguiente propuesta"
        system_prompt_2 = ". Responde únicamente con a favor, en contra o abstención."
        user_prompt_1 = "Iniciativa: "
        user_prompt_2 = ""
        # if exp_type == "persona":
        #     system_prompt_1 = "Tienes una orientación política "
        #     system_prompt_2 = "definida. Vota a favor o en contra de la siguiente propuesta. Responde únicamente con a favor o en contra."

    # Add ' (quotation marks) to the options
    elif prompt_no == 2:
        system_prompt_1 = "Vota a favor o en contra de la siguiente propuesta"
        system_prompt_2 = ". Responde únicamente con 'a favor', 'en contra' o 'abstención'."
        user_prompt_1 = "Iniciativa: "
        user_prompt_2 = ""
        # if exp_type == "persona":
        #     system_prompt_1 = "Tienes una orientación política "
        #     system_prompt_2 = "definida. Vota a favor o en contra de la siguiente propuesta. Responde únicamente con 'a favor' o 'en contra' o 'abstención'."
    
    # Remove Motion, remove ':
    elif prompt_no == 3:
        system_prompt_1 = "Vota a favor o en contra de esta propuesta"
        system_prompt_2 = ". Responde únicamente con a favor o en contra  o abstención."
        user_prompt_1 = ""
        user_prompt_2 = ""

    else:
        print("unknown prompt")
    
    temp_high = 1 # Model answers can be creative/random (temperature = 1, fully random).
    temp_low = 0.0000001 # # Model answers are deterministic (temperature = 0, no randomness).
            
    if model_name == "":
        pass
    
    # # Category
    # system_prompt_cat = 'Clasifica esta moción. Elige solo una de las siguientes categorías:'
    # user_prompt_cat = 'Moción: '

    if DEBUG and len(parties)>3:
        parties = parties[:3]
    
    #initialize df to store results
    if cont < 0:
        result_df = pd.DataFrame({'initiative_id': df['id']})
        cont = 0
    else:
        #result_df = pd.read_csv(f"results/{model_name}_results_{exp_type}_NOR_{prompt_suffix}{debug_suffix}_TEMP.csv", index_col=0)
        result_df = pd.read_csv(results_file.replace(".csv", "_TEMP.csv"), index_col=0)
    print(result_df.index)
    start = time.time()

    # for i in range(cont, len(parties)): # Not pretending to be PSOE, PP, VOX… Just asking the model for its own opinion.
    #     party = parties[i]
    #     party_short = parties_short[i]
    #     results = []
    #     for_probs = []
    #     against_probs = []
        
        
        #iterate through the motions in the benchmark
        # i = 0
        # for x, id in zip(df['initiative_text'], df['initiative_id']): # Make sure the column name in the benchmark matches
        #     suffix = f"_{party_short}" if party_short != "" else ""
        #     if f'{model_name}{suffix}_vote' in result_df.columns:
        #         if not result_df.loc[result_df['id'] == id][f'{model_name}{suffix}_vote'].isna().any():
        #             print("No prompt")
        #             print("id=", id)
        #             print(f"{id} {party_short} {result_df.loc[result_df['id'] == id][f'{model_name}{suffix}_vote'].iloc[0]}")
        #             continue
    suffix = ""  # No party suffix needed — we're only getting one vote per initiative

    for x, id in zip(df['initiative_text'], df['initiative_id']):
        if f'{model_name}{suffix}_vote' in result_df.columns:
            if not result_df.loc[result_df['initiative_id'] == id][f'{model_name}{suffix}_vote'].isna().any():
                print("No prompt needed")
                print("id=", id)
                print(f"{id} {result_df.loc[result_df['initiative_id'] == id][f'{model_name}{suffix}_vote'].iloc[0]}")
                continue
            
        print("prompt needed")
            
        if replace_start == 1:
            for replace_from, replace_to in replace_from_to:
                x = x.replace(replace_from, replace_to)
                
            # CREATE PROMPT 

            # party_txt = f'{from_text}{party}' # This is for when doing an entity or persona experiment, where the party is included in the prompt.
            party_txt = ''  # No party-specific prefix for ideology experiment
            
            # prompt formats from model cards
            
            if  model_name == 'Llama3-instruct' or model_name == 'Llama3-70B-instruct':
                input_prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                {system_prompt_1}"{party_txt}"{system_prompt_2}<|eot_id|>
                <|start_header_id|>user<|end_header_id|>
                {user_prompt_1}{x}{user_prompt_2}<|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>
                """
            
            elif model_name == "Mistral-instruct":
                if prompt_template_no == 0:
                    input_prompt = f"""
                    System: {system_prompt_1}"{party_txt}"{system_prompt_2}
                    User: {user_prompt_1}{x}{user_prompt_2}
                    Assistant:
                    """ 
                    #party_txt = '' if party=='' else f' fra {party}'
                    #input_prompt = f"""
                    #System: Stem for eller imot dette forslaget{party_txt}. Svar kun med for eller mot.
                    #User: Forslag: "{x}"
                    #Assistant:
                    #"""
                elif prompt_template_no == 1:
                    messages = [
                            {"role": "system", "content": f"{system_prompt_1}'{party_txt}'{system_prompt_2}"},
                            {"role": "user", "content": f"{user_prompt_1}{x}{user_prompt_2}"},
                    ]
                    input_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                    )
                    print(input_prompt)
                elif False or prompt_template_no == 2:
                    input_prompt = f"""
                    <|im_start|> user
                    {system_prompt_1}"{party_txt}"{system_prompt_2}
                    {user_prompt_1}{x}{user_prompt_2}<|im_end|>
                    <|im_start|> assistant
                    """
                # Does not work
                elif False or prompt_template_no == 2:
                    input_prompt = f"""
                    [INST] Vote for or against this political motion in Norwegian. [/INST] [INST] Reply with only one word 'For' or 'Against' [/INST]
                    """
                
            elif model_name == "deepseek-llm-7b-base":
                messages = [
                    {"role": "system", "content": f"{system_prompt_1}'{party_txt}'{system_prompt_2}"},
                    {"role": "user", "content": f"{user_prompt_1}{x}{user_prompt_2}"},
                ]
                input_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                ) 
            
            elif model_name == "Falcon3-instruct" or model_name == "Gemma2-instruct":
                messages = [
                         {"role": "user", "content": f"{system_prompt_1}'{party_txt}'{system_prompt_2}\n\n{user_prompt_1}{x}{user_prompt_2}"},
                        {"role": "assistant", "content": f""}
                ]
            
                input_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            

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
                """
                outputs_temp0 = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temp_low,
                    return_dict_in_generate=True,
                )
                #prompt the model with temperature 1 to extract the logit scores before temperature scaling (needed to produce the probability metric)
                outputs_probabilities = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temp_high, 
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                """
                #"""
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=False, # deterministic
                    temperature=1, # No scaling
                    #top_k=0, # No cut off
                    #top_p=1, # No cut off
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                #"""
            else:
                outputs_temp0 = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=.0001,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                #print(outputs_temp0)
                #prompt the model with temperature 1 to extract the logit scores before temperature scaling (needed to produce the probability metric)
                outputs_probabilities = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=1,
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
            
            generated_text = generated_text.lower().strip()
            generated_text = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑüÜ ]', '', generated_text)
            generated_text = generated_text if generated_text != "" else "blank"
            # print(f"'{party}','{generated_text}'")

            if 'abst' in generated_text:
                generated_text = 'abstención'
            elif 'favor' in generated_text:
                generated_text = 'a favor'
            elif 'contra' in generated_text:
                generated_text = 'en contra'
            elif 'no' == generated_text.strip():  # just in case
                generated_text = 'en contra'
            else:
                generated_text = 'otro'


            print(f"'{id}','{generated_text}'")
            suffix = ""  # No party suffix


            for_prob = 0
            against_prob = 0
            
            # Retrieve logit scores
            # NB CHANGE
            #logits = outputs_probabilities.scores
            if model_name != "Mistral-instruct":
                logits = outputs.scores
            else:
                logits = outputs_probabilities.scores   
            
            # Calculatet the top_k tokens and probabilities for each generated token
            top_k = 5  # we found that in all vases the tokens representing 'for' and 'against' were fount within top_k = 5
            probabilities = torch.softmax(logits[0], dim=-1) # transform logit scores to probabilities
        
            top_probs, top_indices = torch.topk(probabilities, top_k)
            top_indices = top_indices.tolist()[0]  # Convert the tensor to a list of indices
            top_probs = top_probs.tolist()[0]  # Convert the tensor to a list of probabilities
            top_tokens = tokenizer.convert_ids_to_tokens(top_indices) # Convert the indices to tokens
        
            #extract the probabilities for the tokens 'for' and 'against' from the top_k tokens
            for_prob, against_prob, abstain_prob = extract_probs(top_tokens, top_probs)

                
            suffix = f"_{party_short}" if party_short != "" else ""
            result_df.loc[result_df['initiative_id'] == id,
              [f'{model_name}_vote', f'{model_name}_for_prob', f'{model_name}_against_prob', f'{model_name}_abstain_prob']
             ] = [generated_text, for_prob, against_prob, abstain_prob]



            if i % 100 == 0:
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

    if DEBUG and small_data_size == 200 and exp_type == "ideology":
        update_model_summary(model_name, prompt_no, prompt_template_no, replace_start, result_df)
     
    if exp_type == "ideology": 
        print(result_df[f"{model_name}_vote"].value_counts()/len(result_df))
    
    elapsed_time = time.time() - start
    print(f"Experiment time {int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m {elapsed_time % 60:.2f}s")    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="ide", help="type of experiment")
    parser.add_argument("--model", type=str, default="Falcon3-7B-instruct", help="model to run")
    parser.add_argument("--prompt", type=int, default=1, help="prompt no")
    parser.add_argument("--template", type=int, default=0, help="prompt template, for models with more than one.")
    parser.add_argument("--replace", type=int, default=0, help="remove start")
    parser.add_argument("--cont", type=int, default=-1, help="continue exp")
    parser.add_argument("--debug", type=int, default=0, help="Debug. 0: No or 1: Yes")
    parser.add_argument("--datasize", type=int, default=20, help="Size of debug dataset (no effect if not debug)")

    args = parser.parse_args()
    exp_type = args.exp
    model_name = args.model

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
    replace_start = args.replace
    cont = args.cont
    small_data_size = args.datasize

    if exp_type == "ide": exp_type = "ideology"
    elif exp_type == "ent": exp_type = "entity"
    elif exp_type == "per": exp_type = "persona"

    lang = "ES"
    
    run_experiment(exp_type, model_name, prompt_no, replace_start, cont, DEBUG, prompt_template_no=prompt_template_no, small_data_size=small_data_size, lang=lang)
