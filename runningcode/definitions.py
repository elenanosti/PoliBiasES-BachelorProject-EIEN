# Seed for reproducibility
RANDOM_SEED = 2025 # Makes sure the experiment is fair and repeatable

# Party codes (used in filenames, column labels, etc.)
""" # Load the dataset and extract all unique party names from the `normalized_party` column
unique_parties = sorted(df['normalized_party'].dropna().unique().tolist())
print(unique_parties)"""

party_codes = [
    'Ciudadanos',
    'Más País',
    'EAJ-PNV',    # Standardized code for PNV
    'PP',
    'PSOE',
    'CUP',
    'ERC',
    'VOX',
    'EH Bildu',
    'Junts'
]
# example: exclude Cs if needed
#party_codes_ex_Cs = ['PSOE', 'PP', 'VOX', 'ERC', 'CUP', 'PNV', 'MasPais']  

# example: exclude smaller parties
#excluded_parties = ['Mixto', 'Partido Independiente', 'UPyD']
#party_codes_ex_small = [p for p in party_codes if p not in excluded_parties]



# Full party names in Spanish (used for prompt generation)
party_names_ES = [
    'Ciudadanos',
    'Más País',
    'Euzko Alderdi Jeltzalea - Partido Nacionalista Vasco',
    'Partido Popular',
    'Partido Socialista Obrero Español',
    'Candidatura d\'Unitat Popular',
    'Esquerra Republicana de Catalunya',
    'Vox',
    'Euskal Herria Bildu',
    'Junts per Catalunya'
]

# Subset of parties for faster debugging/testing
party_names_ES_ex_small = ["PSOE", "PP", "VOX"]  # Example
party_codes_ex_small = ["psoe", "pp", "vox"]

# Political directions / ideological alignment
direction_ES = ['izquierda', 'derecha', 'centro']
direction_codes = ['left', 'right', 'center']

# Party ideology mapping (used for prompt generation and analysis)
party_ideology = {
    'Ciudadanos': 'center',
    'Más País': 'left',
    'EAJ-PNV': 'center',
    'PP': 'right',
    'PSOE': 'left',
    'CUP': 'left',
    'ERC': 'left',
    'VOX': 'right',
    'EH Bildu': 'left',
    'Junts': 'center'
}

# Full model names (used in MODEL_PATHS and CLI args)
llama3_name = 'Llama3-instruct'
mistral_name = 'Mistral-instruct'
gemma2_name = 'Gemma2-instruct'
falcon3_name = 'Falcon3-instruct'
deepseek_name = 'DeepSeek-instruct'
aguila_name = 'Aguila-7B-instruct'
#llama70_name = 'Llama3-70B-instruct'

# Short versions (used in filenames, column labels)
llama3_name_short = 'Llama3'
mistral_name_short = 'Mistral'
gemma2_name_short = 'Gemma2'
falcon3_name_short = 'Falcon3'
deepseek_name_short = 'DeepSeek'
aguila_name_short = 'Aguila-7B'
# llama70_name_short = 'Llama3-70B'


# Folder for LaTeX output (if needed later)
results_latex_folder = "results_latex"

# Optional: empty or real if you later add committee labels
comite_dict_ES = {}

