import pandas as pd
import os
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.io as pio
from definitions import *

def word_count(text):
    return len(text.split())

"""
variant:
0: Default
1: Categories
2: Parties votes
3: All
"""

def get_dataset(DEBUG, small_data_size = 20, variant=0, exp="ideology", lang="ES", drop_motiontypes=False):
    cols = []
    if variant == 0:
        cols = ['id', 'initiative', 'documentcategory', "subcategory"]
    elif variant == 1:
        cols = ['id', 'initiative', 'documentcategory', "Ciudadanos", "Más País", "PNV", "PP", "PSOE", "CUP", "ERC", "VOX", "EH Bildu", "Junts"]
        cols += [f"{p}_vote" for p in party_codes]
        print(cols)
    else:
        print("Invalid variant.")
        return

    fname = "data/All_initiatives_2016-2025.csv"
    df = pd.read_csv(fname, usecols=cols)
    print(len(df))
    return df

def update_model_summary(model_name, prompt_no, prompt_template_no, result_df, exp):
    vote_col = f"{model_name}_vote"
    KNOWN_VOTE_KEYS = ["a favor", "en contra", "abstención"]

    # Defensive: Check if vote column exists and is not empty
    if vote_col not in result_df.columns or result_df[vote_col].dropna().empty:
        print(f"⚠️ Warning: No votes found in column '{vote_col}'. Skipping summary update.")
        return

    vote_series = result_df[vote_col].value_counts(normalize=True)
    vote_distribution = vote_series.to_dict()

    row = {
        "model": model_name,
        "prompt": prompt_no,
        "prompt_template": prompt_template_no
    }
    for known_key in KNOWN_VOTE_KEYS:
        row[known_key] = 0

    other_sum = 0.0
    for key, value in vote_distribution.items():
        if key in KNOWN_VOTE_KEYS:
            row[key] = value
        else:
            other_sum += value
    row["other"] = other_sum

    summary_file = "results/summary_results.csv"
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file, index_col=None)
    else:
        summary_df = pd.DataFrame(columns=["model", "prompt", "prompt_template"] + KNOWN_VOTE_KEYS + ["other"])

    needed_cols = ["model", "prompt", "prompt_template"] + KNOWN_VOTE_KEYS + ["other"]
    for col in needed_cols:
        if col not in summary_df.columns:
            summary_df[col] = np.nan

    print("old row")
    print(row)
    mask = (
        (summary_df["model"] == model_name) &
        (summary_df["prompt"] == prompt_no) &
        (summary_df["prompt_template"] == prompt_template_no)
    )
    matched_indices = summary_df.index[mask]

    if len(matched_indices) > 1:
        print("update > 1")
        keep_idx = matched_indices[0]
        drop_idx = matched_indices[1:]
        summary_df.drop(index=drop_idx, inplace=True)
        for key, value in row.items():
            if key in summary_df.columns:
                summary_df.loc[keep_idx, key] = value
    elif len(matched_indices) == 1:
        print("update = 1")
        idx = matched_indices[0]
        for key, value in row.items():
            if key in summary_df.columns:
                summary_df.loc[idx, key] = value
    else:
        print("new row")
        new_row = {col: row.get(col, np.nan) for col in summary_df.columns}
        print(new_row)
        summary_df = pd.concat([summary_df, pd.DataFrame([new_row])], ignore_index=True)

    summary_df.sort_values(by=["model", "prompt", "prompt_template"], inplace=True)
    summary_df.to_csv(summary_file, encoding='utf-8-sig', index=False)

##########  process log probabilities   ########################

def logprob_to_prob(logprobs, no_log):
    probs = []
    for lprob in logprobs:
        if lprob == 'None':
            probs.append(0)
        elif no_log == False:
            probs.append(math.exp(float(lprob)))
        else:
            probs.append(float(lprob))
    return probs

def normalize_probs(voor_probs, tegen_probs, no_log=True):
    normalised_probs = []
    voor_probs = logprob_to_prob(voor_probs, no_log)
    tegen_probs = logprob_to_prob(tegen_probs, no_log)
    for voor, tegen in zip(voor_probs, tegen_probs):
        if voor > tegen:
            normalised_probs.append(voor / (voor + tegen))
        elif voor < tegen:
            normalised_probs.append(tegen / (voor + tegen))
        else:
            normalised_probs.append(0.5)
    return normalised_probs