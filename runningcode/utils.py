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

def get_dataset(DEBUG, small_data_size = 20, variant=0, exp="ideology", lang="ES", replace_start=0, drop_motiontypes=False):
    cols = [] # Set the correct columns depending on the variant (adjusted to benchmark column names)

    # CHOOSING THE DATASET BASED ON THE EXPERIMENT TYPE (exp) AND VARIANT
    # For LLMs 
    if variant == 0: # WHEN THE LLM VOTES ON MOTIONS
        cols = ['id', 'initiative', 'documentcategory', "subcategory"]  # Only load the relevant fields

    # Parties votes: For experiments where you're comparing model predictions to how political parties actually voted.
    elif variant == 1: # WHEN THE LLM VOTES ON MOTIONS AND WE WANT TO COMPARE IT TO PARTY VOTES
        cols = ['id', 'initiative', 'documentcategory', "Ciudadanos", "Más País", "PNV", "PP", "PSOE", "CUP", "ERC", "VOX", "EH Bildu", "Junts"]
        cols += [f"{p}_vote" for p in party_codes]
        print(cols)    
    
    else:
        print("Invalid variant.")
        return

    # OTHER VARIANTS (commented out for now, but can be used if needed):
    # If you want to load the dataset with all motions and their metadata, including text.
    # elif variant == 2:
    #     cols = None  

    # Categories: If you're trying to classify motions into policy categories like Economy, Health, etc.
    # elif variant == 3:
    #     cols = ['id', 'motion_id', 'session_id', 'text' if replace_start != 2 else 'text_modified', 'category', 'document_group']
    # Load everything from the file (no column filtering).
    
    # Load the dataset based on the experiment type (exp):
    fname = ""  # Chooses a file to load:
    if exp == "topic": # for annotating with missing topic 
        fname =f"data/All_initiatives_2016-2025.csv" # If you're running a topic prediction task (exp == "category"), 
                                               # it loads a file where the topic column is missing or incomplete, i.e., all_votes_no_top_ES.csv.
    else:
        fname = "data/All_initiatives_2016-2025.csv"  # If not, it loads the default dataset with full motion metadata and text.
    df = pd.read_csv(fname, usecols=cols)

    print(len(df)) # Prints the row count, mostly for logging/debugging purposes
def update_model_summary(model_name, prompt_no, prompt_template_no, replace_start, result_df, jb=0):    
    # Compute the vote distribution for the run
    vote_series = result_df[f"{model_name}_vote"].value_counts() / len(result_df)
    vote_distribution = vote_series.to_dict()

    KNOWN_VOTE_KEYS = ["for", "mot", "against", "blank"]
    
    # Build a dictionary for the row to insert/update.
    row = {
        "model": model_name,
        "prompt": prompt_no,
        "prompt_template": prompt_template_no,
        "replace": replace_start,
        "jb": jb,
    }
    
    # Initialize all known keys to 0
    for known_key in KNOWN_VOTE_KEYS:
        row[known_key] = 0
    
    # Include the vote distribution values (e.g., for, against, etc.)
    other_sum = 0.0
    for key, value in vote_distribution.items():
        if key in KNOWN_VOTE_KEYS:
            row[key] = value
        else:
            other_sum += value
    row["other"] = other_sum
    
    # Path for the summary CSV file
    summary_file = "results/summary_results.csv"
    
    # If the file exists, load it; otherwise, create a new DataFrame with the required columns.
    # Load or create summary_df
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file, index_col=None)
    else:
        summary_df = pd.DataFrame(columns=["model", "prompt", "prompt_template", "replace", "jb"])


    needed_cols = ["model", "prompt", "prompt_template", "replace"] + KNOWN_VOTE_KEYS + ["other"]
    for col in needed_cols:
        if col not in summary_df.columns:
            summary_df[col] = None

    print("old row")
    print(row)
    # Identify a row matching the current parameters (model, prompt, replace)
    mask = (
        (summary_df["model"] == model_name) &
        (summary_df["prompt"] == prompt_no) &
        (summary_df["prompt_template"] == prompt_template_no) &
        (summary_df["replace"] == replace_start) &
        (summary_df["jb"] == jb)
    )
    
    matched_indices = summary_df.index[mask]  # All row indices that match

    if len(matched_indices) > 1:
        print("update > 1")
        # Keep ONLY the first matching row, drop the others
        keep_idx = matched_indices[0]
        drop_idx = matched_indices[1:]
        summary_df.drop(index=drop_idx, inplace=True)
        # Update the first matching row for matching columns
        for key, value in row.items():
            if key in summary_df.columns:
                summary_df.loc[keep_idx, key] = value
    elif len(matched_indices) == 1:
        print("update = 1")
        idx = matched_indices[0]
        # Update the matching row
        for key, value in row.items():
            if key in summary_df.columns:
                summary_df.loc[idx, key] = value
    else:
        print("new row")
        # No match => append a new row ensuring all columns are represented
        new_row = {col: row.get(col, np.nan) for col in summary_df.columns}
        print(new_row)
        summary_df = pd.concat([summary_df, pd.DataFrame([new_row])], ignore_index=True)

    summary_df.sort_values(by=["model", "prompt", "prompt_template", "replace", "jb"], inplace=True)
    
    # Save the updated DataFrame back to the CSV file.
    summary_df.to_csv(summary_file, encoding='utf-8-sig', index=False)
 
 # The next part of the code loads LLM-predicted categories from a CSV file and merges them into the main dataset.
    """
    spanish_to_english_categories = {
    "Economía": "Economy",
    "Educación": "Education",
    "Sanidad": "Health",
    "Justicia": "Justice",
    "Transporte": "Transport",
    "Defensa": "Defense",
    "Exteriores": "Foreign Affairs",
    "Trabajo": "Labor",
    "Cultura": "Culture",
    "Medio Ambiente": "Environment",
    "Agricultura": "Agriculture",
    "Interior": "Interior",
    "Vivienda": "Housing",
    "Tecnología": "Technology",
    "Energía": "Energy",
    "Seguridad Social": "Social Security",
    "Otros": "Other"
}
    # Load LLM-predicted categories (from results CSV)
    df_pred_cat = pd.read_csv("results/gpt-4o-mini_results_category_ES_prompt=1.csv")
    #print(df_pred_cat['gpt-4o-mini_vote'].unique())
    
    print("is NaN:", df['category'].isna().sum())
    
    # Merge predictions into your main dataset by motion ID
    df = df.merge(df_pred_cat[['id', 'gpt-4o-mini_category']], on='id', how='left')
    
    # Fill missing values in the 'category' column using LLM predictions
    missing_mask = df['category'].isna()

    # Try translating the predicted category to English, fallback to raw label
    df.loc[missing_mask, 'category'] = df.loc[missing_mask, 'gpt-4o-mini_category'] \
        .map(spanish_to_english_categories) \
        .fillna(df.loc[missing_mask, 'gpt-4o-mini_category'])
    
    # Drop the temporary prediction column
    df.drop(columns=['gpt-4o-mini_category'], inplace=True)
    print("is NaN:", df['category'].isna().sum())    
    
    # Dropping irrelevant categories
    # print("before drop unpolitical categories:", len(df))
    # del_cat = ['Presidency of the Storting', 'Control and Constitution', 'Other'] # Choosing which ones
    # df = df[~df['category'].isin(del_cat)]
    # print("after drop unpolitical categories:", len(df))

    # Drop duplicates
    print("before drop duplicates:", len(df))
    df.drop_duplicates(subset='initiative_text', keep='first', inplace=True)
    print("after drop duplicates:", len(df))


def update_model_summary(model_name, prompt_no, prompt_template_no, replace_start, result_df, jb=0):    
    # Compute the vote distribution for the run
    vote_series = result_df[f"{model_name}_vote"].value_counts() / len(result_df)
    vote_distribution = vote_series.to_dict()

    KNOWN_VOTE_KEYS = ["afavor", "encontra", "abstencion"]
    
    # Build a dictionary for the row to insert/update.
    row = {
        "model": model_name,
        "prompt": prompt_no,
        "prompt_template": prompt_template_no,
        "replace": replace_start,
        "jb": jb,
    }
    
    # Initialize all known keys to 0
    for known_key in KNOWN_VOTE_KEYS:
        row[known_key] = 0
    
    # Include the vote distribution values (e.g., for, against, etc.)
    other_sum = 0.0
    for key, value in vote_distribution.items():
        if key in KNOWN_VOTE_KEYS:
            row[key] = value
        else:
            other_sum += value
    row["other"] = other_sum
    
    # Path for the summary CSV file
    summary_file = "results/summary_results.csv"
    
    # If the file exists, load it; otherwise, create a new DataFrame with the required columns.
    # Load or create summary_df
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file, index_col=None)
    else:
        summary_df = pd.DataFrame(columns=["model", "prompt", "prompt_template", "replace", "jb"])


    needed_cols = ["model", "prompt", "prompt_template", "replace"] + KNOWN_VOTE_KEYS + ["other"]
    for col in needed_cols:
        if col not in summary_df.columns:
            summary_df[col] = None

    print("old row")
    print(row)
    # Identify a row matching the current parameters (model, prompt, replace)
    mask = (
        (summary_df["model"] == model_name) &
        (summary_df["prompt"] == prompt_no) &
        (summary_df["prompt_template"] == prompt_template_no) &
        (summary_df["replace"] == replace_start) &
        (summary_df["jb"] == jb)
    )
    
    matched_indices = summary_df.index[mask]  # All row indices that match

    if len(matched_indices) > 1:
        print("update > 1")
        # Keep ONLY the first matching row, drop the others
        keep_idx = matched_indices[0]
        drop_idx = matched_indices[1:]
        summary_df.drop(index=drop_idx, inplace=True)
        # Update the first matching row for matching columns
        for key, value in row.items():
            if key in summary_df.columns:
                summary_df.loc[keep_idx, key] = value
    elif len(matched_indices) == 1:
        print("update = 1")
        idx = matched_indices[0]
        # Update the matching row
        for key, value in row.items():
            if key in summary_df.columns:
                summary_df.loc[idx, key] = value
    else:
        print("new row")
        # No match => append a new row ensuring all columns are represented
        new_row = {col: row.get(col, np.nan) for col in summary_df.columns}
        print(new_row)
        summary_df = pd.concat([summary_df, pd.DataFrame([new_row])], ignore_index=True)

    summary_df.sort_values(by=["model", "prompt", "prompt_template", "replace", "jb"], inplace=True)
    
    # Save the updated DataFrame back to the CSV file.
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


##########  plotting  ##########

def plot_landscape2(df, title, models, exp, party_codes, colors_models, colors_parties, exp_var, x_pts=None, y_pts=None, signs=[1, 1], pca_model=None, show=True):
    pca_df, pca_model = do_PCA(df, models, party_codes, pca_model)
    # DO plotting
    make_landscape_plot(pca_df, title, models, exp, party_codes, colors_models, colors_parties, exp_var, x_pts, y_pts, signs, show)
    return pca_df, pca_model

def do_PCA(df, models, party_codes, pca_model=None, n_components=2):
    
    columns = party_codes+models
    
    df[columns] = df[columns].fillna(0)
    df[columns] = df[columns].apply(pd.to_numeric)

    df_transposed = df[columns].transpose()
    if pca_model == None:
        pca_model = PCA(n_components=n_components)
        pca_result = pca_model.fit_transform(df_transposed)
    else:
        pca_result = pca_model.transform(df_transposed)
        
    print(len(pca_model.mean_))
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PCA{i+1}' for i in range(pca_result.shape[1])], index=df_transposed.index)
    print(pca_df)
    return pca_df, pca_model

def make_landscape_plot(pca_df, title, models, exp, party_codes, colors_models, colors_parties, exp_var, x_pts=None, y_pts=None, signs=[1, 1], show=True, xlim=None, ylim=None):
    """
    if exp == "NL":
        colors_parties = [
        '#E00000',  # Replaces Indigo
        '#0CB54F',  # Forest Green
        '#C90068',  # Pink VVD
        '#FFAE00',  # Yellow NSC
        '#DB7093',  # D66
        '#E74A18',  # BBB red
        '#FF69B4',  # Hot Pink
        '#79A000',  # Spring Green
        '#DE37FF',  # Pink christenunie
        '#9ACC00',  # Pale Green
        '#016D28',  # PVDD
        '#FF6C00',  # orange SGP
        '#A90000',  # Replaces Plum
        '#499275',  # Chartreuse
        '#AB0000']  # JA21 red
    else:
        colors_parties = ['#D91A39','#3C79C1','#0A8E3E','#18295E','#FC9A2B','#85C046','#B21D62','#064B2F','#701C44','#F75822']
    
    colors_models = ['#4682B4', '#87CEEB', '#1E90FF','#000080']
    """
    print(colors_parties)
    print(colors_models)
    colors = colors_parties + colors_models
    
    plt.figure(figsize=(13, 13))

    df_plot = pca_df.copy()
    
    df_plot.iloc[:, 0] = signs[0] * df_plot.iloc[:, 0] 
    df_plot.iloc[:, 1] = signs[1] * df_plot.iloc[:, 1] 
    
    texts = []
    pca_cols = df_plot.columns
    print(pca_cols)
    for idx, col in enumerate(df_plot.index):
        try:
            if idx >= 9+7:
                marker_style = '^'
            elif idx >= len(df_plot.index) - len(models):
                marker_style = '*'
            else:
                marker_style = 'o'  # Default marker style (circle)

            plt.scatter(df_plot.loc[col, pca_cols[0]], df_plot.loc[col, pca_cols[1]], 
                        label=col, s=400, color=colors[idx % len(colors)], marker=marker_style)
            x_offset = 0.05  # Adjust as needed
            y_offset = 0.05  # Adjust as needed
            texts.append(plt.text(df_plot.loc[col, pca_cols[0]] + x_offset, df_plot.loc[col, pca_cols[1]] + y_offset, col, fontsize=13, fontweight='medium'))
        except Exception as e:
            print(f"Error plotting index {idx}: {e}")

    if xlim == None:        
        adjust_text(texts, 
                    force_text=(4, 3),#(4, 3), 
                    expand_text=(0, 0),#(0, 0), 
        
                    only_move={'text': 'xy'})

    plt.xlabel(pca_cols[0], fontsize=18)
    plt.ylabel(pca_cols[1], fontsize=18)
    #plt.title(title)
    plt.grid(True, linestyle=':', linewidth=1, alpha=0.7)

    
    if x_pts != None:
        for i in range(len(x_pts)): 
            plt.plot([-60, 80], [-70, 80], color='black', linestyle='--', label='Extra line')
            plt.plot([-60, 80], [70, -70], color='black', linestyle='--', label='Extra line')

    if xlim != None:
        plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)
    
    #plt.savefig(f'Results/plots/{title}_plot.pdf')
    plt.savefig(f"{results_latex_folder}/{title.replace(' ', '_')}_plot_{exp_var}.png")
    if show:
        plt.show()
    plt.close()

"""
# def make_landscape_plot2(...):
#     # … all your setup up to the scatter loop …
#     fig, ax = plt.subplots(figsize=(13,13))
#     # shift the plotting area up a bit so legend fits below
#     fig.subplots_adjust(bottom=0.2)

#     for idx, col in enumerate(df_plot.index):
#         # decide if this point is one of the “models”
#         is_model = (idx >= len(df_plot.index) - len(models))

#         # choose marker
#         if idx >= 9+7:
#             marker_style = '^'
#         elif is_model:
#             marker_style = '*'
#         else:
#             marker_style = 'o'

#         # only give a real label to models; suppress legend for others
#         this_label = col if is_model else '_nolegend_'

#         ax.scatter(df_plot.loc[col, pca_cols[0]],
#                    df_plot.loc[col, pca_cols[1]],
#                    label=this_label,
#                    s=400,
#                    color=colors[idx % len(colors)],
#                    marker=marker_style)

#         # only annotate non‑models
#         if not is_model:
#             x_off, y_off = 0.05, 0.05
#             ax.text(df_plot.loc[col, pca_cols[0]]+x_off,
#                     df_plot.loc[col, pca_cols[1]]+y_off,
#                     col,
#                     fontsize=15,
#                     fontweight='medium')

#     # … your adjust_text, labels, grid, etc. …

#     # put legend under the plot, one column per model
#     ax.legend(title="Models",
#               loc='upper center',
#               bbox_to_anchor=(0.5, -0.1),
#               ncol=len(models),
#               fontsize=12)

#     # save & show as before
#     plt.savefig(...)
#     if show: plt.show()
#     plt.close()
# """


"""
def plot_landscape(df, title, models, exp, x_pts=None, y_pts=None):
    '''
    given the df consisting of the votes of party and the votes of each of them models, we apply PCA do compress the vectors into 2-dimensions,
    and plot them using a scatterplot, whereby we use colors and shapes to distinguish between the models and ideologies of the existing parties.
    '''
    if exp == "ES":
        columns = party_codes + models
    else:
        pass
    
    df[columns] = df[columns].fillna(0)
    df[columns] = df[columns].apply(pd.to_numeric)

    df_transposed = df[columns].transpose()

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_transposed)


    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'], index=df_transposed.index)

    if exp == "NL":
        colors_parties = [
        '#E00000',  # Replaces Indigo
        '#0CB54F',  # Forest Green
        '#C90068',  # Pink VVD
        '#FFAE00',  # Yellow NSC
        '#DB7093',  # D66
        '#E74A18',  # BBB red
        '#FF69B4',  # Hot Pink
        '#79A000',  # Spring Green
        '#DE37FF',  # Pink christenunie
        '#9ACC00',  # Pale Green
        '#016D28',  # PVDD
        '#FF6C00',  # orange SGP
        '#A90000',  # Replaces Plum
        '#499275',  # Chartreuse
        '#AB0000']  # JA21 red

    else:
        colors_parties = ['#D91A39','#3C79C1','#0A8E3E','#18295E','#FC9A2B','#85C046','#B21D62','#064B2F','#701C44','#F75822']
    
    colors_models = ['#4682B4', '#87CEEB', '#1E90FF','#000080']
    colors = colors_parties + colors_models
    
    plt.figure(figsize=(13, 13))

    texts = []
    for idx, col in enumerate(pca_df.index):
        try:
            if idx >= len(pca_df.index) - len(models):
                marker_style = '*'
            else:
                marker_style = 'o'  # Default marker style (circle)

            plt.scatter(pca_df.loc[col, 'PCA1'], pca_df.loc[col, 'PCA2'], 
                        label=col, s=400, color=colors[idx % len(colors)], marker=marker_style)
            x_offset = 0.05  # Adjust as needed
            y_offset = 0.05  # Adjust as needed
            texts.append(plt.text(pca_df.loc[col, 'PCA1'] + x_offset, pca_df.loc[col, 'PCA2'] + y_offset, col, fontsize=18, fontweight='medium'))
        except Exception as e:
            print(f"Error plotting index {idx}: {e}")

            
    adjust_text(texts, 
                force_text=(4, 3),#(4, 3), 
                expand_text=(0, 0),#(0, 0), 
    
                only_move={'text': 'xy'})

    plt.xlabel('PCA1', fontsize=18)
    plt.ylabel('PCA2', fontsize=18)
    #plt.title(title)
    plt.grid(True, linestyle=':', linewidth=1, alpha=0.7)

    
    if x_pts != None:
        for i in range(len(x_pts)): 
            plt.plot([-60, 80], [-70, 80], color='black', linestyle='--', label='Extra line')
            plt.plot([-60, 80], [70, -70], color='black', linestyle='--', label='Extra line')
    
    #plt.savefig(f'Results/plots/{title}_plot.pdf')
    plt.savefig(f'imgs/{title}_plot.png')
    plt.show()
    
    return pca_df
"""

def violinplot(certainty_vals, labels, colors, exp_var, fname):
    '''
    given the computer probabily metrics the function plots a violinplot per model
    '''
    
    #colors = ['#1E90FF', '#87CEEB','#000080', '#4682B4']  # Navy, Sky Blue, Dodger Blue, Steel Blue

    # Set figure size
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set ylabel
    ax.set_ylabel('Certainty', fontsize=14)

    # Create the violin plot
    vplot = ax.violinplot(certainty_vals, showmedians=True)

    # Customize each violin plot with a different color
    for i, body in enumerate(vplot['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor('black')
        body.set_alpha(0.7)

    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = vplot[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)

    # Set the color and width of the median line
    vplot['cmedians'].set_color('black')
    vplot['cmedians'].set_linewidth(2)

    # Set the x-tick labels
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=14)

    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    # Display the plot
    #plt.savefig('violin_plot.pdf')
    plt.savefig(f'{results_latex_folder}/{fname}_{exp_var}.png')
    plt.show()


def plot_heatmap(fname, positive_data, negative_data, party_codes, models, exp_var):
    '''
    given the calculated positive entity bias and negative entity bias, we plot the generated biases across each political party and model
    distinguising between positive and negative bias through color and the degree of bias through the intensity of their respective colors.
    '''
    # Categories for the columns
    #categories = [
    #    'PvdD', 'GL-PvdA', 'Volt', 'SP', 'DENK', 'D66', 'CU', 
    #    'NSC', 'CDA', 'BBB', 'VVD', 'SGP', 'PVV', 'FVD', 'JA21'
    #]

    # Group names for the rows
    #groups = ['GPT4o-mini', 'GPT3.5-turbo', 'LLaMA3', 'LLaMA2']

    categories = party_codes
    groups = models
    
    # Create subplots for positive and negative biases
    fig = sp.make_subplots(
        rows=2, cols=1, 
        subplot_titles=('Positive Bias', 'Negative Bias'),
        vertical_spacing=0.15  # Adjust this value to reduce the space between plots
    )

    # Positive Bias Heatmap
    positive_heatmap = go.Heatmap(
        z=positive_data,
        x=categories,
        y=groups,
        colorscale='greens',  # Using a green scale for positive
        zmin=0,
        zmax=60,
        colorbar=dict(title='(%)', x=1.02, y=0.8, len=0.4),
        text=positive_data,
        texttemplate="%{text}",  # Display the values
        textfont={"size": 12},
    )
    fig.add_trace(positive_heatmap, row=1, col=1)

    # Negative Bias Heatmap
    negative_heatmap = go.Heatmap(
        z=negative_data,
        x=categories,
        y=groups,
        colorscale='reds',  # Using a red scale for negative
        zmin=0,
        zmax=60,
        colorbar=dict(title='(%)', x=1.02, y=0.2, len=0.4),
        text=negative_data,
        texttemplate="%{text}",  # Display the values
        textfont={"size": 12},
    )
    fig.add_trace(negative_heatmap, row=2, col=1)

    # Layout adjustments
    fig.update_layout(
        height=1000,
        width=1200,
        showlegend=False,
        xaxis=dict(tickangle=0),  # Rotate x-axis labels
        yaxis=dict(tickmode='array', tickvals=np.arange(len(groups)), ticktext=groups, autorange='reversed'),
        yaxis2=dict(autorange='reversed')
    )
    
    # Show the plot
    pio.write_image(fig, f'{results_latex_folder}/{fname}_{exp_var}.png', format='png')
    fig.show()
    #pio.write_image(fig, 'Results/plots/heatmap_plot.pdf', format='pdf')