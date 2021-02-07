import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import dump, load

nrows = None

# default for suzuki
suzuki_n_metals = 28
suzuki_n_ligands = 23
suzuki_n_bases = 35
suzuki_n_solvents = 10
suzuki_n_additives = 17

suzuki_cutoffs = [
    0,
    suzuki_n_metals,
    suzuki_n_metals + suzuki_n_ligands,
    suzuki_n_metals + suzuki_n_ligands + suzuki_n_bases,
    suzuki_n_metals + suzuki_n_ligands + suzuki_n_bases + suzuki_n_solvents,
    suzuki_n_metals + suzuki_n_ligands + suzuki_n_bases + suzuki_n_solvents + suzuki_n_additives
]
suzuki_group_names = [
    'metals', 'ligands', 'bases', 'solvents', 'additives'    
]
suzuki_group_initials = [
    'M', 'L', 'B', 'S', 'A'
]



def find_correct_path(paths=None):
    try:
        FileNotFoundError
    except NameError:
        FileNotFoundError = IOError
    if paths is None:
        paths = [
<<<<<<< HEAD
            '/Users/agarbuno/gDrive/postdoc/cms-273/data/',
            '/media/sf_alexb/Google Drive/Caltech/cms-273/data/',
            '/home/ubuntu/cms273/data/',
            '/home/ubuntu/chemistry-ai/data/'
=======
            '/home/ubuntu/reaction-gcnn/data/'
>>>>>>> 837d55df5229b4f7aac4a3b92c0674f074d6a226
        ]
    for path in paths:
        if os.path.exists(path):
            return path
    if df is None:
        raise FileNotFoundError("None of the paths are valid")


<<<<<<< HEAD
def load_base_xlsx(xl_name='Alyllic Oxidation Initial Training Data.xlsx',
=======
def load_base_xlsx(xl_name='Negishi_binned_final.xlsx',
>>>>>>> 837d55df5229b4f7aac4a3b92c0674f074d6a226
				   sheet_name = 1,
				   nrows=3843,
				   usecols=tuple([1,2,3,4,5,6]),
				   columns=['reaction', 'temperature', 'yield', 'reagent', 'catalyst', 'solvent'],
                   paths=None):
    df = None
    my_path = find_correct_path(paths)
    df = pd.read_excel(my_path+xl_name, 
                 sheet_name = sheet_name, usecols = usecols, 
                 nrows = nrows, verbose = True)
    # Sometimes, the data is on sheet 2 instead
    if len(df) < 10:
        df = pd.read_excel(my_path+xl_name, 
                     sheet_name = 2, usecols = usecols, 
                     nrows = nrows, verbose = True)
    df.columns = columns
    df = df.truncate(after=nrows-1)
    return df, my_path

def topn_preds(Y_pred_proba, n=5, include_nulls=True, group_initials=None, cutoffs=None):
    # Return a dataframe of the predictions in their initial name format
    columns = ["pred_" + initial for initial in group_initials]
    all_preds = []
    
    for row in range(len(Y_pred_proba)):
        reaction_preds = []
        for i in range(len(cutoffs) - 1):
            group_preds = []
            num_group_options = cutoffs[i+1] - cutoffs[i]
            preds = Y_pred_proba[row][cutoffs[i]:cutoffs[i+1]]
            if include_nulls:
                null_pred = Y_pred_proba[row][cutoffs[-1]+i]
                preds = np.hstack((preds, null_pred))
            topn_preds_inds = np.argsort(-preds)[:n]
            for ind in topn_preds_inds:
                # Skip if the model gives this prediction zero probability
                if preds[ind] == 0:
                    continue
                if ind == num_group_options:
                    group_preds.append('Null')
                else:
                    # Turn the index to a reagent name
                    # i.e. ind = 3 in solvents turns into S3
                    group_preds.append(group_initials[i] + str(ind + 1))
            reaction_preds.append(group_preds)
        all_preds.append(reaction_preds)
    preds_df = pd.DataFrame(all_preds, columns=columns)
    return preds_df


def topn_recall(Y_pred_proba, Y_test_raw, n=5, get_categorized=False, has_null=False, cutoffs=suzuki_group_names, group_names=suzuki_group_names):
    
    topn_score_sum = 0
    pos_labels = Y_test_raw.sum()
    total_reagents = cutoffs[-1]
    
    categorized_score = np.zeros(len(group_names))
    categorized_pos_labels = np.zeros(len(group_names))
    
    for row in range(len(Y_test_raw)):
        for i in range(len(group_names)):
            labels = Y_test_raw[row][cutoffs[i]:cutoffs[i+1]]
            preds = Y_pred_proba[row][cutoffs[i]:cutoffs[i+1]]

            if has_null:
                labels = np.append(labels, Y_test_raw[row][total_reagents + i])
                preds = np.append(preds, Y_pred_proba[row][total_reagents + i])
            
            pos_labels_inds = np.argwhere(labels == 1)
            topn_preds_inds = np.argsort(-preds)[:n]
            for ind in pos_labels_inds:
                if ind in topn_preds_inds:
                    topn_score_sum += 1
                    categorized_score[i] += 1
            categorized_pos_labels[i] += labels.sum()
                    
    if get_categorized:
        return categorized_score / categorized_pos_labels
    return topn_score_sum / pos_labels           
  
def load_X_and_Y(X_file_name, Y_file_name, nrows=None):
    my_path = find_correct_path()
    X_raw = pd.read_csv(my_path + X_file_name, nrows=nrows)
    Y_raw = pd.read_csv(my_path + Y_file_name, nrows=nrows, index_col=0, header=None)
    reagent_cols = [c for c in X_raw.columns if 'reagent' in c]
    other_cols = [c for c in X_raw.columns if 'reagent' not in c]
    Y = X_raw[reagent_cols][X_raw.index.isin(Y_raw.index)]
    X = X_raw[other_cols]
    X['yield'] = Y_raw
    if 'reaction_id' in X.columns:
        X = X.drop(columns='reaction_id')
    return X, Y

def load_X_and_Y_classification(X_file_name, Y_file_name, nrows=None):
    my_path = find_correct_path()
    X = pd.read_csv(my_path + X_file_name, nrows=nrows, index_col='Unnamed: 0')
    Y = pd.read_csv(my_path + Y_file_name, nrows=nrows, index_col='Unnamed: 0')
    return X, Y

def add_null_labels(Y, cutoffs=suzuki_cutoffs, group_names=suzuki_group_names):
    print("cutoffs", cutoffs)
    col_names = ["null_" + group_name for group_name in group_names]
    for col_name in col_names:
        Y[col_name] = 0
    for row in Y.index:
        for i in range(len(cutoffs) - 1):
            try:
                labels = Y.loc[row][cutoffs[i]:cutoffs[i+1]]
                if sum(labels) == 0:
                    Y.loc[row, col_names[i]] = 1
            except Exception as e:
                print("row: {}, cutoffs[i]: {}, cutoffs[i+1]: {}".format(
                    row, cutoffs[i], cutoffs[i+1]
                ))
                print(e)
                raise(e)
    return Y
  
def dataset_split(X, Y, percent_valid=0.1, percent_test=0.1):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = percent_test, random_state = 0)
    X_train, X_valid, Y_train, Y_valid = train_test_split(
    X_train, Y_train, test_size = percent_valid / (1 - percent_test), random_state = 0)
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

def train_lgbm_classifier(X_train, Y_train, model_file_name):
    clf = OneVsRestClassifier(LGBMClassifier(
        n_jobs=-1,
        max_depth=7,
        tree_method='gpu_hist',
        gpu_id=0,
        verbosity=2,
        eval_metric='aucpr'
    ))
    clf.fit(X_train, Y_train)
    dump(clf, model_file_name)
    return clf

def clf_predict(X_test, clf):
    Y_pred = clf.predict(X_test)
    Y_pred_proba = clf.predict_proba(X_test)
    return Y_pred, Y_pred_proba

def prediction_counts(Y_pred, cutoffs=suzuki_cutoffs):
    for i in range(len(cutoffs) - 1):
        print(group_names[i], "top 1 predictions count:",
            Y_pred.sum(axis=0)[cutoffs[i]: cutoffs[i+1]], '\n')

def top_5_and_1_scores(Y_pred_proba, Y_test, has_null=False, cutoffs=suzuki_cutoffs, group_names=suzuki_group_names, sheets_format=True):
    Y_test_raw = Y_test.values
    top5_scores = topn_recall(Y_pred_proba, Y_test_raw, 5, get_categorized=True, has_null=has_null, cutoffs=cutoffs, group_names=group_names)
    top1_scores = topn_recall(Y_pred_proba, Y_test_raw, 1, get_categorized=True, has_null=has_null, cutoffs=cutoffs, group_names=group_names)
    
    print('\n'.join([group_names[i] + " top 1 score: " + str(top1_scores[i]) for i in range(len(group_names))]))
    print()
    print('\n'.join([group_names[i] + " top 5 score: " + str(top5_scores[i]) for i in range(len(group_names))]))
    print()
    print("Top 5: {}".format(topn_recall(Y_pred_proba, Y_test_raw, 5, has_null=has_null, cutoffs=cutoffs, group_names=group_names)))
    print("Top 1: {}".format(topn_recall(Y_pred_proba, Y_test_raw, 1, has_null=has_null, cutoffs=cutoffs, group_names=group_names)))
    if sheets_format:
        print(topn_recall(Y_pred_proba, Y_test_raw, 1, has_null=has_null, cutoffs=cutoffs, group_names=group_names))
        print()
        print('\n'.join([str(top1_scores[i]) for i in range(len(group_names))]))
        print()
        print(topn_recall(Y_pred_proba, Y_test_raw, 5, has_null=has_null, cutoffs=cutoffs, group_names=group_names))
        print()
        print('\n'.join([str(top5_scores[i]) for i in range(len(group_names))]))
        