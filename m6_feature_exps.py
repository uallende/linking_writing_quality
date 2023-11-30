import pandas as pd
import numpy as np

from m4_feats_functions import *
from m5_models import *
from m3_model_params import lgb_params_1, xgb_params
from sklearn.metrics import mean_squared_error
from itertools import combinations

import os, gc
import pandas as pd

def merge_pkl_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    merged_df = None

    for file in files:
        file_path = os.path.join(directory, file)
        df = pd.read_pickle(file_path)

        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, on=['id'], how='left')

    return merged_df


def load_feature_set(base_dir, feature_type, is_train=True):
    subdir = 'train' if is_train else 'test'
    prefix = 'train_' if is_train else 'test_'
    file_path = os.path.join(base_dir, subdir, prefix + feature_type + '.pkl')

    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}")
        return None

    return pd.read_pickle(file_path)


def get_feature_sets_from_folder(folder_path):
    feature_sets = set()
    for file in os.listdir(folder_path):
        if file.endswith('.pkl'):
            feature_set_name = os.path.splitext(file)[0]
            if feature_set_name.startswith('train_'):
                feature_set_name = feature_set_name.replace('train_', '')
            elif feature_set_name.startswith('test_'):
                feature_set_name = feature_set_name.replace('test_', '')
            feature_sets.add(feature_set_name)
    return list(feature_sets)

def cv_pipeline(train_feats, test_feats, seed, n_repeats, n_splits):

    target_col = ['score']
    drop_cols = ['id']
    train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]

    missing_cols = [col for col in train_cols if col not in test_feats.columns]
    missing_cols_df = pd.DataFrame({col: np.nan for col in missing_cols}, index=test_feats.index)
    test_feats = pd.concat([test_feats, missing_cols_df], axis=1)

    train_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_feats.replace([np.inf, -np.inf], np.nan, inplace=True)

    oof_results, rmse = run_lgb_cv(train_feats, test_feats, train_cols, target_col, lgb_params_1, seed, n_repeats, n_splits)
    return oof_results, rmse

def compare_with_baseline(base_dir, base_train_feats, base_test_feats, baseline_metrics, train_scores, seed=42, n_repeats=5, n_splits=10):
    results = []
    feature_sets = get_feature_sets_from_folder(os.path.join(base_dir, 'train'))

    for feature_set in feature_sets:
        print(feature_set)
        train_feats = load_feature_set(base_dir, feature_set, is_train=True)
        test_feats = load_feature_set(base_dir, feature_set, is_train=False)

        if train_feats is not None and test_feats is not None:
            train_feats = train_feats.merge(train_scores, on=['id'], how='left')
            train_feats = train_feats.merge(base_train_feats, on=['id'], how='left')
            test_feats = test_feats.merge(base_test_feats, on=['id'], how='left')

            oof_results, rmse = cv_pipeline(train_feats, test_feats, seed, n_repeats, n_splits)
            improvement = baseline_metrics - rmse
            results.append({'Feature Set': feature_set, 'Metric': rmse, 'Improvement': improvement})
            print(f'Features: {feature_set}. RMSE: {rmse:.6f}, Improvement: {improvement:.6f}')

            # Cleanup
            del train_feats, test_feats, oof_results
            gc.collect()
        else:
            print(f"Skipping feature set {feature_set} due to missing data.")
    return pd.DataFrame(results)
# Paths to the train and test directories
INPUT_DIR = 'kaggle/input/linking-writing-processes-to-writing-quality'
FEATURE_STORE = 'feature_store'
train_scores = pd.read_csv(f'{INPUT_DIR}/train_scores.csv')
base_train_feats = pd.read_pickle(f'{FEATURE_STORE}/base_feats/train_base_feats.pkl')
base_test_feats = pd.read_pickle(f'{FEATURE_STORE}/base_feats/test_base_feats.pkl')
train_dir = 'feature_store/train'
test_dir = 'feature_store/test'
# Usage
seed = 42
n_repeats = 5
n_splits = 10
target_col = 'score'

results = compare_with_baseline(base_dir=FEATURE_STORE, 
                                base_train_feats=base_train_feats,
                                base_test_feats=base_test_feats,
                                baseline_metrics=0.6242,
                                train_scores=train_scores)
print(results.head(13))
results.to_csv('feature_results.csv', index=False)

def generate_feature_combinations(feature_sets, max_combination_length):
    all_combinations = []
    for r in range(2, max_combination_length + 1):
        all_combinations.extend(combinations(feature_sets, r))
    return all_combinations

def compare_feature_combinations(base_dir, base_train_feats, base_test_feats, baseline_metrics, train_scores, seed=42, n_repeats=5, n_splits=10, max_combination_length=4):
    results = []
    feature_sets = get_feature_sets_from_folder(os.path.join(base_dir, 'train'))
    feature_combinations = generate_feature_combinations(feature_sets, max_combination_length)

    for combo in feature_combinations:
        combo_train_feats = None
        combo_test_feats = None

        for feature_set in combo:
            train_feats = load_feature_set(base_dir, feature_set, is_train=True)
            test_feats = load_feature_set(base_dir, feature_set, is_train=False)

            if train_feats is None or test_feats is None:
                print(f"Skipping combination {combo} due to missing data for {feature_set}.")
                continue

            if combo_train_feats is None:
                combo_train_feats = train_feats
                combo_test_feats = test_feats
            else:
                combo_train_feats = combo_train_feats.merge(train_feats, on=['id'], how='left')
                combo_test_feats = combo_test_feats.merge(test_feats, on=['id'], how='left')

        if combo_train_feats is not None and combo_test_feats is not None:
            combo_train_feats = combo_train_feats.merge(train_scores, on=['id'], how='left')
            combo_train_feats = combo_train_feats.merge(base_train_feats, on=['id'], how='left')
            combo_test_feats = combo_test_feats.merge(base_test_feats, on=['id'], how='left')

            oof_results, rmse = cv_pipeline(combo_train_feats, combo_test_feats, seed, n_repeats, n_splits)
            improvement = baseline_metrics - rmse
            results.append({'Feature Combination': combo, 'Metric': rmse, 'Improvement': improvement})
            print(f'Features: {combo}. RMSE: {rmse:.6f}, Improvement: {improvement:.6f}')

    return pd.DataFrame(results)

# Usage
max_combination_length = 4 # You can adjust this to test different combination lengths
results = compare_feature_combinations(base_dir=FEATURE_STORE, 
                                       base_train_feats=base_train_feats,
                                       base_test_feats=base_test_feats,
                                       baseline_metrics=0.6242,
                                       train_scores=train_scores,
                                       max_combination_length=max_combination_length)

print(results.head(5))
results.to_csv('feature_results_comb.csv', index=False)
