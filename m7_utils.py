import os, gc
from itertools import combinations
from m5_models import *

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

def compare_with_baseline(base_dir, base_train_feats, base_test_feats, params, baseline_metrics, train_scores, seed=42, n_repeats=5, n_splits=10):
    results = []
    feature_sets = get_feature_sets_from_folder(os.path.join(base_dir, 'train'))
    print(feature_sets)

    for feature_set in feature_sets:
        print(feature_set)
        train_feats = load_feature_set(base_dir, feature_set, is_train=True)
        test_feats = load_feature_set(base_dir, feature_set, is_train=False)

        if train_feats is not None and test_feats is not None:
            train_feats = train_feats.merge(train_scores, on=['id'], how='left')
            train_feats = train_feats.merge(base_train_feats, on=['id'], how='left')
            test_feats = test_feats.merge(base_test_feats, on=['id'], how='left')

            _, oof_results, rmse, _ = cv_pipeline(train_feats, test_feats, params, seed, n_repeats, n_splits)
            improvement = baseline_metrics - rmse
            results.append({'Feature Set': feature_set, 'Metric': rmse, 'Improvement': improvement})
            print(f'Features: {feature_set}. RMSE: {rmse:.6f}, Improvement: {improvement:.6f}')

            # Cleanup
            del train_feats, test_feats, oof_results
            gc.collect()
        else:
            print(f"Skipping feature set {feature_set} due to missing data.")
    return pd.DataFrame(results)

def create_specific_balanced_datasets(df, target_col, scores_to_split, num_datasets):
    # Group by the target column and collect IDs
    grouped = df.groupby(target_col)['id'].apply(list)

    # Initialize dataset splits
    dataset_splits = {i: [] for i in range(num_datasets)}

    # Splitting IDs for specified scores into num_datasets parts
    for score, ids in grouped.items():
        if score in scores_to_split:
            np.random.shuffle(ids)  # Shuffle for randomness
            split_sizes = np.full(num_datasets, len(ids) // (num_datasets-0.5), dtype=int)
            split_sizes[:len(ids) % num_datasets] += 1
            start = 0
            for i, size in enumerate(split_sizes):
                dataset_splits[i].extend(ids[start:start + size])
                start += size
        else:
            # For other scores, add all IDs to each split
            for i in range(num_datasets):
                dataset_splits[i].extend(ids)

    # Create balanced datasets
    balanced_datasets = []
    for i in range(num_datasets):
        dataset_ids = dataset_splits[i]
        balanced_dataset = df[df['id'].isin(dataset_ids)]
        balanced_datasets.append(balanced_dataset)

    return balanced_datasets

def generate_feature_combinations(feature_sets, max_combination_length):
    all_combinations = []
    for r in range(max_combination_length, 0, -1):
        all_combinations.extend(combinations(feature_sets, r))
    return all_combinations

def compare_feature_combinations(base_dir, base_train_feats, base_test_feats, params, baseline_metrics, seed=42, n_repeats=5, n_splits=10, max_combination_length=7):
    results = []
    feature_sets = get_feature_sets_from_folder(os.path.join(base_dir, 'train'))
    feature_combinations = generate_feature_combinations(feature_sets, max_combination_length)
    print(f'Number of combinations: {len(feature_combinations)}')
    # feature_combinations = [('IKI', 'rep_cut')]

    for combo in feature_combinations:
        print(f'Feature set: {combo}')

        # Reset train_feats and test_feats to base features at the beginning of each combination
        train_feats = base_train_feats.copy()
        test_feats = base_test_feats.copy()
        print(f'Base train size: {base_train_feats.shape}')

        for feature_set in combo:
            train_feat_set = load_feature_set(base_dir, feature_set, is_train=True)
            test_feat_set = load_feature_set(base_dir, feature_set, is_train=False)

            train_feats = train_feats.merge(train_feat_set, on=['id'], how='left')
            test_feats = test_feats.merge(test_feat_set, on=['id'], how='left')

        target_col = ['score']
        drop_cols = ['id']
        train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]

        missing_cols = [col for col in train_cols if col not in test_feats.columns]
        missing_cols_df = pd.DataFrame({col: np.nan for col in missing_cols}, index=test_feats.index)
        test_feats = pd.concat([test_feats, missing_cols_df], axis=1)

        train_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
        test_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(train_feats.shape, test_feats.shape)
        # avg_pred, rmse = run_lgb_cv(train_feats, test_feats, train_cols, target_col, lgb_params_1, seed, n_repeats, n_splits) RUN FOR DEBUGGING
        _, oof_preds, rmse, _ = cv_pipeline(train_feats, test_feats, params, seed, n_repeats, n_splits)
        improvement = baseline_metrics - rmse
        results.append({'Feature Combination': combo, 'Metric': rmse, 'Improvement': improvement})
        print(f'Features: {combo}. RMSE: {rmse:.6f}, Improvement: {improvement:.6f}')

    return pd.DataFrame(results)

def create_specific_balanced_datasets(train_scores, scores_to_split=[3, 3.5, 4, 4.5], pct_to_remv=0.1, n_datasets=2, seed=42):

    balanced_scores = []
    shuffled_scores = train_scores.copy()
    for i in range(n_datasets):
        shuffled_scores = shuffled_scores.sample(frac=1, random_state=seed + i)
        scores_to_keep = [score for score in shuffled_scores.score.unique() if score not in scores_to_split]
        ix_to_keep = []
        ix_to_keep.append(shuffled_scores[shuffled_scores['score'].isin(scores_to_keep)].index.values)

        for score in [3, 3.5, 4, 4.5]:
            tpm_scores = shuffled_scores[shuffled_scores['score']==score]
            rows_to_rmv = int(len(tpm_scores) * pct_to_remv) # calcualte rows to remove
            keep_ix = tpm_scores.index.values[rows_to_rmv:] # df is shuffled - removing first chunk
            ix_to_keep.append(keep_ix)

        ix_to_keep = [j for i in ix_to_keep for j in i]
        temp_scores = shuffled_scores.loc[ix_to_keep].copy()
        balanced_scores.append(temp_scores)
    
    return balanced_scores

def load_feature_set_comb(base_dir, feature_type, is_train=True):

    file_path = os.path.join(base_dir, 'train_' + feature_type + '.pkl')

    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}")
        return None

    return pd.read_pickle(file_path)

def compare_with_baseline_comb(base_dir, base_feats, train_ids, test_ids, params, baseline_metrics, train_scores, seed=42, n_repeats=5, n_splits=10):
    results = []
    feature_sets = get_feature_sets_from_folder(os.path.join(base_dir))
    print(feature_sets)

    for feature_set in feature_sets:
        print(feature_set)
        new_feats = load_feature_set_comb(base_dir, feature_set, is_train=True)
        feats = base_feats.merge(new_feats, on='id', how='left')
        feats = preprocess_feats(feats)
        train_feats = feats[feats['id'].isin(train_ids)]
        test_feats = feats[feats['id'].isin(test_ids)]

        if train_feats is not None:
            train_feats = train_feats.merge(train_scores, on=['id'], how='left')
            _, oof_results, rmse, _ = cv_pipeline(train_feats, test_feats, params, seed, n_repeats, n_splits)
            improvement = baseline_metrics - rmse
            results.append({'Feature Set': feature_set, 'Metric': rmse, 'Improvement': improvement})
            print(f'Features: {feature_set}. RMSE: {rmse:.6f}, Improvement: {improvement:.6f}')

            # Cleanup
            del train_feats, oof_results
            gc.collect()
        else:
            print(f"Skipping feature set {feature_set} due to missing data.")
    return pd.DataFrame(results)

def preprocess_feats(feats, scaler):
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    feats.fillna(-1e10, inplace=True)
    feats_columns = feats.columns
    feats.loc[:, feats_columns != 'id'] = scaler.fit_transform(feats.loc[:, feats_columns != 'id'])
    return feats