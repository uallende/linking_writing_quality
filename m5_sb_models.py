import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from itertools import combinations

def train_valid_split(data_x, data_y, train_idx, valid_idx):
    x_train = data_x.iloc[train_idx]
    y_train = data_y[train_idx]
    x_valid = data_x.iloc[valid_idx]
    y_valid = data_y[valid_idx]
    return x_train, y_train, x_valid, y_valid

def calculate_rmse(y, yhat):
    return mean_squared_error(y, yhat, squared=False)

def lgb_pipeline(train, test, param, n_splits=10, iterations=5):
        
    x = train.drop(['id', 'score'], axis=1)
    y = train['score'].values
    test_x = test.drop(columns = ['id'])
 
    test_preds = []
    valid_preds = pd.DataFrame()

    for iter in range(iterations):

        skf = StratifiedKFold(n_splits=n_splits, random_state=42+iter, shuffle=True)
        model = LGBMRegressor(**param, random_state = 42 + iter)

        for i, (train_index, valid_index) in enumerate(skf.split(x, y.astype(str))):
            train_x, train_y, valid_x, valid_y = train_valid_split(x, y, train_index, valid_index)
            
            model.fit(train_x, train_y)
            valid_predictions = model.predict(valid_x)
            test_predictions = model.predict(test_x)
            test_preds.append(test_predictions)

            tmp_df = train.loc[valid_index][['id','score']]
            tmp_df['preds'] = valid_predictions
            tmp_df['iteration'] = i + 1
            valid_preds = pd.concat([valid_preds, tmp_df])

        final_rmse = mean_squared_error(valid_preds['score'], valid_preds['preds'], squared=False)
        final_std = np.std(valid_preds['preds'])
        cv_rmse = valid_preds.groupby(['iteration']).apply(lambda g: calculate_rmse(g['score'], g['preds']))

    print(f'Final RMSE over {n_splits * iterations}: {final_rmse:.6f}. Std {final_std:.4f}')
    print(f'RMSE by fold {np.mean(cv_rmse):.6f}. Std {np.std(cv_rmse):.4f}')
    return test_preds, valid_preds, final_rmse, model 

def lgb_pipeline_kfold(train, test, param, n_splits=10, iterations=5):
        
    x = train.drop(['id', 'score'], axis=1)
    y = train['score'].values
    test_x = test.drop(columns = ['id'])
 
    test_preds = []
    valid_preds = pd.DataFrame()

    for iter in range(iterations):

        skf = KFold(n_splits=n_splits, random_state=42+iter, shuffle=True)
        model = LGBMRegressor(**param, random_state = 42 + iter)

        for i, (train_index, valid_index) in enumerate(skf.split(x, y.astype(str))):
            train_x, train_y, valid_x, valid_y = train_valid_split(x, y, train_index, valid_index)
            
            model.fit(train_x, train_y)
            valid_predictions = model.predict(valid_x)
            test_predictions = model.predict(test_x)
            test_preds.append(test_predictions)

            tmp_df = train.loc[valid_index][['id','score']]
            tmp_df['preds'] = valid_predictions
            tmp_df['iteration'] = i + 1
            valid_preds = pd.concat([valid_preds, tmp_df])

        final_rmse = mean_squared_error(valid_preds['score'], valid_preds['preds'], squared=False)
        final_std = np.std(valid_preds['preds'])
        cv_rmse = valid_preds.groupby(['iteration']).apply(lambda g: calculate_rmse(g['score'], g['preds']))

    print(f'Final RMSE over {n_splits * iterations}: {final_rmse:.6f}. Std {final_std:.4f}')
    print(f'RMSE by fold {np.mean(cv_rmse):.6f}. Std {np.std(cv_rmse):.4f}')
    return test_preds, valid_preds, final_rmse, model 


def xgb_pipeline(train, test, param, n_splits=10, iterations=5):

    x = train.drop(['id', 'score'], axis=1)
    y = train['score'].values
    test_x = test.drop(columns = ['id'])
    
    test_preds = []
    valid_preds = pd.DataFrame()

    for iter in range(iterations):
        skf = StratifiedKFold(n_splits=n_splits, random_state=42+iter, shuffle=True)
        model = xgb.XGBRegressor(**param, random_state = 42+ iter) # early_stopping_rounds=250, 

        for i, (train_index, valid_index) in enumerate(skf.split(x, y.astype(str))):
            train_x, train_y, valid_x, valid_y = train_valid_split(x, y, train_index, valid_index)
            
            model.fit(
                train_x, train_y, 
                eval_set=[(valid_x, valid_y)],
                verbose=False
                )

            #print(model.best_iteration)
            valid_predictions = model.predict(valid_x)
            test_predictions = model.predict(test_x)
            test_preds.append(test_predictions)

            tmp_df = train.loc[valid_index][['id','score']]
            tmp_df['preds'] = valid_predictions
            tmp_df['iteration'] = i + 1
            valid_preds = pd.concat([valid_preds, tmp_df])

        final_rmse = mean_squared_error(valid_preds['score'], valid_preds['preds'], squared=False)
        final_std = np.std(valid_preds['preds'])
        cv_rmse = valid_preds.groupby(['iteration']).apply(lambda g: calculate_rmse(g['score'], g['preds']))

    print(f'Final RMSE over {n_splits * iterations}: {final_rmse:.6f}. Std {final_std:.4f}')
    print(f'RMSE by fold {np.mean(cv_rmse):.6f}. Std {np.std(cv_rmse)}')
    return test_preds, valid_preds, final_rmse, model 

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

import os, gc

def compare_with_baseline(base_dir, base_train_feats, base_test_feats, params, baseline_metrics):
    results = []
    feature_sets = get_feature_sets_from_folder(os.path.join(base_dir, 'train'))
    print(feature_sets)

    for feature_set in feature_sets:
        print(f'Set of features to test: {feature_set}')
        train_feats = load_feature_set(base_dir, feature_set, is_train=True)
        test_feats = load_feature_set(base_dir, feature_set, is_train=False)

        if train_feats is not None and test_feats is not None:
            train_feats = train_feats.merge(base_train_feats, on=['id'], how='left')
            print(train_feats.shape)
            test_feats = test_feats.merge(base_test_feats, on=['id'], how='left')

            _, oof_results, rmse, _ = lgb_pipeline(train_feats, test_feats, params)
            improvement = baseline_metrics - rmse
            results.append({'Feature Set': feature_set, 'Metric': rmse, 'Improvement': improvement})
            print(f'Features: {feature_set}. RMSE: {rmse:.6f}, Improvement: {improvement:.6f}')

            # Cleanup
            del train_feats, test_feats, oof_results
            gc.collect()
        else:
            print(f"Skipping feature set {feature_set} due to missing data.")
    return pd.DataFrame(results)

def compare_feature_combinations(base_dir, base_train_feats, base_test_feats, params, baseline_metrics, max_combination_length=8, min_combination_length=3):
    results = []
    feature_sets = get_feature_sets_from_folder(os.path.join(base_dir, 'train'))
    feature_combinations = generate_feature_combinations(feature_sets, max_combination_length, min_combination_length)
    print(f'Number of combinations: {len(feature_combinations)}')

    for combo in feature_combinations:
        print(f'Feature set: {combo}')

        train_feats = base_train_feats.copy()
        test_feats = base_test_feats.copy()
        print(f'Base train size: {base_train_feats.shape}')

        for feature_set in combo:
            train_feat_set = load_feature_set(base_dir, feature_set, is_train=True)
            test_feat_set = load_feature_set(base_dir, feature_set, is_train=False)

            train_feats = train_feats.merge(train_feat_set, on=['id'], how='left')
            test_feats = test_feats.merge(test_feat_set, on=['id'], how='left')

        print(train_feats.shape, test_feats.shape)

        _, oof_preds, rmse, _ = lgb_pipeline(train_feats, test_feats, params)
        improvement = baseline_metrics - rmse
        results.append({'Feature Combination': combo, 'Metric': rmse, 'Improvement': improvement})
        print(f'Features: {combo}. RMSE: {rmse:.6f}, Improvement: {improvement:.6f}')

    return pd.DataFrame(results)

def generate_feature_combinations(feature_sets, max_length, min_length=2):
    all_combinations = []
    for r in range(min_length, max_length + 1):
        for combo in combinations(feature_sets, r):
            all_combinations.append(combo)
    return all_combinations