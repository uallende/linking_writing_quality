import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm  
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge


def calculate_rmse(df):
    return mean_squared_error(df['score'], df['prediction'], squared=False)

def run_lgb_cv(train_feats, test_feats, train_cols, target_col, lgb_params, boosting_type, seed, n_repeats, n_splits):

    oof_results = pd.DataFrame(columns = ['id', 'iteration', 'score', 'prediction'])

    X = train_feats[train_cols]
    y = train_feats[target_col]
    X_test = test_feats[train_cols]
    test_preds = []

    for i in range(n_repeats):
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + i)

        for train_idx, valid_idx in skf.split(train_feats, y):
            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]

            model = lgb.LGBMRegressor(**lgb_params, verbose=-1, random_state=seed)
            if boosting_type != 'dart':
                model.fit(X_train, y_train, 
                        eval_set=[(X_valid, y_valid)], 
                        callbacks=[lgb.early_stopping(250, first_metric_only=True, verbose=False)])
            else:
                model.fit(X_train, y_train)  # No early stopping for DART

            valid_predictions = model.predict(X_valid, num_iteration=model.best_iteration_)
            test_predictions = model.predict(X_test, num_iteration=model.best_iteration_)
            test_preds.append(test_predictions)
        
            tmp_df = train_feats.loc[valid_idx][['id','score']]
            tmp_df['prediction'] = valid_predictions
            tmp_df['iteration'] = i + 1
            oof_results = pd.concat([oof_results, tmp_df])



                

    avg_preds = oof_results.groupby(['id','score'])['prediction'].mean().reset_index()
    rmse = mean_squared_error(avg_preds['score'], avg_preds['prediction'], squared=False)
    print(f"LGBM Average RMSE over {n_repeats * n_splits} folds: {rmse:.6f}")
    return test_preds, oof_results, rmse, model  

def cv_pipeline(train_feats, test_feats, lgb_params, boosting_type, seed=42, n_repeats=5, n_splits=10):

    target_col = ['score']
    drop_cols = ['id']
    train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]

    missing_cols = [col for col in train_cols if col not in test_feats.columns]
    missing_cols_df = pd.DataFrame({col: np.nan for col in missing_cols}, index=test_feats.index)
    test_feats = pd.concat([test_feats, missing_cols_df], axis=1)


    test_preds, oof_preds, rmse, model = run_lgb_cv(train_feats=train_feats, test_feats=test_feats, 
                                             train_cols=train_cols, target_col=target_col, 
                                             lgb_params=lgb_params, boosting_type=boosting_type,
                                             seed=seed, n_repeats=n_repeats, n_splits=n_splits)
    
    rmse_per_iteration = oof_preds.groupby('iteration').apply(calculate_rmse)
    print(f'Mean RMSE of all iterations: {np.mean(rmse_per_iteration):.6f}')

    return test_preds, oof_preds, rmse, model

def cv_pipeline(train_feats, test_feats, lgb_params, boosting_type, seed=42, n_repeats=5, n_splits=10):

    target_col = ['score']
    drop_cols = ['id']
    train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]

    missing_cols = [col for col in train_cols if col not in test_feats.columns]
    missing_cols_df = pd.DataFrame({col: np.nan for col in missing_cols}, index=test_feats.index)
    test_feats = pd.concat([test_feats, missing_cols_df], axis=1)


    test_preds, oof_preds, rmse, model = run_lgb_cv(train_feats=train_feats, test_feats=test_feats, 
                                             train_cols=train_cols, target_col=target_col, 
                                             lgb_params=lgb_params, boosting_type=boosting_type,
                                             seed=seed, n_repeats=n_repeats, n_splits=n_splits)
    
    rmse_per_iteration = oof_preds.groupby('iteration').apply(calculate_rmse)
    print(f'Mean RMSE of all iterations: {np.mean(rmse_per_iteration):.6f}')
    

    return test_preds, oof_preds, rmse, model

def run_lgb_cv_for_balanced_set(train_feats, test_feats, 
                                train_cols, target_col, 
                                lgb_params, balanced_dataset_ids, boosting_type,
                                seed=42, n_repeats=5, n_splits=10):

    oof_results = pd.DataFrame(columns = ['id', 'score', 'prediction'])
    binned_y = np.digitize(train_feats[target_col], bins=sorted(train_feats[target_col].value_counts()))

    X = train_feats[train_cols]
    y = train_feats[target_col]
    X_test = test_feats[train_cols]
    test_preds = []

    for i in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + i)

        for train_idx, valid_idx in skf.split(train_feats, binned_y):
            
            filtered_train_idx = train_idx[train_feats.loc[train_idx, 'id'].isin(balanced_dataset_ids)]

            X_train, y_train = X.loc[filtered_train_idx], y.loc[filtered_train_idx]
            X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]

            model = lgb.LGBMRegressor(**lgb_params, verbose=-1, random_state=seed)
            if boosting_type != 'dart':
                model.fit(X_train, y_train, 
                        eval_set=[(X_valid, y_valid)], 
                        callbacks=[lgb.early_stopping(250, first_metric_only=True, verbose=False)])
            else:
                model.fit(X_train, y_train)  # No early stopping for DART

            valid_predictions = model.predict(X_valid, num_iteration=model.best_iteration_)
            test_predictions = model.predict(X_test, num_iteration=model.best_iteration_)
            test_preds.append(test_predictions)
        
            tmp_df = train_feats.loc[valid_idx][['id','score']]
            tmp_df['prediction'] = valid_predictions
            tmp_df['iteration'] = i + 1
            oof_results = pd.concat([oof_results, tmp_df])

    avg_preds = oof_results.groupby(['id', 'score'])['prediction'].mean().reset_index()
    rmse = mean_squared_error(avg_preds['score'], avg_preds['prediction'], squared=False)
    print(f"LGBM Average RMSE over {n_repeats * n_splits} folds: {rmse:.6f}")
    return test_preds, avg_preds, rmse


def cv_balanced_pipeline(train_feats, test_feats, lgb_params, 
                         balanced_dataset_ids, boosting_type, 
                         seed=42, n_repeats=5, n_splits=10):

    target_col = ['score']
    drop_cols = ['id']
    train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]

    missing_cols = [col for col in train_cols if col not in test_feats.columns]
    missing_cols_df = pd.DataFrame({col: np.nan for col in missing_cols}, index=test_feats.index)
    test_feats = pd.concat([test_feats, missing_cols_df], axis=1)

    train_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_feats.replace([np.inf, -np.inf], np.nan, inplace=True)

    test_preds, oof_preds, rmse = run_lgb_cv_for_balanced_set(train_feats=train_feats, test_feats=test_feats, 
                                             train_cols=train_cols, target_col=target_col, 
                                             lgb_params=lgb_params, balanced_dataset_ids=balanced_dataset_ids,
                                             boosting_type=boosting_type,
                                             seed=seed, n_repeats=n_repeats, n_splits=n_splits)
    
    return test_preds, oof_preds, rmse

def run_xgb_model(model, X_train, y_train, X_valid, y_valid, X_test):
    model.fit(
        X_train, y_train, 
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    valid_predictions = model.predict(X_valid)
    test_predictions = model.predict(X_test)
    return valid_predictions, test_predictions

def run_xgb_cv(train_feats, test_feats, train_cols, target_col, xgb_params, seed, n_repeats, n_splits):

    oof_results = pd.DataFrame(columns = ['id', 'score', 'prediction'])
    binned_y = np.digitize(train_feats[target_col], bins=sorted(train_feats[target_col].value_counts()))

    X = train_feats[train_cols]
    y = train_feats[target_col]
    X_test = test_feats[train_cols]

    test_preds_list = []

    for i in tqdm(range(n_repeats), desc="Iterations"):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + i)

        for train_idx, valid_idx in skf.split(train_feats, binned_y):
            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]

            model_xgb = xgb.XGBRegressor(**xgb_params).set_params(early_stopping_rounds=250, random_state = seed)
            valid_preds_xgb, test_preds_xgb = run_xgb_model(model_xgb, 
                                               X_train, y_train, 
                                               X_valid, y_valid, 
                                               X_test)

            test_preds_list.append(test_preds_xgb)

            tmp_df = train_feats.loc[valid_idx][['id','score']]
            tmp_df['prediction'] = valid_preds_xgb
            oof_results = pd.concat([oof_results, tmp_df])

    avg_preds = oof_results.groupby(['id','score'])['prediction'].mean().reset_index()
    rmse = mean_squared_error(avg_preds['score'], avg_preds['prediction'], squared=False)
    print(f"XGB Average RMSE over {n_repeats * n_splits} folds: {rmse:.6f}")
    return np.mean(test_preds_list, axis=0), avg_preds, rmse, model_xgb

def xgb_cv_pipeline(train_feats, test_feats, xgb_params, seed=42, n_repeats=5, n_splits=10):

    target_col = ['score']
    drop_cols = ['id']
    train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]

    missing_cols = [col for col in train_cols if col not in test_feats.columns]
    missing_cols_df = pd.DataFrame({col: np.nan for col in missing_cols}, index=test_feats.index)
    test_feats = pd.concat([test_feats, missing_cols_df], axis=1)

    train_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_feats.replace([np.inf, -np.inf], np.nan, inplace=True)

    test_preds, oof_preds, rmse, model = run_xgb_cv(train_feats=train_feats, test_feats=test_feats, 
                      train_cols=train_cols, target_col=target_col, 
                      xgb_params=xgb_params,
                      seed=seed, n_repeats=n_repeats, n_splits=n_splits)
    
    return test_preds, oof_preds, rmse, model

def final_processing(scores_lgb, scores_xgb, scores_blend, test_predict_list_lgb, test_predict_list_xgb, test_feats):
    print(f'OOF AVG LGBM: {np.mean(scores_lgb):.6f}, XGB: {np.mean(scores_xgb):.6f}, Blend: {np.mean(scores_blend):.6f}')
    final_pred_lgb = np.mean(test_predict_list_lgb, axis=0)
    final_pred_xgb = np.mean(test_predict_list_xgb, axis=0)
    final_pred_blend = 0.5 * (final_pred_lgb + final_pred_xgb)
    predictions_df = pd.DataFrame({'id': test_feats['id'], 'score': final_pred_blend})
    return final_pred_lgb, final_pred_xgb, final_pred_blend, predictions_df

def run_ridge_model(model, X_train, y_train, X_valid, X_test):
    model.fit(X_train, y_train)
    valid_predictions = model.predict(X_valid)
    test_predictions = model.predict(X_test)
    return valid_predictions, test_predictions

def run_ridge_cv(train_feats, test_feats, train_cols, target_col, ridge_params, seed, n_repeats, n_splits):
    oof_results = pd.DataFrame(columns=['id', 'score', 'prediction'])
    binned_y = np.digitize(train_feats[target_col], bins=sorted(train_feats[target_col].value_counts()))

    X = train_feats[train_cols]
    y = train_feats[target_col]
    X_test = test_feats[train_cols]

    models = []
    for i in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + i)

        for train_idx, valid_idx in skf.split(train_feats, binned_y):
            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]

            model_ridge = Ridge(**ridge_params, random_state=seed)
            valid_preds_ridge, test_preds_ridge = run_ridge_model(model=model_ridge,
                                                                  X_train=X_train, y_train=y_train,
                                                                  X_valid=X_valid, X_test=X_test)
            
            tmp_df = train_feats.loc[valid_idx][['id', 'score']]
            tmp_df['prediction'] = valid_preds_ridge
            oof_results = pd.concat([oof_results, tmp_df])

            models.append(model_ridge)

    oof_preds = oof_results.groupby(['id', 'score'])['prediction'].mean().reset_index()
    rmse = mean_squared_error(oof_preds['score'], oof_preds['prediction'], squared=False)
    print(f"Ridge Average RMSE over {n_repeats * n_splits} folds: {rmse:.6f}")

    return test_preds_ridge, models, oof_preds, rmse

def ridge_cv_pipeline(train_feats, test_feats, ridge_params, seed=42, n_repeats=5, n_splits=10):

    target_col = ['score']
    drop_cols = ['id']
    train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]

    missing_cols = [col for col in train_cols if col not in test_feats.columns]
    missing_cols_df = pd.DataFrame({col: np.nan for col in missing_cols}, index=test_feats.index)
    test_feats = pd.concat([test_feats, missing_cols_df], axis=1)

    train_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_feats.fillna(-10e6, inplace=True)
    test_feats.fillna(-10e6, inplace=True)

    test_preds_ridge, models, oof_preds, rmse = run_ridge_cv(train_feats=train_feats,
                                                      test_feats=test_feats,
                                                      train_cols=train_cols,
                                                      target_col=target_col,
                                                      ridge_params=ridge_params,
                                                      seed=seed,
                                                      n_repeats=n_repeats,
                                                      n_splits=n_splits)
    
    return test_preds_ridge, models, oof_preds, rmse