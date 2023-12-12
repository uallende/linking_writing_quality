import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from m3_model_params import xgb_params

def train_valid_split(data_x, data_y, train_idx, valid_idx):
    x_train = data_x.iloc[train_idx]
    y_train = data_y[train_idx]
    x_valid = data_x.iloc[valid_idx]
    y_valid = data_y[valid_idx]
    return x_train, y_train, x_valid, y_valid

def calculate_rmse(y, yhat):
    return mean_squared_error(y, yhat, squared=False)

def lgb_pipeline(train, test, n_splits=10, iterations=5):
        
    param = {'n_estimators': 1024,
    'learning_rate': 0.005,
    'metric': 'rmse',
    'force_col_wise': True,
    'verbosity': 0}

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
    print(f'RMSE by fold {np.mean(cv_rmse):.6f}. Std {np.std(cv_rmse)}')
    return test_preds, valid_preds, final_rmse, cv_rmse 


def xgb_pipeline(train, test, n_splits=10, iterations=5):

    x = train.drop(['id', 'score'], axis=1)
    y = train['score'].values
    test_x = test.drop(columns = ['id'])
    
    test_preds = []
    valid_preds = pd.DataFrame()

    for iter in range(iterations):
        model = xgb.XGBRegressor(**xgb_params).set_params(early_stopping_rounds=250, random_state = 42+iter)
        skf = StratifiedKFold(n_splits=n_splits, random_state=42+iter, shuffle=True)

        for i, (train_index, valid_index) in enumerate(skf.split(x, y.astype(str))):
            train_x, train_y, valid_x, valid_y = train_valid_split(x, y, train_index, valid_index)
            
            model.fit(
                train_x, train_y, 
                eval_set=[(valid_x, valid_y)],
                verbose=False
                )

            print(model.best_iteration)
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
    return test_preds, valid_preds, final_rmse, cv_rmse 