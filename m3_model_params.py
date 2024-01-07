import numpy as np

lgb_params_1 = {
    'boosting_type': 'gbdt', 
    'metric': 'rmse',
    'reg_alpha': 0.003188447814669599, 
    'reg_lambda': 0.0010228604507564066, 
    'colsample_bytree': 0.8,  
    'subsample_freq': 1,  
    'subsample': 0.75,  
    'learning_rate': 0.01716485155812008, 
    'num_leaves': 19, 
    'min_child_samples': 46,
    'n_estimators': 2000,
    'force_col_wise': True,
    'verbosity': -1,
    }

new_params = {'boosting_type': 'gbdt', 
              'colsample_bytree': 1.0, 
              'importance_type': 'split', 
              'learning_rate': 0.0183, 
              'max_depth': 6, 'min_child_samples': 49, 
              'min_child_weight': 0.001, 
              'min_split_gain': 0.0, 
              'n_estimators': 338, 
              'num_leaves': 69, 
              'reg_alpha': 0.6073, 
              'reg_lambda': 0.3127, 
              'subsample': 1.0, 
              'subsample_for_bin': 200000, 
              'subsample_freq': 0, 
              'verbose': -1}

xgb_params = {
    'alpha': 1,
    'colsample_bytree': 0.8,
    'gamma': 1.5,
    'learning_rate': 0.05,
    'max_depth': 4,
    'min_child_weight': 10,
    'subsample': 0.8,
    'device': 'cuda',
    'n_estimators': 1024 
    }

# FROM KAGGLE
xgb_param={
'reg_alpha': 0.00087,
'reg_lambda': 2.5428,
'colsample_bynode': 0.78390,
'subsample': 0.89942, 
'eta': 0.04730, 
'max_depth': 3, 
'n_estimators': 1024,
'eval_metric': 'rmse'
}