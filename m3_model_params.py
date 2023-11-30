lgb_params_1 = {
    'boosting_type': 'gbdt', 
    'metric': 'rmse',
    'reg_alpha': 0.003188447814669599, 
    'reg_lambda': 0.0010228604507564066, 
    'feature_fraction': 0.8,
    'bagging_freq': 1,
    'bagging_fraction': 0.75,
    'learning_rate': 0.01716485155812008, 
    'num_leaves': 19, 
    'min_child_samples': 46,
    'verbosity': -1,
    'n_estimators': 2000
    }

lgb_params_2 = {'boosting_type': 'gbdt', 
                'num_leaves': 16, 
                'learning_rate': 0.15347777609888505, 
                'max_depth': 11, 
                'min_child_samples': 49, 
                'reg_alpha': 0.011839618542937691, 
                'reg_lambda': 0.8759242646774177,
                'objective': 'regression',
                'n_estimators': 2000}


lgb_params_3 = {'boosting_type': 'gbdt', 
                'num_leaves': 16, 
                'learning_rate': 0.17129057152457167, 
                'max_depth': 6, 
                'min_child_samples': 31, 
                'reg_alpha': 0.9620195102280669, 
                'reg_lambda': 0.5531047842086327,
                'objective': 'regression',
                'n_estimators': 2000}

xgb_params = {
    'alpha': 1,
    'colsample_bytree': 0.8,
    'gamma': 1.5,
    'learning_rate': 0.05,
    'max_depth': 4,
    'min_child_weight': 10,
    'subsample': 0.8,
    'device': 'cuda',
    'tree_method': 'hist',
    'n_estimators': 300 
    }