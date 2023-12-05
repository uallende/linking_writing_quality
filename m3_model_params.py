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
    'verbosity': -1,
    }

lgb_params_2 = {
    'boosting_type': 'gbdt', 
    'colsample_bytree': 1.0, 
    'importance_type': 'split', 
    'learning_rate': 0.17106535627270134, 
    'max_depth': 16, 
    'min_child_samples': 39, 
    'min_child_weight': 0.001, 
    'min_split_gain': 0.0, 
    'n_jobs': None, 
    'num_leaves': 15, 
    'reg_alpha': 0.8577521098353755, 
    'reg_lambda': 0.7679447672996995, 
    'subsample': 1.0, 
    'n_estimators': 2000,
    'subsample_for_bin': 200000, 
    'subsample_freq': 0
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
    'n_estimators': 2000 
    }

xgb_params_2 = {
    'objective': 'reg:squarederror', 
    'base_score': None, 
    'booster': 'gbtree', 
    'callbacks': None, 
    'colsample_bylevel': None, 
    'colsample_bynode': None, 
    'colsample_bytree': 0.8739265814988941, 
    'device': 'cuda', 
    'early_stopping_rounds': 250, 
    'enable_categorical': False, 
    'gamma': 0.3893408430829386, 
    'learning_rate': 0.18205557831997368, 
    'max_depth': 3, 
    'min_child_weight': 4, 
    'missing': np.nan, 
    'random_state': 42, 
    'subsample': 0.6407940935121966,
    'device': 'cuda',
    'tree_method': 'hist',
    'n_estimators': 2000 }


xgb_params_3 = {
    'boosting_type': 'goss', 
    'colsample_bytree': 1.0, 
    'importance_type': 'split', 
    'learning_rate': 0.09078219853571294, 
    'max_depth': 7, 
    'min_child_samples': 10, 
    'min_child_weight': 0.001, 
    'min_split_gain': 0.0, 
    'n_estimators': 2000, 
    'num_leaves': 15, 
    'random_state': 42, 
    'reg_alpha': 0.025677282656185296, 
    'reg_lambda': 0.1519356321376626, 
    'subsample': 1.0, 
    'subsample_for_bin': 200000, 
    'subsample_freq': 0, 
    'verbose': -1,
    'device': 'cuda',
    'tree_method': 'hist'}

non_important_feats = [
        'tok_21',
        'tok_20',
        'tok_14',
        'action_time_min',
        'tok_15',
        'tok_23',
        'tok_16',
        'tok_22',
        'tok_17',
        'tok_24',
        'tok_25',
        'tok_18',
        'tok_19',
        'up_event_12_count',
        'cursor_position_change1_quantile',
        'word_count_change3_quantile',
        'up_event_15_count',
        'up_event_10_count',
        'up_event_2_count',
        'down_event_15_count',
        'word_count_change5_quantile',
        'cursor_position_change5_quantile',
        'word_count_change3_max',
        'essay_words_min',
        'cursor_position_change3_quantile',
        'word_count_change2_quantile',
        'word_count_change2_max',
        'cursor_position_change2_quantile',
        'essay_words_median',
        'word_count_change1_quantile',
        'word_count_change1_max',
        'tok_26'
]