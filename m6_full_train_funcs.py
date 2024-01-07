
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib, torch

from sklearn import svm
from lightgbm import LGBMRegressor
from sklearn import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


def lgb_full_train_set(train, test, param, iterations=50):
        
    x = train.drop(['id', 'score'], axis=1)
    y = train['score'].values
    test_x = test.drop(columns = ['id'])
 
    test_preds = pd.DataFrame()

    for iter in range(iterations):

        model = LGBMRegressor(**param, random_state = 42 + iter)
        model.fit(x, y)
        test_predictions = model.predict(test_x)

        test_tmp = test[['id']].copy()
        test_tmp['score'] = test_predictions
        test_preds = pd.concat([test_preds, test_tmp], axis=0)

    return test_preds

def xgb_full_train_set(train, test, param, iterations=50):
        
    x = train.drop(['id', 'score'], axis=1)
    y = train['score'].values
    test_x = test.drop(columns = ['id'])
 
    test_preds = pd.DataFrame()

    for iter in range(iterations):

        model = xgb.XGBRegressor(**param, random_state = 42+iter, verbosity=0) 
        model.fit(x, y)
        test_predictions = model.predict(test_x)

        test_tmp = test[['id']].copy()
        test_tmp['score'] = test_predictions
        test_preds = pd.concat([test_preds, test_tmp], axis=0)

    return test_preds

def catboost_full_train_set(train, test, param, iterations=50):
        
    x = train.drop(['id', 'score'], axis=1)
    y = train['score'].values
    test_x = test.drop(columns = ['id'])
 
    test_preds = pd.DataFrame()

    for iter in range(iterations):

        model = cb.CatBoostRegressor(**param, random_state = 42 + iter)
        model.fit(x, y)
        test_predictions = model.predict(test_x)

        test_tmp = test[['id']].copy()
        test_tmp['score'] = test_predictions
        test_preds = pd.concat([test_preds, test_tmp], axis=0)

    return test_preds

def svr_pipeline(train, test, iterations=50):
        
    def preprocess_feats(feats, scaler=StandardScaler()):
        feats = np.where(np.isinf(feats), np.nan, feats)
        feats = np.nan_to_num(feats, nan=-1e8)
        return scaler.fit_transform(feats)
    
    tr_ids = train.id
    ts_ids = test.id
    tr_cols = train.columns[~train.columns.isin(['id','score'])]

    feats = pd.concat([train, test], axis=0)
    feats.loc[:,tr_cols] = preprocess_feats(feats.loc[:,tr_cols])

    train_feats = feats[feats['id'].isin(tr_ids)]
    test_feats = feats[feats['id'].isin(ts_ids)]

    test_x = test_feats.drop(columns = ['id', 'score'])
    test_x = preprocess_feats(test_x)

    x = train_feats.drop(['id', 'score'], axis=1)
    y = train_feats['score']

    test_preds = pd.DataFrame()

    for iter in range(iterations):

            model = svm.SVR(kernel='rbf', C=1.0, epsilon=0.1)
            model.fit(x, y)

            test_predictions = model.predict(test_x)

            test_tmp = test[['id']].copy()
            test_tmp['score'] = test_predictions
            test_preds = pd.concat([test_preds, test_tmp], axis=0)
    
    return test_preds

def ridge_pipeline(train, test, param, iterations=50):
        
    def preprocess_feats(feats, scaler=StandardScaler()):
        feats = np.where(np.isinf(feats), np.nan, feats)
        feats = np.nan_to_num(feats, nan=-1e8)
        return scaler.fit_transform(feats)
    
    tr_ids = train.id
    ts_ids = test.id
    tr_cols = train.columns[~train.columns.isin(['id','score'])]

    feats = pd.concat([train, test], axis=0)
    feats.loc[:,tr_cols] = preprocess_feats(feats.loc[:,tr_cols])

    train_feats = feats[feats['id'].isin(tr_ids)]
    test_feats = feats[feats['id'].isin(ts_ids)]

    test_x = test_feats.drop(columns = ['id', 'score'])
    test_x = preprocess_feats(test_x)

    x = train_feats.drop(['id', 'score'], axis=1)
    y = train_feats['score']

    test_preds = pd.DataFrame()

    for iter in range(iterations):

            model = Ridge(**param, random_state=42 + iter)
            model.fit(x, y)

            test_predictions = model.predict(test_x)

            test_tmp = test[['id']].copy()
            test_tmp['score'] = test_predictions
            test_preds = pd.concat([test_preds, test_tmp], axis=0)
    
    return test_preds

def map_class(x, task, reader):
    if task.name == 'multiclass':
        return reader[x]
    else:
        return x

mapped = np.vectorize(map_class)

def score(task, y_true, y_pred):
    if task.name == 'binary':
        return roc_auc_score(y_true, y_pred)
    elif task.name == 'multiclass':
        return log_loss(y_true, y_pred)
    elif task.name == 'reg' or task.name == 'multi:reg':
        return mean_squared_error(y_true, y_pred, squared=False)
    else:
        raise 'Task is not correct.'
        
def take_pred_from_task(pred, task):
    if task.name == 'binary' or task.name == 'reg':
        return pred[:, 0]
    elif task.name == 'multiclass' or task.name == 'multi:reg':
        return pred
    else:
        raise 'Task is not correct.'
        
def use_plr(USE_PLR):
    if USE_PLR:
        return "plr"
    else:
        return "cont"
    
def automl_pipeline(train_feats, test_feats):

    valid_preds = pd.DataFrame()
    test_preds = pd.DataFrame()
    ITERATIONS = 1
    TRAIN_BS = [128,256]  
    RANDOM_STATE = 42
    N_THREADS = 2
    N_FOLDS = 10
    TIMEOUT = 10000
    ADVANCED_ROLES = False
    USE_QNT = True
    TASK = 'reg'
    TARGET_NAME = 'score'

    for i in range(ITERATIONS):
        for b in TRAIN_BS:
                
            np.random.seed(RANDOM_STATE+i)
            torch.set_num_threads(N_THREADS)
            task = Task(TASK)

            roles = {
                'target': TARGET_NAME,
                'drop': ['id']
            }
            algo = 'denselight'
            automl = TabularAutoML(
                task = task, 
                timeout = TIMEOUT,
                cpu_limit = N_THREADS,
                general_params = {"use_algos": [[algo]]},
                nn_params = {
                    "n_epochs": 350, 
                    "bs": b, 
                    "num_workers": 0, 
                    "path_to_save": None, 
                    "freeze_defaults": True,
                },
                nn_pipeline_params = {"use_qnt": USE_QNT, "use_te": False},
                reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE+i, 'advanced_roles': ADVANCED_ROLES},
            )

            valid_pred = automl.fit_predict(train_feats, roles = roles, verbose = 0)
            valid_tpm = train_feats[['id','score']].copy()
            valid_tpm['preds'] = valid_pred.data.ravel()
            valid_preds = pd.concat([valid_preds, valid_tpm], ignore_index=True)
            joblib.dump(automl, f'automl_model_{b}_{i}.joblib')       


            test_pred = automl.predict(test_feats)    
            test_tmp = test_feats[['id']].copy()
            test_tmp['score'] = test_pred.data.ravel()
            test_preds = pd.concat([test_preds, test_tmp], ignore_index=True)

    final_rmse = mean_squared_error(valid_preds['score'], valid_preds['preds'], squared=False)
    final_std = np.std(valid_preds['preds'])

    valid_preds = valid_preds.groupby(['id','score'])['preds'].mean().reset_index()
    valid_preds = valid_preds.sort_values('id')

    test_preds = test_preds.groupby(['id'])['score'].mean().reset_index()
    test_preds = test_preds.sort_values('id')    

    return valid_preds, test_preds, final_rmse