from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import torch, joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def map_class(x, task, reader):
    if task.name == 'multiclass':
        return reader[x]
    else:
        return x

mapped = np.vectorize(map_class)

def score(task, y_true, y_pred):
    if task.name == 'reg' or task.name == 'multi:reg':
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
    ITERATIONS = 10
    TRAIN_BS = [128,256,312,396,512]  
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