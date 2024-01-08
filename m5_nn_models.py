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

# def TabNet_pipeline(train_feats, test_feats, params):

#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # DEVICE = 'cpu'

#     def train_valid_split(data_x, data_y, train_idx, valid_idx):
#         x_train = data_x.loc[train_idx].values
#         y_train = data_y[train_idx].values.reshape(-1,1)
#         x_valid = data_x.loc[valid_idx].values
#         y_valid = data_y[valid_idx].values.reshape(-1,1)
#         return x_train, y_train, x_valid, y_valid

#     def preprocess_feats(feats, scaler=StandardScaler()):
#         # Replace inf/-inf with NaN and then fill NaNs with a large negative number
#         feats = np.where(np.isinf(feats), np.nan, feats)
#         feats = np.nan_to_num(feats, nan=-1e6)
#         return scaler.fit_transform(feats)
    
#     tr_ids = train_feats.id
#     ts_ids = test_feats.id
#     tr_cols = train_feats.columns[~train_feats.columns.isin(['id','score'])]

#     feats = pd.concat([train_feats, test_feats], axis=0)
#     feats.loc[:,tr_cols] = preprocess_feats(feats.loc[:,tr_cols])

#     train_feats = feats[feats['id'].isin(tr_ids)]
#     test_feats = feats[feats['id'].isin(ts_ids)]

#     x = train_feats.drop(['id', 'score'], axis=1)
#     y = train_feats['score']

#     test_x = test_feats.drop(columns = ['id', 'score'])
#     test_x = preprocess_feats(test_x)

#     test_preds = []
#     valid_preds = pd.DataFrame()

#     for iter in range(1):
#         skf = StratifiedKFold(n_splits=10, random_state=42+iter, shuffle=True)
#         for i, (train_index, valid_index) in enumerate(skf.split(x, y.astype(str))):

#             model = TabNetRegressor(**params, device_name=DEVICE, verbose=0)
#             train_x, train_y, valid_x, valid_y = train_valid_split(x, y, train_index, valid_index)

#             model.fit(
#                 X_train = train_x, y_train = train_y,
#                 eval_set = [(train_x, train_y), (valid_x, valid_y)],
#                 max_epochs = 1000,
#                 patience = 70,
#             )

#             valid_predictions = model.predict(valid_x)
#             test_predictions = model.predict(test_x)
#             test_preds.append(test_predictions)

#             tmp_df = train_feats.loc[valid_index][['id','score']]
#             tmp_df['preds'] = valid_predictions
#             tmp_df['iteration'] = i + 1
#             valid_preds = pd.concat([valid_preds, tmp_df])

#             torch.cuda.empty_cache()
#             del train_x, train_y, valid_x, valid_y

#     final_rmse = mean_squared_error(valid_preds['score'], valid_preds['preds'], squared=False)
#     final_std = np.std(valid_preds['preds'])
#     cv_rmse = valid_preds.groupby(['iteration']).apply(lambda g: calculate_rmse(g['score'], g['preds']))
#     # print(f'Final RMSE over {n_splits * iterations}: {final_rmse:.6f}. Std {final_std:.4f}')
#     # print(f'RMSE by fold {np.mean(cv_rmse):.6f}. Std {np.std(cv_rmse):.4f}')
#     return test_preds, valid_preds, final_rmse, model 