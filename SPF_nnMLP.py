import torch
import pandas as pd
import wandb
import numpy as np
import warnings

import torch.nn.functional as F
from torch import nn
from torch import tensor
from torch.nn.functional import relu
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import os

wandb.init(mode='disabled')
warnings.filterwarnings("ignore")

def preprocess_feats(feats, scaler=StandardScaler()):
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    feats.fillna(-1e10, inplace=True)
    feats_columns = feats.columns
    feats.loc[:, feats_columns != 'id'] = scaler.fit_transform(feats.loc[:, feats_columns != 'id'])
    return feats

def train_valid_split(data_x, data_y, train_idx, valid_idx):
    x_train = data_x.iloc[train_idx]
    y_train = data_y[train_idx]
    x_valid = data_x.iloc[valid_idx]
    y_valid = data_y[valid_idx]
    return x_train, y_train, x_valid, y_valid
class MLPModel(nn.Module):
    def __init__(self, feat_dim, 
    layer_size=32, 
    dropout_rate=0.3) -> None:
        super().__init__()
        
        self.input = nn.Linear(feat_dim, layer_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(layer_size, layer_size)
        self.out = nn.Linear(layer_size, 1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = self.dropout(x)
        x = F.relu(self.linear(x))
        x = self.out(x)
        return x
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_feats = pd.read_pickle('feature_selection/train_feats.pkl')
test_feats = pd.read_pickle('feature_selection/test_feats.pkl')

train_feats.iloc[:,:-1] = preprocess_feats(train_feats.iloc[:,:-1])


def calculate_rmse(y, yhat):
    return mean_squared_error(y, yhat, squared=False)

def nn_pipeline(train, test, model_class, param, epochs, n_splits=10, iterations=5):

    criterion = nn.MSELoss()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    test_preds = []
    valid_preds = pd.DataFrame()
    criterion = nn.MSELoss()

    targets = train_feats['score'].values.copy()
    x = tensor(train_feats.drop(['id','score'], axis=1).values, dtype=torch.float).to(device)
    y = tensor(targets, dtype=torch.float).to(device)

    for iter in range(iterations):
        for i, (train_index, valid_index) in enumerate(skf.split(x, targets.astype(str))):
            # Splitting data
            x_train, y_train = x[train_index], y[train_index]
            x_valid, y_valid = x[valid_index], y[valid_index]
            
            # Model setup
            model = model_class(**param).to(device)
            optimizer = torch.optim.Adam(model.parameters())

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(x_train)
                outputs = outputs.squeeze()  
                loss = criterion(outputs, y_train)
                rmse_loss = torch.sqrt(loss)
                rmse_loss.backward()
                optimizer.step()

                # Log training metrics

            # Validation predictions
            model.eval()
            valid_predictions = []
            with torch.no_grad():
                outputs = model(x_valid).squeeze()
                valid_predictions.extend(outputs.tolist())

            # Log validation metrics
            tmp_rmse = np.sqrt(mean_squared_error(y_valid.cpu().numpy(), valid_predictions))

            tmp_df = pd.DataFrame({'id': valid_index, 'score': y_valid.cpu().numpy(), 'preds': valid_predictions})
            tmp_df['iteration'] = i + 1
            valid_preds = pd.concat([valid_preds, tmp_df])

    final_rmse = np.sqrt(mean_squared_error(valid_preds['score'], valid_preds['preds']))
    cv_rmse = valid_preds.groupby('iteration').apply(lambda g: np.sqrt(mean_squared_error(g['score'], g['preds'])))
    return valid_preds, final_rmse, model

def run_experiment(layer_size, dropout_rate, epochs):

    model = MLPModel(feat_dim=train_feats.shape[1]-2, layer_size=layer_size, dropout_rate=dropout_rate).to(device)
    param = {'feat_dim': train_feats.shape[1]-2, 'layer_size': layer_size, 'dropout_rate': dropout_rate}

    valid_preds, final_rmse, trained_model = nn_pipeline(
        train=train_feats,  
        test=test_feats,
        model_class=MLPModel,
        param=param,
        epochs=epochs,
        n_splits=10,  
        iterations=5
    )

    print(f"Layer Size: {layer_size}, Dropout: {dropout_rate}, Epochs: {epochs}, Final RMSE: {final_rmse}")
    return valid_preds, final_rmse, trained_model

run = wandb.init(project='linking-quality-writing')

# Hyperparameters
layer_sizes = [24, 32, 48, 64, 128]
dropout_rates = list(np.arange(0.1, 0.3, 0.05))
epoch_settings = [100, 200, 300, 400, 500, 600]

# Running the experiments
for layer_size in layer_sizes:
    for dropout_rate in dropout_rates:
        for epochs in epoch_settings:
            # Start a new wandb session for each set of hyperparameters
            run = wandb.init(project='linking-quality-writing')
            
            # Update wandb configuration
            wandb.config.update({
                'epochs': epochs,
                'dropout_rate': dropout_rate,
                'layer_size': layer_size
            })

            # Run the experiment and log the results
            valid_preds, final_rmse, trained_model = run_experiment(layer_size, dropout_rate, epochs)
            wandb.log({'final_rmse': final_rmse})
            
            # Finish the wandb session
            run.finish()

epochs = 500
model = MLPModel(train_feats.shape[1]).to(device)

param = {'feat_dim': train_feats.drop(['id','score'], axis=1).shape[1]}
# Using the nn_pipeline function
valid_preds, final_rmse, trained_model = nn_pipeline(
    train=train_feats,  
    test=test_feats,
    model_class=MLPModel,
    param=param,
    n_splits=10,  
    iterations=5  
)

print(f"Final RMSE: {final_rmse}")