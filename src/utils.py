"""Module for utils functions"""
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_percentage_error, mean_squared_error, r2_score
from time import time


def grid_search_by_hand(model, param_grid, X_train, y_train, cv, scoring):
    best_score = 0
    param_combinations = list(itertools.product(*param_grid.values()))

    for param_comb in tqdm(param_combinations):
        param_dict = dict(zip(param_grid.keys(), param_comb))
        model.set_params(**param_dict)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        score = scores.mean()
        if score > best_score:
            best_score = score
            best_params = param_dict
    model.set_params(**best_params)
    return {'best_model': model, 'best_params': best_params}


def model_evaluation_clf(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred), 3)
    recall = round(recall_score(y_test, y_pred), 3)
    f1 = round(f1_score(y_test, y_pred), 3)
    roc_auc = round(roc_auc_score(y_test, y_pred), 3)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}


def model_evaluation_lr(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mape = round(mean_absolute_percentage_error(y_test, y_pred), 3)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
    r2 = round(r2_score(y_test, y_pred), 3)
    return {'mape': mape, 'rmse': rmse, 'r2': r2}


def VIF_calculation(X):
    """Calculate the Variance Inflation Factor (VIF) for each variable in the dataset"""
    VIF = pd.DataFrame()
    VIF["variable"] = X.columns
    VIF["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    VIF = VIF.sort_values('VIF', ascending=False).reset_index(drop = True)
    return(VIF)


def delete_multicollinearity(df, target_name, VIF_threshold):
    """Delete multicollinearity from the dataset"""
    X = df.drop(target_name, axis=1)
    start = time()
    VIF_mat = VIF_calculation(X)
    end = time()
    n_VIF = VIF_mat["VIF"][0]
    if (n_VIF <= VIF_threshold):
        print("There is no multicollinearity!")
    else:
        while (n_VIF > VIF_threshold):
            print(f'Dropped column {VIF_mat["variable"][0]} with VIF: {round(VIF_mat["VIF"][0], 1)} ({int(end - start)}s)')
            X = X.drop(VIF_mat["variable"][0], axis=1)
            start = time()
            VIF_mat = VIF_calculation(X)
            end = time()
            n_VIF = VIF_mat["VIF"][0]
    return X