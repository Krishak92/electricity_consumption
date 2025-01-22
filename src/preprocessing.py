import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler

def download_data():

    df_train = pd.read_csv("data/train_df.csv")
    df_test = pd.read_csv("data/test_df.csv")

    return df_train, df_test

def scaling(df_train, df_test):

    scaler = StandardScaler()
    columns_to_scale = ['Factor_A', 'Factor_B', 'Factor_C', 'Factor_D', 'Factor_E', 'Factor_F']
    df_train[columns_to_scale] = scaler.fit_transform(df_train[columns_to_scale])
    df_test[columns_to_scale] = scaler.transform(df_test[columns_to_scale])

    for df in [df_train, df_test]:
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['Weekday'] = df.index.weekday

def new_columns(df_train, df_test):

    target_column = "Electric_Consumption"  
    feature_columns_train = [col for col in df_train.columns if col not in ['Date', target_column]] 
    feature_columns_test = [col for col in df_test.columns if col not in ['Date']]

    # Préparation des séries temporelles
    train_target = df_train[target_column]
    train_features = df_train[feature_columns_train]
    test_features = df_test[feature_columns_test]

    return target_column, train_target, train_features, test_features