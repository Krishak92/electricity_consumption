import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarima(df_train, df_test, train_features, test_features, train_target, target_column):

    exog_train = train_features.values
    exog_test = test_features.values

    #configuration modèle SARIMA
    sarima_model = SARIMAX(
        endog=train_target,
        exog=exog_train,
        order=(3, 0, 2),            #paramètres ARIMA (p, d, q)
        seasonal_order=(2, 0, 1, 12),  #paramètres saisonniers (P, D, Q, S)
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    sarima_result = sarima_model.fit(disp=False)

    predictions = sarima_result.predict(
        exog=exog_test
    )

    #ajout predictions dans le dataset
    df_test['Electric_Consumption'] = predictions

    #visualisation résultats
    plt.figure(figsize=(10, 6))
    plt.plot(df_train[target_column], label='Train Data', color='blue')
    plt.plot(df_test['Electric_Consumption'], label='SARIMA Predictions', color='green')
    plt.legend()
    plt.title('SARIMA Predictions vs Actual Data')
    plt.xlabel('Date')
    plt.ylabel('Target')
    plt.show()

    #on retire "Date" comme index pour pouvoir étudier correctement le dataset avec LSTM ensuite
    df_test.reset_index(inplace=True)
    df_test[['Date', 'Electric_Consumption']].to_csv("results/predictions.csv", index=False)

def train_lstm(df_train, columns_to_scale, scaler):

    df_test = pd.read_csv("data/test_df.csv")
    df_test[columns_to_scale] = scaler.transform(df_test[columns_to_scale])
    df_test['Date'] = pd.to_datetime(df_test['Date'])
    df_test = df_test.sort_values(by='Date')

    target_column = "Electric_Consumption"  
    feature_columns_train = [col for col in df_train.columns if col not in ['Date', target_column]] 
    feature_columns_test = [col for col in df_test.columns if col not in ['Date']]

    # Définir la variable cible et les features
    X_train = df_train[feature_columns_train]  # Toutes les colonnes avant 'target'
    y_train = df_train[target_column]  # Colonne 'target'
    X_test = df_test[feature_columns_test]

    # Fonction pour créer des séquences
    def create_sequences(X, y, time_steps):
        if len(X) <= time_steps:
            raise ValueError("Le paramètre `time_steps` est trop grand pour la longueur des données.")
        
        X_seq, y_seq = [], []
        for i in range(len(X) - time_steps):
            X_seq.append(X[i:i + time_steps])
            y_seq.append(y[i + time_steps])
        return np.array(X_seq), np.array(y_seq)


    # Paramètre de séquencement (nombre de pas de temps)
    time_steps = 10

    # Créer les séquences
    X_train, y_train = create_sequences(X_train, y_train, time_steps)
    X_seq = []
    for i in range(len(X_test) - time_steps):
            X_seq.append(X_test[i:i + time_steps])
    X_test = np.array(X_seq)

    print("Shape finale de X_train :", X_train.shape)
    print("Shape finale de X_test :", X_test.shape)

    # Construire le modèle LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))  # Une sortie pour la prédiction de la target

    # Compiler le modèle
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entraîner le modèle
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    # Prédire sur l'ensemble de test
    predicted = model.predict(X_test)

    import numpy as np

    # Si `predicted` est à 2 dimensions (par exemple, shape (2150, 1))
    if len(predicted) < len(df_test):
        # Créer un padding de même forme que les prédictions
        padding = np.full((len(df_test) - len(predicted), predicted.shape[1]), np.nan)  # Padding avec NaN
        predicted = np.vstack([padding, predicted])  # Empiler verticalement

    # Vérification de la taille finale
    assert len(predicted) == len(df_test), "Les tailles de `predicted` et `df_test` ne correspondent toujours pas."

    # Ajouter les prédictions au DataFrame
    df_test['Electric_Consumption'] = predicted.squeeze()  # Retirer les dimensions inutiles si nécessaire

    # Remplir les valeurs manquantes ou non valides
    df_test['Electric_Consumption'] = df_test['Electric_Consumption'].fillna(0)  # Remplacer NaN par 0 ou une autre valeur

    # Exportation au format CSV
    df_test[['Date', 'Electric_Consumption']].to_csv("results/predictions2.csv", index=False)
    print("Exportation réussie.")