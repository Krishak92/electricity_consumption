import os
from src.preprocessing import preprocess_data
from src.models import train_sarima, train_lstm
# from src.evaluation import evaluate_model

def main():
    #1.download data
    train_data_path = os.path.join("data", "train_df.csv")
    test_data_path = os.path.join("data", "test_df.csv")

    print("Data preprocessing...")
    df_train, df_test, target_column, train_target, train_features, test_features = preprocess_data(train_data_path, test_data_path)
    
    #2. modelisation
    print("SARIMA model training...")
    sarima_forecasts = train_sarima(df_train, df_test, train_features, test_features, train_target, target_column)

    print("LSTM model training...")
    lstm_forecasts = train_lstm(df_test, train_features, test_features, train_target, target_column)

    # #3. évaluation
    # print("Model evaluation...")
    # evaluate_model(sarima_forecasts, lstm_forecasts, processed_data_path)

    #3. publishing results

    sarima_forecasts[['Date', 'Electric_Consumption']].to_csv("results/predictions.csv", index=False)
    lstm_forecasts[['Date', 'Electric_Consumption']].to_csv("results/predictions2.csv", index=False)

    print("Pipeline ended successfully.")

if __name__ == "__main__":
    main()
