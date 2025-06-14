import os
import argparse
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, mae, r2, rmse

def main(train_path, test_path):
    print("📥 Memuat dataset hasil preprocessing...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop("Chance_of_Admit", axis=1)
    y_train = train_df["Chance_of_Admit"]
    X_test = test_df.drop("Chance_of_Admit", axis=1)
    y_test = test_df["Chance_of_Admit"]

    print("🚀 Hyperparameter Tuning XGBoost dengan GridSearchCV...")
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0]
    }

    model = xgb.XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("✅ Best Parameters:", grid_search.best_params_)

    print("📈 Melakukan prediksi dengan model terbaik...")
    preds = best_model.predict(X_test)

    print("🧮 Evaluasi model...")
    mse, mae, r2, rmse = evaluate_model(y_test, preds)

    print("📦 Logging hasil ke MLflow...")
    with mlflow.start_run(run_name="xgboost-tuning"):
        # Log best parameters
        mlflow.log_params(grid_search.best_params_)

        # Log evaluation metrics
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2_Score", r2)

        # Log the model
        mlflow.xgboost.log_model(best_model, artifact_path="model")

    print(f"🔍 MSE      : {mse:.4f}")
    print(f"🔍 MAE      : {mae:.4f}")
    print(f"🔍 RMSE     : {rmse:.4f}")
    print(f"🔍 R2 Score : {r2:.4f}")
    print("✅ Model dan metrik telah disimpan ke MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model for Graduate Admission prediction.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset CSV.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the testing dataset CSV.")
    args = parser.parse_args()

    # Set MLflow tracking URI using environment variables (for security)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        print("⚠️ MLFLOW_TRACKING_URI tidak ditemukan di environment variable, menggunakan default (local logging).")

    mlflow.set_experiment("Graduate_Admission2")

    main(args.train_path, args.test_path)
