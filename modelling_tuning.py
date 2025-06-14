import os
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create a new MLflow Experiment
mlflow.set_experiment("Graduate_Admission2")

mlflow.set_tracking_uri("https://dagshub.com/<username>/<repo>.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = "<your_dagshub_username>"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "<your_token>"

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, mae, r2, rmse

def main():
    print("📥 Memuat dataset hasil preprocessing...")
    train_df = pd.read_csv("../preprocessing/Graduate_Admission2_preprocessing/train_clean.csv")
    test_df = pd.read_csv("../preprocessing/Graduate_Admission2_preprocessing/test_clean.csv")

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
    main()
