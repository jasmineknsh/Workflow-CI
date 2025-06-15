import os
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

mlflow.set_experiment("Graduate_Admission2")

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
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100],
        'subsample': [1.0]
    }

    model = xgb.XGBRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("✅ Best Parameters:", grid_search.best_params_)

    preds = best_model.predict(X_test)
    mse, mae, r2, rmse = evaluate_model(y_test, preds)

    print("📦 Logging ke MLflow & menyimpan artefak model...")
    with mlflow.start_run(run_name="xgboost-tuning"):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics({
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2
        })
        mlflow.xgboost.log_model(best_model, "model")

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(best_model, "artifacts/xgboost_best_model.pkl")
        mlflow.log_artifact("artifacts/xgboost_best_model.pkl")

    print("✅ Logging selesai dan artefak berhasil disimpan.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    args = parser.parse_args()
    main(args.train_path, args.test_path)