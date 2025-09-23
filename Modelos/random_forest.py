import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
from datetime import datetime
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# MLflow setup
mlflow.set_experiment("demand_forecasting_successful_experiments")

def load_and_prepare_data(data_path="../data/raw/train.csv"):
    """
    Load and prepare the training data with optimized feature engineering
    Based on feature importance analysis from successful experiments:
    Top features: sales_rolling_mean_7, sales_lag_7, sales_rolling_mean_14, 
    dayofweek, dayofweek_sin/cos, sales_lag_1, sales_lag_14
    """
    print("Loading and preparing data...")
    
    # Load data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Stores: {df['store'].nunique()}, Items: {df['item'].nunique()}")
    
    # Core temporal features (most important)
    df['dayofweek'] = df['date'].dt.dayofweek
    df['day'] = df['date'].dt.day
    df['dayofyear'] = df['date'].dt.dayofyear
    
    # Essential seasonal indicators
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Critical cyclical features (high importance in experiments)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Sort by store, item, and date for lag features
    df = df.sort_values(['store', 'item', 'date'])
    
    # Most important lag features (based on feature importance)
    for lag in [1, 7, 14]:  # Removed 30-day lag (lower importance)
        df[f'sales_lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)
    
    # Most important rolling statistics (top performers in experiments)
    for window in [7, 14, 30]:  # Keep 30 for rolling_mean as it shows high importance
        df[f'sales_rolling_mean_{window}'] = df.groupby(['store', 'item'])['sales'].rolling(
            window=window, min_periods=1
        ).mean().reset_index(0, drop=True)
    
    # Only essential rolling std (7-day shows consistent importance)
    df[f'sales_rolling_std_7'] = df.groupby(['store', 'item'])['sales'].rolling(
        window=7, min_periods=1
    ).std().reset_index(0, drop=True)
    
    # Fill NaN values for lag features with 0 (for the beginning of time series)
    lag_columns = [col for col in df.columns if 'lag' in col or 'rolling' in col]
    df[lag_columns] = df[lag_columns].fillna(0)
    
    print(f"Final dataset shape after feature engineering: {df.shape}")
    
    return df

def train_model(df, params, run_name):
    """
    Train a RandomForest model with specified parameters
    """
    print(f"\nTraining model: {run_name}")
    print(f"Parameters: {params}")
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Log dataset info
        mlflow.set_tag("data_shape", f"{df.shape[0]} x {df.shape[1]}")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("experiment_date", datetime.now().isoformat())
        
        # Prepare features
        feature_columns = [col for col in df.columns if col not in ['date', 'sales']]
        X = df[feature_columns]
        y = df['sales']
        
        # Train-test split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False  # Keep temporal order
        )
        
        mlflow.set_tag("train_test_split", f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Create and train model
        model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **params
        )
        
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Cross-validation
        print("Performing cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                  scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse_scores = np.sqrt(-cv_scores)
        cv_rmse_mean = cv_rmse_scores.mean()
        cv_rmse_std = cv_rmse_scores.std()
        
        # Overfitting ratio
        overfitting_ratio = test_rmse / train_rmse
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("cv_rmse_mean", cv_rmse_mean)
        mlflow.log_metric("cv_rmse_std", cv_rmse_std)
        mlflow.log_metric("overfitting_ratio", overfitting_ratio)
        
        # Log top 10 feature importances
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            mlflow.log_metric(f"feature_importance_{i+1}_{row['feature']}", row['importance'])
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"CV RMSE: {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")
        print(f"Overfitting ratio: {overfitting_ratio:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        return {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'cv_rmse_mean': cv_rmse_mean,
            'feature_importance': feature_importance
        }

def main():
    
    # Load data
    try:
        df = load_and_prepare_data()
    except FileNotFoundError:
        print("Error: Could not find train.csv file. Please make sure the data file exists at ../data/raw/train.csv")
        sys.exit(1)
    
    experiments = [
        {
            'name': 'RF_Best_Performance',
            'params': {
                'n_estimators': 137,
                'max_depth': 23,
                'min_samples_split': 12,
                'min_samples_leaf': 11,
                'max_features': 0.57
            }
        },
        {
            'name': 'RF_High_Trees_Deep',
            'params': {
                'n_estimators': 199,
                'max_depth': 25,
                'min_samples_split': 4,
                'min_samples_leaf': 8,
                'max_features': 0.87
            }
        },
        {
            'name': 'RF_Balanced_Medium',
            'params': {
                'n_estimators': 171,
                'max_depth': 12,
                'min_samples_split': 8,
                'min_samples_leaf': 5,
                'max_features': 0.37
            }
        },
        {
            'name': 'RF_Conservative_Small',
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        },
        {
            'name': 'RF_Deep_Medium',
            'params': {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        },
        {
            'name': 'RF_Feature_Rich',
            'params': {
                'n_estimators': 156,
                'max_depth': 11,
                'min_samples_split': 16,
                'min_samples_leaf': 13,
                'max_features': 0.87
            }
        },
        {
            'name': 'RF_Medium_Depth',
            'params': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        },
        {
            'name': 'RF_High_Trees_Conservative',
            'params': {
                'n_estimators': 300,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        },
        {
            'name': 'RF_High_Trees_Medium',
            'params': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        }
    ]
    
    results = []
    
    for exp in experiments:
        try:
            result = train_model(df, exp['params'], exp['name'])
            result['config_name'] = exp['name']
            results.append(result)
        except Exception as e:
            print(f"Error training {exp['name']}: {str(e)}")
            continue
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY OF RESULTS")
    print("="*80)
    
    results_df = pd.DataFrame([
        {
            'Configuration': r['config_name'],
            'Test RMSE': f"{r['test_rmse']:.4f}",
            'Test R²': f"{r['test_r2']:.4f}",
            'CV RMSE': f"{r['cv_rmse_mean']:.4f}",
            'Top Feature': r['feature_importance'].iloc[0]['feature']
        }
        for r in results
    ])
    
    # Sort by test RMSE (ascending - lower is better)
    results_df['Test RMSE (numeric)'] = [r['test_rmse'] for r in results]
    results_df = results_df.sort_values('Test RMSE (numeric)')
    results_df = results_df.drop('Test RMSE (numeric)', axis=1)
    
    print(results_df.to_string(index=False))
    
    best_model = min(results, key=lambda x: x['test_rmse'])
    print(f"\nBest performing model: {best_model['config_name']}")
    print(f"Best Test RMSE: {best_model['test_rmse']:.4f}")
    print(f"Best Test R²: {best_model['test_r2']:.4f}")
    
    print("\nAll models have been logged to MLflow experiment: 'demand_forecasting_successful_experiments'")
    print("Use 'mlflow ui' to view detailed comparison and metrics.")

if __name__ == "__main__":
    main()