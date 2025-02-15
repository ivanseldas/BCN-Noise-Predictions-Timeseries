from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd


folder_path = '../../data/processed/ETL/sensor_id=496'

df_sensor = pd.read_parquet(folder_path)

# One-hot-encoding for lockdown periods (categorical data to numerical)
df_sensor_encoded = pd.get_dummies(df_sensor, columns=['Lockdown'], prefix='lockdown')

# Prepare features (X) and target (y)
X = df_sensor_encoded.drop(columns=['Noise_db'])
y = df_sensor_encoded['Noise_db']

# Define the models to evaluate
model_definitions = {
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
}

# Create a TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=3)

# Initialize lists to store results
all_results = []
all_predictions = []

# Iterate over each model
for model_name, model in model_definitions.items():
    print(f"Evaluating {model_name}...")
    fold_results = []
    predictions_list = []

    # Perform TimeSeriesSplit
    for fold, (train_index, test_index) in enumerate(tscv.split(X), start=1):
        # Split the data into train and test sets for the current fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model on the current fold
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics for the current fold
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store fold results
        fold_results.append({"Model": model_name, "Fold": fold, "RMSE": rmse, "MAPE": mape, "R2": r2})

        # Save predictions
        fold_predictions = pd.DataFrame({
            "Model": model_name,
            "Fold": fold,
            "Actual": y_test,
            "Predicted": y_pred
        })
        predictions_list.append(fold_predictions)

    # Aggregate results for this model
    all_results.extend(fold_results)
    all_predictions.extend(predictions_list)

# Create a DataFrame for all results
all_results_df = pd.DataFrame(all_results)

# Ensure each DataFrame in `all_predictions` has a datetime index before concatenating
for df in all_predictions:
    if 'Timestamp' in df.columns:  # Replace 'Timestamp' with the column representing datetime
        df.set_index('Timestamp', inplace=True)

# Combine all predictions into a single DataFrame
all_predictions_df = pd.concat(all_predictions, ignore_index=False)