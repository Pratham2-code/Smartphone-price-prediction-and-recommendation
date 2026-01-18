import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv('smartphones_data3.csv')

# --- 1. Preprocessing (Same as previous step) ---

# Convert numerical columns to integers
df['RAM'] = df['RAM'].astype(int)
df['storage'] = df['storage'].astype(int)
df['primery_rear_camera'] = df['primery_rear_camera'].astype(int)
df['primary_front_camera'] = df['primary_front_camera'].astype(int)

# Handle missing values and convert boolean-like columns to binary
df['has_5g'] = df['has_5g'].fillna('No')
df['has_5g'] = df['has_5g'].apply(lambda x: 1 if x in ['Yes', 'yes'] else 0)
df['has_fast_charging'] = df['has_fast_charging'].apply(lambda x: 1 if x in ['Yes', 'yes'] else 0)

# Handle missing values in 'refresh_rate(hz)' (assume standard is 60 Hz)
df['refresh_rate(hz)'] = df['refresh_rate(hz)'].fillna(60.0).astype(int)

# Drop the 'Model Name' column
df = df.drop('Model Name', axis=1)

# --- 2. Target Transformation and Feature Encoding ---

# Apply log transformation to the target variable 'Price'
df['log_Price'] = np.log1p(df['Price'])
df = df.drop('Price', axis=1)

# One-Hot Encoding
categorical_cols = ['brand_name', 'OS', 'processor_brand', 'display_types']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# --- 3. Split Data ---
X = df_encoded.drop('log_Price', axis=1)
y = df_encoded['log_Price']

# Store feature names for later use in the web app
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Train a Gradient Boosting Regressor ---
# Trying a more powerful model for a marginal gain
gbr_model = GradientBoostingRegressor(
    n_estimators=300,        # Increased estimators
    learning_rate=0.05,      # Lower learning rate for better convergence
    max_depth=5,             # Deeper trees than standard GBR for this dataset size
    random_state=42
)
gbr_model.fit(X_train, y_train)

# --- 5. Evaluate the Model ---
y_pred_log = gbr_model.predict(X_test)

# Inverse transform predictions for evaluation on the original price scale
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

# Calculate R2 and RMSE on the original price scale
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)

print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# --- 6. Save the trained model and feature list ---
# Overwrite the old files with the improved model
joblib.dump(gbr_model, 'smartphone_price_model.joblib')
joblib.dump(feature_names, 'model_features.joblib')