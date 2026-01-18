from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# --- 1. Define Allowed Chipset Brands and Price Tolerance ---
ALLOWED_PROCESSOR_BRANDS = ['snapdragon', 'mediatek', 'apple', 'unisoc', 'samsung']
PRICE_TOLERANCE_PERCENT = 0.10  # 10% tolerance for recommendations (e.g., +/- 10%)

# --- 2. Load Model, Features, and Data ---
try:
    gbr_model = joblib.load('smartphone_price_model.joblib')
    feature_names = joblib.load('model_features.joblib')
    
    # Load the full dataset to pull recommendation details
    df_raw = pd.read_csv('smartphones_data3.csv')
    
    unique_os = sorted(df_raw['OS'].unique().tolist())
    
    # Filter processor brands for the UI
    unique_processors = [p for p in sorted(df_raw['processor_brand'].unique().tolist()) if p in ALLOWED_PROCESSOR_BRANDS]
    
    unique_displays = sorted(df_raw['display_types'].unique().tolist())

except FileNotFoundError:
    print("ERROR: Model, features file, or data file not found. Please ensure all files are in the directory.")
    exit()

# --- 3. Flask Setup ---
app = Flask(__name__)

def prepare_user_input(form_data, feature_names):
    """Converts form data into the feature vector expected by the model."""
    
    # Create a zero-filled DataFrame with all expected features
    data = {}
    for feature in feature_names:
        data[feature] = [0]
    
    input_df = pd.DataFrame(data)

    # Set numerical/simple binary features
    input_df['RAM'] = int(form_data.get('ram') or 0)
    input_df['storage'] = int(form_data.get('storage') or 0)
    input_df['Battery_capacity'] = int(form_data.get('battery_capacity') or 0)
    input_df['primery_rear_camera'] = int(form_data.get('rear_camera') or 0)
    input_df['primary_front_camera'] = int(form_data.get('front_camera') or 0)
    input_df['refresh_rate(hz)'] = int(form_data.get('refresh_rate') or 0)
    
    input_df['has_fast_charging'] = 1 if form_data.get('fast_charging') == 'on' else 0
    input_df['has_5g'] = 1 if form_data.get('has_5g') == 'on' else 0

    # Set One-Hot Encoded features
    
    # Handle OS
    os_col = f"OS_{form_data['os']}"
    if os_col in input_df.columns:
        input_df[os_col] = 1

    # Handle processor_brand
    processor_brand = form_data['processor_brand']
    processor_col = f"processor_brand_{processor_brand}"
    if processor_col in input_df.columns:
        input_df[processor_col] = 1
        
    # Handle display_types
    display_col = f"display_types_{form_data['display_types']}"
    if display_col in input_df.columns:
        input_df[display_col] = 1

    # Ensure the order of columns matches the training data
    return input_df[feature_names]

def get_recommendations(predicted_price, df, tolerance):
    """Finds phones in the dataset whose price is within the tolerance band."""
    
    # Calculate the min and max price for the recommendation band
    min_price = predicted_price * (1 - tolerance)
    max_price = predicted_price * (1 + tolerance)
    
    # Filter the dataset
    recommendations_df = df[
        (df['Price'] >= min_price) & 
        (df['Price'] <= max_price)
    ]
    
    # Select Model Name, Price, and Brand, and sort by price difference from predicted price
    recommendations_df['price_diff'] = np.abs(recommendations_df['Price'] - predicted_price)
    
    # Grab the top 5 closest matches, format their price, and return a list of dictionaries
    top_recommendations = recommendations_df.sort_values(by='price_diff').head(5)
    
    recommendation_list = []
    for index, row in top_recommendations.iterrows():
        recommendation_list.append({
            'name': row['Model Name'],
            'price': f"₹{row['Price']:,.0f}",
            'brand': row['brand_name']
        })
        
    return recommendation_list

@app.route('/')
def index():
    """Renders the main input form."""
    return render_template(
        'index.html',
        oss=unique_os,
        processors=unique_processors,
        displays=unique_displays
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    try:
        # 1. Prepare and Predict
        input_data = prepare_user_input(request.form, feature_names)
        log_prediction = gbr_model.predict(input_data)[0]
        predicted_price = np.expm1(log_prediction)
        
        # 2. Format Prediction Output
        formatted_price = f"₹{predicted_price:,.0f}"
        
        # 3. Get Recommendations
        recommendations = get_recommendations(predicted_price, df_raw, PRICE_TOLERANCE_PERCENT)
        
        return render_template(
            'index.html', 
            prediction_text=f"The estimated price is: {formatted_price}",
            recommendations=recommendations,
            oss=unique_os,
            processors=unique_processors,
            displays=unique_displays
        )

    except Exception as e:
        error_message = f"An error occurred: {e}"
        # Render the page with the error and original options
        return render_template(
            'index.html', 
            prediction_text=error_message,
            oss=unique_os,
            processors=unique_processors,
            displays=unique_displays
        )

if __name__ == '__main__':
    app.run(debug=True)