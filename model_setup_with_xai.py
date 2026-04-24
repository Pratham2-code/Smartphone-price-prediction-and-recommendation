import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ============================================================
# XAI IMPORTS — only addition to the original imports block
# ============================================================
import shap
import matplotlib
matplotlib.use('Agg')  # saves plots to files instead of opening windows
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import os

# Create a folder to save all XAI output plots
os.makedirs('xai_outputs', exist_ok=True)

# ============================================================
# SECTION 1 — Preprocessing  (UNCHANGED from your original)
# ============================================================

df = pd.read_csv('smartphones_data3.csv')

df['RAM'] = df['RAM'].astype(int)
df['storage'] = df['storage'].astype(int)
df['primery_rear_camera'] = df['primery_rear_camera'].astype(int)
df['primary_front_camera'] = df['primary_front_camera'].astype(int)

df['has_5g'] = df['has_5g'].fillna('No')
df['has_5g'] = df['has_5g'].apply(lambda x: 1 if x in ['Yes', 'yes'] else 0)
df['has_fast_charging'] = df['has_fast_charging'].apply(lambda x: 1 if x in ['Yes', 'yes'] else 0)

df['refresh_rate(hz)'] = df['refresh_rate(hz)'].fillna(60.0).astype(int)

df = df.drop('Model Name', axis=1)

# ============================================================
# SECTION 2 — Target Transformation & Encoding  (UNCHANGED)
# ============================================================

df['log_Price'] = np.log1p(df['Price'])
df = df.drop('Price', axis=1)

categorical_cols = ['brand_name', 'OS', 'processor_brand', 'display_types']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ============================================================
# SECTION 3 — Split Data  (UNCHANGED)
# ============================================================

X = df_encoded.drop('log_Price', axis=1)
y = df_encoded['log_Price']

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# SECTION 4 — Train Model  (UNCHANGED)
# ============================================================

gbr_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
gbr_model.fit(X_train, y_train)

# ============================================================
# SECTION 5 — Evaluate  (UNCHANGED)
# ============================================================

y_pred_log = gbr_model.predict(X_test)

y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)

print(f"Root Mean Squared Error (RMSE): ₹{rmse:,.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# ============================================================
# SECTION 6 — Save Model  (UNCHANGED)
# ============================================================

joblib.dump(gbr_model, 'smartphone_price_model.joblib')
joblib.dump(feature_names, 'model_features.joblib')
print("Model and features saved successfully.")

# ============================================================
# SECTION 7 — XAI  (NEW — everything below is added by you)
# ============================================================

print("\n--- Running XAI Analysis ---")

# ── XAI 1: Built-in Feature Importance (quick sanity check) ──────────────────
# This uses the model's own internal importance scores.
# It tells you which features the model relied on most during training.
# Limitation: does NOT show direction (positive/negative impact).

importances = gbr_model.feature_importances_
indices = np.argsort(importances)[::-1][:20]  # top 20 features only

plt.figure(figsize=(14, 6))
plt.bar(range(20), importances[indices], color='steelblue')
plt.xticks(
    range(20),
    [feature_names[i] for i in indices],
    rotation=45,
    ha='right',
    fontsize=9
)
plt.title("Top 20 Feature Importances (Built-in GBR)", fontsize=13)
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig('xai_outputs/1_builtin_feature_importance.png', dpi=150)
plt.close()
print("Saved: xai_outputs/1_builtin_feature_importance.png")

# ── XAI 2: SHAP Global Bar Chart ─────────────────────────────────────────────
# TreeExplainer is the right explainer for GradientBoostingRegressor.
# It computes exact SHAP values (not approximations) for tree-based models.
# shap_values shape: (n_test_samples, n_features)
# Each value = how much that feature pushed the log_Price up or down
# for that specific prediction.

explainer = shap.TreeExplainer(gbr_model)

# Compute SHAP values for the entire test set
# This may take 10–30 seconds depending on your machine
print("Computing SHAP values (this may take ~20 seconds)...")
shap_values = explainer.shap_values(X_test)

# Global bar chart: mean(|SHAP value|) per feature
# = average magnitude of impact across all predictions
plt.figure()
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    plot_type="bar",
    show=False,
    max_display=20
)
plt.title("SHAP Global Feature Importance", fontsize=13)
plt.tight_layout()
plt.savefig('xai_outputs/2_shap_global_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xai_outputs/2_shap_global_bar.png")

# ── XAI 3: SHAP Beeswarm Plot ────────────────────────────────────────────────
# Each dot = one phone in the test set.
# Horizontal position = SHAP value (right = pushes price UP, left = pushes DOWN).
# Color = feature value (red = high value like 16GB RAM, blue = low like 4GB).
# This shows both importance AND direction at the same time.

plt.figure()
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    show=False,
    max_display=20
)
plt.title("SHAP Beeswarm — Feature Impact Direction", fontsize=13)
plt.tight_layout()
plt.savefig('xai_outputs/3_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xai_outputs/3_shap_beeswarm.png")

# ── XAI 4: SHAP Waterfall Plot (single phone explanation) ────────────────────
# Explains ONE specific prediction in detail.
# Starts from the average predicted price across all phones (base value),
# then shows each feature either adding to or subtracting from that baseline.
# You can change PHONE_INDEX to explain any phone in the test set.

PHONE_INDEX = 0  # change this to explain a different phone (0 to len(X_test)-1)

phone_row = X_test.iloc[PHONE_INDEX]
phone_shap = shap_values[PHONE_INDEX]
base_value = explainer.expected_value

predicted_log_price = gbr_model.predict(phone_row.values.reshape(1, -1))[0]
predicted_price = np.expm1(predicted_log_price)
print(f"\nExplaining Phone #{PHONE_INDEX} — Predicted Price: ₹{predicted_price:,.0f}")

explanation = shap.Explanation(
    values=phone_shap,
    base_values=base_value,
    data=phone_row.values,
    feature_names=feature_names
)

plt.figure()
shap.plots.waterfall(explanation, show=False, max_display=15)
plt.title(f"Why did this phone get ₹{predicted_price:,.0f}?", fontsize=12)
plt.tight_layout()
plt.savefig('xai_outputs/4_shap_waterfall_single_phone.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xai_outputs/4_shap_waterfall_single_phone.png")

# ── XAI 5: SHAP Dependence Plot — RAM vs Price ───────────────────────────────
# Shows how the RAM feature affects predicted price across all phones.
# X-axis = actual RAM value (4, 6, 8, 12, 16 GB).
# Y-axis = SHAP value for RAM (how much it contributed to predicted price).
# Dot color = value of a second interacting feature (SHAP picks it automatically).
# This reveals non-linear effects: e.g. 8→12GB might add more than 4→8GB.

plt.figure()
shap.dependence_plot(
    "RAM",
    shap_values,
    X_test,
    feature_names=feature_names,
    show=False
)
plt.title("How RAM Affects Predicted Price (SHAP Dependence)", fontsize=13)
plt.tight_layout()
plt.savefig('xai_outputs/5_shap_dependence_RAM.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xai_outputs/5_shap_dependence_RAM.png")

# ── XAI 6: SHAP Dependence Plot — Storage vs Price ───────────────────────────

plt.figure()
shap.dependence_plot(
    "storage",
    shap_values,
    X_test,
    feature_names=feature_names,
    show=False
)
plt.title("How Storage Affects Predicted Price (SHAP Dependence)", fontsize=13)
plt.tight_layout()
plt.savefig('xai_outputs/6_shap_dependence_storage.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xai_outputs/6_shap_dependence_storage.png")

# ── XAI 7: LIME Explanation (single phone) ───────────────────────────────────
# LIME builds a simple local linear model around one prediction to explain it.
# It works differently from SHAP: it perturbs the input and observes changes.
# predict_fn must return predictions in the ORIGINAL price scale (rupees),
# not the log scale — so we wrap model.predict with np.expm1.

def predict_original_price(X_array):
    """Wrapper: model predicts log_Price, we convert back to ₹"""
    return np.expm1(gbr_model.predict(X_array))

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    mode='regression',
    random_state=42
)

lime_exp = lime_explainer.explain_instance(
    data_row=X_test.iloc[PHONE_INDEX].values,
    predict_fn=predict_original_price,
    num_features=12  # show top 12 influential features
)

# Save as HTML (opens in any browser — shows interactive bar chart)
lime_exp.save_to_file('xai_outputs/7_lime_single_phone.html')
print("Saved: xai_outputs/7_lime_single_phone.html  (open in browser)")

# Also save LIME as a static PNG
lime_fig = lime_exp.as_pyplot_figure()
lime_fig.suptitle(f"LIME Explanation — Phone #{PHONE_INDEX} (₹{predicted_price:,.0f})", fontsize=12)
lime_fig.tight_layout()
lime_fig.savefig('xai_outputs/7_lime_single_phone.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xai_outputs/7_lime_single_phone.png")

# ============================================================
# SECTION 8 — What-If Analysis Function  (NEW)
# ============================================================
# Call this function anywhere in your code to answer:
# "What if this phone had 12GB RAM instead of 8GB?"
# "What if it had 5G support?"

def what_if_analysis(phone_index, feature_to_change, new_value):
    """
    Compares the predicted price of a phone before and after
    changing one feature value.

    Parameters:
        phone_index    : int  — index of the phone in X_test
        feature_to_change : str  — exact column name (e.g. 'RAM', 'storage', 'has_5g')
        new_value      : the new value to set for that feature

    Returns:
        dict with original_price, new_price, and price_change in rupees
    """
    original_row = X_test.iloc[phone_index].copy()
    modified_row = original_row.copy()
    modified_row[feature_to_change] = new_value

    original_price = np.expm1(gbr_model.predict(original_row.values.reshape(1, -1))[0])
    new_price      = np.expm1(gbr_model.predict(modified_row.values.reshape(1, -1))[0])
    delta          = new_price - original_price

    print(f"\n--- What-If Analysis ---")
    print(f"Phone index        : {phone_index}")
    print(f"Feature changed    : {feature_to_change}")
    print(f"  {original_row[feature_to_change]}  →  {new_value}")
    print(f"Original price     : ₹{original_price:,.0f}")
    print(f"New predicted price: ₹{new_price:,.0f}")
    print(f"Price change       : ₹{delta:+,.0f}")

    return {
        'original_price': round(original_price, 2),
        'new_price':      round(new_price, 2),
        'price_change':   round(delta, 2)
    }

# Example calls — these run automatically when you execute the file
print("\n=== What-If Examples ===")
what_if_analysis(PHONE_INDEX, 'RAM', 12)          # What if RAM goes to 12GB?
what_if_analysis(PHONE_INDEX, 'storage', 256)     # What if storage goes to 256GB?
what_if_analysis(PHONE_INDEX, 'has_5g', 0)        # What if 5G is removed?

# ============================================================
# Final summary
# ============================================================
print("\n=== XAI Analysis Complete ===")
print("All outputs saved in the 'xai_outputs/' folder:")
print("  1_builtin_feature_importance.png  — model's own ranking of features")
print("  2_shap_global_bar.png             — SHAP global importance (better than built-in)")
print("  3_shap_beeswarm.png               — SHAP direction of impact per feature")
print("  4_shap_waterfall_single_phone.png — why this phone got this price")
print("  5_shap_dependence_RAM.png         — how RAM level affects price")
print("  6_shap_dependence_storage.png     — how storage affects price")
print("  7_lime_single_phone.html          — LIME explanation (open in browser)")
print("  7_lime_single_phone.png           — LIME explanation (static image)")
