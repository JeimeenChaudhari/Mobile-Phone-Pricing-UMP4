import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('dataset.csv')

# Feature Engineering
print("\nEngineering features...")

# 1. Screen Area (height * width)
df['screen_area'] = df['sc_h'] * df['sc_w']

# 2. Pixel Density (simplified version)
# Handle division by zero
denominator = df['sc_h'] + df['sc_w']
df['px_density'] = np.where(
    denominator > 0,
    (df['px_height'] + df['px_width']) / denominator,
    0
)

# 3. Total Cameras
df['total_cameras'] = df['fc'] + df['pc']

print(f"Created features: screen_area, px_density, total_cameras")
print(f"Dataset shape after feature engineering: {df.shape}")

# Separate features and target
X = df.drop('price_range', axis=1)
y = df['price_range']

# Apply StandardScaler
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train RandomForestClassifier
print("\nTraining RandomForestClassifier with 200 estimators...")
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=20,
    min_samples_split=5,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Model training complete!")

# Evaluate the model
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy Score: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Low Cost', 'Medium', 'High', 'Very High']))

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Save artifacts
print("\n" + "="*60)
print("SAVING ARTIFACTS")
print("="*60)

# 1. Save the trained model
with open('price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Saved: price_model.pkl")

# 2. Save the fitted scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved: scaler.pkl")

# 3. Save the complete pre-feature-engineered DataFrame
with open('data.pkl', 'wb') as f:
    pickle.dump(df, f)
print("✓ Saved: data.pkl (complete dataset with engineered features)")

print("\n✅ Training pipeline completed successfully!")
print(f"Total features in model: {len(X.columns)}")
print(f"Feature names: {list(X.columns)}")