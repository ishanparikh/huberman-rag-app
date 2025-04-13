# train.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

print("--- Iris Model Training ---")

# 1. Load Data
iris = load_iris()
X = iris.data
y = iris.target
# Optional: Create a DataFrame for easier inspection
df = pd.DataFrame(X, columns=iris.feature_names)
df['species_idx'] = y
df['species_name'] = df['species_idx'].map({i: name for i, name in enumerate(iris.target_names)})
print("Loaded Iris dataset:")
print(df.sample(5)) # Print 5 random samples
print(f"Dataset shape: {df.shape}")

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

# 3. Train Model
# Using a simple Logistic Regression model
model = LogisticRegression(max_iter=200) # Increased max_iter for convergence
print(f"Training model: {type(model).__name__}...")
model.fit(X_train, y_train)
print("Model training complete.")

# 4. Evaluate Model (Optional)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test set: {accuracy:.4f}")

# 5. Save Model
model_filename = 'iris_model.joblib'
print(f"Saving model to {model_filename}...")
joblib.dump(model, model_filename)
print("Model saved successfully.")

# Also save target names for later use in the API
target_names_filename = 'iris_target_names.joblib'
joblib.dump(list(iris.target_names), target_names_filename)
print(f"Target names saved to {target_names_filename}.")

print("--- Training Script Finished ---")