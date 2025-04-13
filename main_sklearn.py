# main_sklearn.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import List

# --- Configuration & Model Loading ---
MODEL_FILE = 'iris_model.joblib'
TARGET_NAMES_FILE = 'iris_target_names.joblib'

# Load the model and target names when the application starts
# Use relative paths assuming the files are in the same directory as main_sklearn.py
try:
    print(f"Loading model from {MODEL_FILE}...")
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")
    
    print(f"Loading target names from {TARGET_NAMES_FILE}...")
    target_names = joblib.load(TARGET_NAMES_FILE)
    print(f"Target names loaded: {target_names}")
    
except FileNotFoundError as e:
    print(f"ERROR: Model file or target names file not found: {e}")
    print("Please ensure train.py has been run successfully in the same directory.")
    # In a real application, you might want to exit or handle this more gracefully
    model = None 
    target_names = None
except Exception as e:
    print(f"An error occurred during model loading: {e}")
    model = None
    target_names = None

# --- Pydantic Models for Input and Output ---

class IrisFeatures(BaseModel):
    """Defines the input features for Iris prediction."""
    sepal_length: float 
    sepal_width: float
    petal_length: float
    petal_width: float
    
    # Example for documentation
    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionResponse(BaseModel):
    """Defines the output prediction."""
    predicted_species: str

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Scikit-learn Iris Model API",
    description="API to predict Iris species using a trained Logistic Regression model.",
    version="1.0.0"
)

# --- API Endpoints ---

@app.get("/")
async def read_root():
    """Root endpoint for health check."""
    if model is None or target_names is None:
         raise HTTPException(status_code=500, detail="Model or target names failed to load.")
    return {"status": "Iris prediction API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_species(features: IrisFeatures):
    """Predicts the Iris species based on input features."""
    if model is None or target_names is None:
         raise HTTPException(status_code=500, detail="Model is not loaded, cannot predict.")
         
    try:
        # Convert input features into a 2D NumPy array (model expects 2D input)
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Make prediction (predict returns an array of predictions)
        prediction_index = model.predict(input_data)[0]
        
        # Map index to species name
        predicted_species_name = target_names[prediction_index]
        
        print(f"Input: {features.model_dump()}, Predicted Index: {prediction_index}, Predicted Name: {predicted_species_name}")
        
        return PredictionResponse(predicted_species=predicted_species_name)
        
    except IndexError:
        print(f"Error: Prediction index {prediction_index} out of bounds for target names {target_names}")
        raise HTTPException(status_code=500, detail="Prediction failed: Invalid prediction index.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# --- Optional: Run directly (usually done via CLI) ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8080)