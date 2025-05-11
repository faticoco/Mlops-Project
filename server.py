import os
import sys
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import uvicorn
from typing import List, Optional, Dict, Any

app = FastAPI(title="Housing Price Prediction API", 
              description="API for predicting housing prices based on features")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define input model
class HousingFeature(BaseModel):
    features: List[Any]
    feature_names: Optional[List[str]] = None

# Define output model
class PredictionResponse(BaseModel):
    prediction: int
    interpretation: str
    probability: Optional[float] = None

def load_model(model_path="models/GradientBoosting_20250511_120457.pkl"):
    """
    Load the model from a pickle file
    
    Args:
        model_path: Path to the model pickle file
    
    Returns:
        Loaded model
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        print(f"Successfully loaded model from {model_path}")
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def prepare_input_data(input_data, feature_names=None):
    """
    Prepare input data for prediction
    
    Args:
        input_data: Input data as a list or dict
        feature_names: List of feature names (optional)
    
    Returns:
        Prepared input data (DataFrame or numpy array)
    """
    if isinstance(input_data, list):
        if feature_names:
            return pd.DataFrame([input_data], columns=feature_names)
        else:
            return np.array([input_data])
    else:
        return input_data

def get_prediction_interpretation(prediction):
    """
    Interpret binary predictions in context of the housing price classification
    
    Args:
        prediction: Binary prediction (0 or 1)
        
    Returns:
        Interpretation string
    """
    if prediction == 1:
        return "Above median housing price"
    else:
        return "Below median housing price"

# Load model at startup
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()
    if model is None:
        print("Warning: Model could not be loaded. API will not function properly.")

@app.get("/")
def read_root():
    return {"message": "Housing Price Prediction API"}

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: HousingFeature):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process the input CSV-like data if needed
        processed_data = prepare_input_data(data.features, data.feature_names)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Get interpretation
        interpretation = get_prediction_interpretation(prediction)
        
        # Get probability if the model supports it
        probability = None
        try:
            if hasattr(model, 'predict_proba'):
                probability = float(model.predict_proba(processed_data)[0][1])
        except:
            pass
        
        return {
            "prediction": int(prediction),
            "interpretation": interpretation,
            "probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
def batch_predict(data: List[HousingFeature]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for item in data:
        try:
            processed_data = prepare_input_data(item.features, item.feature_names)
            prediction = model.predict(processed_data)[0]
            interpretation = get_prediction_interpretation(prediction)
            
            probability = None
            try:
                if hasattr(model, 'predict_proba'):
                    probability = float(model.predict_proba(processed_data)[0][1])
            except:
                pass
            
            results.append({
                "prediction": int(prediction),
                "interpretation": interpretation,
                "probability": probability
            })
        except Exception as e:
            results.append({"error": str(e)})
    
    return results

@app.post("/predict-csv-row")
def predict_csv_row(csv_row: str):
    """
    Predict based on a CSV row like the sample provided
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Parse the CSV row
        features = csv_row.strip().split(',')
        
        # Map the House features to model features
        # Note: You might need to adjust this mapping to match your model's expected input
        processed_features = process_housing_csv_features(features)
        
        # Make prediction using processed features
        processed_data = prepare_input_data(processed_features)
        prediction = model.predict(processed_data)[0]
        interpretation = get_prediction_interpretation(prediction)
        
        probability = None
        try:
            if hasattr(model, 'predict_proba'):
                probability = float(model.predict_proba(processed_data)[0][1])
        except:
            pass
        
        return {
            "prediction": int(prediction),
            "interpretation": interpretation,
            "probability": probability,
            "processed_features": processed_features
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

def process_housing_csv_features(features):
    """
    Process the raw CSV features to match the model's expected input format.
    This is a placeholder function - you'll need to implement the actual transformation
    based on your model's requirements.
    
    Example mapping for the sample data:
    1,60,RL,65.0,8450,Pave,Reg,Lvl,AllPub,Inside,Gtl,CollgCr,Norm,Norm,1Fam,2Story,7,5,2003,2003,Gable,CompShg,VinylSd,VinylSd,BrkFace,196.0,Gd,TA,PConc,Gd,TA,No,GLQ,706,Unf,0,150,856,GasA,Ex,Y,SBrkr,856,854,0,1710,1,0,2,1,3,1,Gd,8,Typ,0,,Attchd,2003.0,RFn,2,548,TA,TA,Y,0,61,0,0,0,0,0,2,2008,WD,Normal,208500
    """
    # This is a simplified example - you'll need to adapt this to your specific model
    # Extract numerical features and convert categorical ones as needed
    
    # For example, let's extract some key numerical features
    try:
        # Example: Extract and transform key features from the CSV row
        # You'll need to modify this based on what your model expects
        processed_features = [
            float(features[4]),  # Lot Area
            float(features[17]) if features[17] else 0,  # Year Built
            float(features[18]) if features[18] else 0,  # Year Remodeled
            float(features[32]) if features[32] else 0,  # 1st Floor SF
            float(features[33]) if features[33] else 0,  # 2nd Floor SF
            float(features[35]) if features[35] else 0,  # Total above ground living area
            float(features[46]) if features[46] else 0,  # Garage Year Built
            float(features[49]) if features[49] else 0   # Garage Area
        ]
        return processed_features
    except Exception as e:
        print(f"Error processing features: {e}")
        # Return a default set of features if processing fails
        return [8450.0, 2003.0, 2003.0, 856.0, 854.0, 1710.0, 2003.0, 548.0]


if __name__ == "__main__":
    # Change from "app:app" to "server:app" since the file is named server.py
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)