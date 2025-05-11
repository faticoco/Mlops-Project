from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to allow importing from src/model
from inference import load_model, prepare_input_data_with_defaults, make_predictions

# Initialize FastAPI app
app = FastAPI(
    title="Housing Price Prediction API",
    description="API for predicting housing prices using an MLflow model",
    version="1.0.0"
)
# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()
    if model is None:
        raise Exception("Failed to load model at startup")

# Pydantic model for input validation
class HouseFeatures(BaseModel):
    LotArea: Optional[int] = None
    OverallQual: Optional[int] = None
    OverallCond: Optional[int] = None
    YearBuilt: Optional[int] = None
    YearRemodAdd: Optional[int] = None
    BsmtFinSF1: Optional[int] = None
    BsmtFinSF2: Optional[int] = None
    BsmtUnfSF: Optional[int] = None
    TotalBsmtSF: Optional[int] = None
    FirstFlrSF: Optional[int] = None  # Using FirstFlrSF as '1stFlrSF' isn't a valid variable name
    SecondFlrSF: Optional[int] = None  # Using SecondFlrSF as '2ndFlrSF' isn't a valid variable name
    LowQualFinSF: Optional[int] = None
    GrLivArea: Optional[int] = None
    BsmtFullBath: Optional[int] = None
    BsmtHalfBath: Optional[int] = None
    FullBath: Optional[int] = None
    HalfBath: Optional[int] = None
    BedroomAbvGr: Optional[int] = None
    KitchenAbvGr: Optional[int] = None
    TotRmsAbvGrd: Optional[int] = None
    Fireplaces: Optional[int] = None
    GarageCars: Optional[int] = None
    GarageArea: Optional[int] = None
    WoodDeckSF: Optional[int] = None
    OpenPorchSF: Optional[int] = None
    EnclosedPorch: Optional[int] = None
    ThreeSsnPorch: Optional[int] = None  # Using ThreeSsnPorch as '3SsnPorch' isn't a valid variable name
    ScreenPorch: Optional[int] = None
    PoolArea: Optional[int] = None
    MiscVal: Optional[int] = None
    MoSold: Optional[int] = None
    YrSold: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
                "LotArea": 10000,
                "OverallQual": 9,
                "OverallCond": 7,
                "YearBuilt": 2015,
                "YearRemodAdd": 2015
            }
        }

# Pydantic model for batch prediction input
class BatchPredictionInput(BaseModel):
    houses: List[HouseFeatures]

    class Config:
        schema_extra = {
            "example": {
                "houses": [
                    {
                        "LotArea": 10000,
                        "OverallQual": 9,
                        "OverallCond": 7, 
                        "YearBuilt": 2015,
                        "YearRemodAdd": 2015
                    },
                    {
                        "LotArea": 8000,
                        "OverallQual": 7,
                        "OverallCond": 5,
                        "YearBuilt": 2000,
                        "YearRemodAdd": 2010
                    }
                ]
            }
        }

# Pydantic model for prediction response
class PredictionResponse(BaseModel):
    predictions: List[float]
    formatted_predictions: List[str]

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(input_data: BatchPredictionInput):
    """
    Predict housing prices based on provided features.
    Unspecified features will use default values.
    """
    global model
    
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    # Convert input data to list of dictionaries
    input_dict_list = []
    for house in input_data.houses:
        # Convert to dictionary and rename fields to match what the model expects
        house_dict = house.dict(exclude_none=True)
        # Rename fields that had to be changed for Pydantic
        if "FirstFlrSF" in house_dict:
            house_dict["1stFlrSF"] = house_dict.pop("FirstFlrSF")
        if "SecondFlrSF" in house_dict:
            house_dict["2ndFlrSF"] = house_dict.pop("SecondFlrSF")
        if "ThreeSsnPorch" in house_dict:
            house_dict["3SsnPorch"] = house_dict.pop("ThreeSsnPorch")
        
        input_dict_list.append(house_dict)
    
    # Prepare input data with defaults
    prepared_data = prepare_input_data_with_defaults(input_dict_list)
    
    # Make predictions
    try:
        predictions = make_predictions(model, prepared_data)
        if predictions is None:
            raise HTTPException(status_code=500, detail="Model prediction failed")
        
        # Format predictions as currency strings
        formatted_predictions = [f"${np.expm1(pred):,.2f}" for pred in predictions]
        
        # Return predictions
        return PredictionResponse(
            predictions=predictions.tolist(),
            formatted_predictions=formatted_predictions
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "message": "Model is loaded and ready for predictions"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)