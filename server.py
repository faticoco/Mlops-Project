from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd

# Load the trained model
with open("models/best_model.pkl", "rb") as f:
    model_package = pickle.load(f)
    model = model_package["model"]

app = FastAPI()

# Define input schema
class HouseFeatures(BaseModel):
    LotArea: int
    YearBuilt: int
    FirstFlrSF: int
    SecondFlrSF: int
    TotalBsmtSF: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    TotRmsAbvGrd: int
    Fireplaces: int
    GarageArea: int
    PoolArea: int
    Neighborhood: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int

class InferenceRequest(BaseModel):
    data: List[HouseFeatures]

# Define output schema
class Prediction(BaseModel):
    id: int
    SalePrice: float

class InferenceResponse(BaseModel):
    predictions: List[Prediction]

@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([house.dict() for house in request.data])
    
    # Make predictions
    predictions = model.predict(input_data)
    
    # Prepare response
    response = {
        "predictions": [
            {"id": idx + 1, "SalePrice": float(pred)}
            for idx, pred in enumerate(predictions)
        ]
    }
    return response