"""
FastAPI application for house price prediction service.
Provides RESTful endpoints for model predictions and health checks.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="Advanced ML API for predicting house prices using ensemble models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global variables for models
model = None
preprocessor = None


class HouseFeatures(BaseModel):
    """Input model for house features."""

    # Core features
    MSSubClass: int = Field(..., description="Type of dwelling")
    MSZoning: str = Field(..., description="General zoning classification")
    LotFrontage: Optional[float] = Field(None, description="Linear feet of street connected to property")
    LotArea: int = Field(..., description="Lot size in square feet")
    Street: str = Field("Pave", description="Type of road access")
    Alley: Optional[str] = Field(None, description="Type of alley access")
    LotShape: str = Field("Reg", description="General shape of property")
    LandContour: str = Field("Lvl", description="Flatness of the property")
    Utilities: str = Field("AllPub", description="Type of utilities available")
    LotConfig: str = Field("Inside", description="Lot configuration")
    LandSlope: str = Field("Gtl", description="Slope of property")
    Neighborhood: str = Field(..., description="Physical locations within Ames city limits")
    Condition1: str = Field("Norm", description="Proximity to main road or railroad")
    Condition2: str = Field("Norm", description="Proximity to main road or railroad (if a second is present)")
    BldgType: str = Field("1Fam", description="Type of dwelling")
    HouseStyle: str = Field(..., description="Style of dwelling")
    OverallQual: int = Field(..., ge=1, le=10, description="Overall material and finish quality")
    OverallCond: int = Field(..., ge=1, le=10, description="Overall condition rating")
    YearBuilt: int = Field(..., ge=1800, le=2024, description="Original construction date")
    YearRemodAdd: int = Field(..., ge=1800, le=2024, description="Remodel date")
    RoofStyle: str = Field("Gable", description="Type of roof")
    RoofMatl: str = Field("CompShg", description="Roof material")
    Exterior1st: str = Field(..., description="Exterior covering on house")
    Exterior2nd: str = Field(..., description="Exterior covering on house (if more than one material)")
    MasVnrType: Optional[str] = Field(None, description="Masonry veneer type")
    MasVnrArea: Optional[float] = Field(None, description="Masonry veneer area in square feet")
    ExterQual: str = Field("TA", description="Exterior material quality")
    ExterCond: str = Field("TA", description="Present condition of the material on the exterior")
    Foundation: str = Field(..., description="Type of foundation")

    # Basement features
    BsmtQual: Optional[str] = Field(None, description="Height of the basement")
    BsmtCond: Optional[str] = Field(None, description="General condition of the basement")
    BsmtExposure: Optional[str] = Field(None, description="Walkout or garden level basement walls")
    BsmtFinType1: Optional[str] = Field(None, description="Quality of basement finished area")
    BsmtFinSF1: Optional[int] = Field(None, description="Type 1 finished square feet")
    BsmtFinType2: Optional[str] = Field(None, description="Quality of second finished area")
    BsmtFinSF2: Optional[int] = Field(None, description="Type 2 finished square feet")
    BsmtUnfSF: Optional[int] = Field(None, description="Unfinished square feet of basement area")
    TotalBsmtSF: Optional[int] = Field(None, description="Total square feet of basement area")

    # Utilities
    Heating: str = Field("GasA", description="Type of heating")
    HeatingQC: str = Field("Ex", description="Heating quality and condition")
    CentralAir: str = Field("Y", description="Central air conditioning")
    Electrical: str = Field("SBrkr", description="Electrical system")

    # Interior features
    FirstFlrSF: int = Field(..., alias="1stFlrSF", description="First Floor square feet")
    SecondFlrSF: int = Field(0, alias="2ndFlrSF", description="Second floor square feet")
    LowQualFinSF: int = Field(0, description="Low quality finished square feet")
    GrLivArea: int = Field(..., description="Above grade living area square feet")
    BsmtFullBath: Optional[int] = Field(None, description="Basement full bathrooms")
    BsmtHalfBath: Optional[int] = Field(None, description="Basement half bathrooms")
    FullBath: int = Field(..., description="Full bathrooms above grade")
    HalfBath: int = Field(0, description="Half baths above grade")
    BedroomAbvGr: int = Field(..., description="Number of bedrooms above basement level")
    KitchenAbvGr: int = Field(1, description="Number of kitchens")
    KitchenQual: str = Field("TA", description="Kitchen quality")
    TotRmsAbvGrd: int = Field(..., description="Total rooms above grade")
    Functional: str = Field("Typ", description="Home functionality rating")
    Fireplaces: int = Field(0, description="Number of fireplaces")
    FireplaceQu: Optional[str] = Field(None, description="Fireplace quality")

    # Garage features
    GarageType: Optional[str] = Field(None, description="Garage location")
    GarageYrBlt: Optional[int] = Field(None, description="Year garage was built")
    GarageFinish: Optional[str] = Field(None, description="Interior finish of the garage")
    GarageCars: Optional[int] = Field(None, description="Size of garage in car capacity")
    GarageArea: Optional[int] = Field(None, description="Size of garage in square feet")
    GarageQual: Optional[str] = Field(None, description="Garage quality")
    GarageCond: Optional[str] = Field(None, description="Garage condition")
    PavedDrive: str = Field("Y", description="Paved driveway")

    # Outdoor features
    WoodDeckSF: int = Field(0, description="Wood deck area in square feet")
    OpenPorchSF: int = Field(0, description="Open porch area in square feet")
    EnclosedPorch: int = Field(0, description="Enclosed porch area in square feet")
    ThreeSsnPorch: int = Field(0, alias="3SsnPorch", description="Three season porch area in square feet")
    ScreenPorch: int = Field(0, description="Screen porch area in square feet")
    PoolArea: int = Field(0, description="Pool area in square feet")
    PoolQC: Optional[str] = Field(None, description="Pool quality")
    Fence: Optional[str] = Field(None, description="Fence quality")
    MiscFeature: Optional[str] = Field(None, description="Miscellaneous feature not covered in other categories")
    MiscVal: int = Field(0, description="Value of miscellaneous feature")
    MoSold: int = Field(..., ge=1, le=12, description="Month Sold")
    YrSold: int = Field(..., ge=1900, le=2024, description="Year Sold")
    SaleType: str = Field("WD", description="Type of sale")
    SaleCondition: str = Field("Normal", description="Condition of sale")

    @validator("YearRemodAdd")
    def validate_remodel_year(cls, v, values):
        if "YearBuilt" in values and v < values["YearBuilt"]:
            raise ValueError("Remodel year cannot be before build year")
        return v

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "MSSubClass": 60,
                "MSZoning": "RL",
                "LotArea": 8450,
                "Neighborhood": "CollgCr",
                "HouseStyle": "2Story",
                "OverallQual": 7,
                "OverallCond": 5,
                "YearBuilt": 2003,
                "YearRemodAdd": 2003,
                "Exterior1st": "VinylSd",
                "Exterior2nd": "VinylSd",
                "Foundation": "PConc",
                "1stFlrSF": 856,
                "2ndFlrSF": 854,
                "GrLivArea": 1710,
                "FullBath": 2,
                "BedroomAbvGr": 3,
                "TotRmsAbvGrd": 8,
                "MoSold": 2,
                "YrSold": 2008,
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    predicted_price: float = Field(..., description="Predicted house price in USD")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="95% confidence interval")
    model_used: str = Field(..., description="Name of the model used for prediction")
    prediction_id: str = Field(..., description="Unique identifier for this prediction")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


# Dependency to get model
def get_model():
    """Dependency to ensure model is loaded."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model, preprocessor


@app.on_event("startup")
async def load_model():
    """Load the trained model and preprocessor on startup."""
    global model

    try:
        # Try to load the best stacking ensemble model
        model_path = "models/stacking_ensemble_model.joblib"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("Loaded stacking ensemble model")
        else:
            # Fallback to any available model
            models_dir = Path("models")
            if models_dir.exists():
                model_files = list(models_dir.glob("*_model.joblib"))
                if model_files:
                    model_path = model_files[0]
                    model = joblib.load(model_path)
                    logger.info(f"Loaded fallback model: {model_path}")
                else:
                    logger.warning("No trained models found")
            else:
                logger.warning("Models directory not found")

        # Load preprocessor (would be saved separately in real implementation)
        # For now, we'll create a basic preprocessing function
        logger.info("Model loading complete")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")


def preprocess_input(features: HouseFeatures) -> pd.DataFrame:
    """Preprocess input features for prediction."""
    # Convert Pydantic model to dictionary
    feature_dict = features.dict(by_alias=True)

    # Create DataFrame
    df = pd.DataFrame([feature_dict])

    # Basic preprocessing (in real implementation, use the trained preprocessor)
    # Handle missing values
    df = df.fillna(0)

    # Ensure all required columns are present and in correct order
    # This would match the training data structure

    return df


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint returning API information."""
    return HealthResponse(status="healthy", model_loaded=model is not None, version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", model_loaded=model is not None, version="1.0.0")


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeatures, model_info: tuple = Depends(get_model)):
    """
    Predict house price based on features.

    Returns predicted price with confidence information.
    """
    try:
        model, preprocessor = model_info

        # Preprocess input
        df = preprocess_input(features)

        # Make prediction
        prediction = model.predict(df)[0]

        # Generate prediction ID
        import uuid

        prediction_id = str(uuid.uuid4())

        # For ensemble models, we could calculate confidence intervals
        # based on individual model predictions
        confidence_interval = None
        if hasattr(model, "fitted_base_models"):
            # Get predictions from all base models
            base_predictions = []
            for base_model in model.fitted_base_models.values():
                base_pred = base_model.predict(df)[0]
                base_predictions.append(base_pred)

            # Calculate confidence interval (simple approach)
            std_dev = np.std(base_predictions)
            confidence_interval = {"lower": float(prediction - 1.96 * std_dev), "upper": float(prediction + 1.96 * std_dev)}

        return PredictionResponse(
            predicted_price=float(prediction),
            confidence_interval=confidence_interval,
            model_used=type(model).__name__,
            prediction_id=prediction_id,
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info")
async def get_model_info(model_info: tuple = Depends(get_model)):
    """Get information about the loaded model."""
    model, preprocessor = model_info

    info = {
        "model_type": type(model).__name__,
        "sklearn_version": "1.3.2",  # Update based on requirements
    }

    # Add ensemble-specific information
    if hasattr(model, "fitted_base_models"):
        info["base_models"] = list(model.fitted_base_models.keys())
        info["model_weights"] = model.get_model_weights()

    return info


@app.get("/model/feature-importance")
async def get_feature_importance(model_info: tuple = Depends(get_model)):
    """Get feature importance from the model."""
    model, preprocessor = model_info

    if hasattr(model, "feature_importance_df") and model.feature_importance_df is not None:
        # Return top 20 most important features
        top_features = (
            model.feature_importance_df.groupby("feature")["importance"].mean().sort_values(ascending=False).head(20)
        )

        return {"feature_importance": top_features.to_dict(), "total_features": len(top_features)}
    else:
        return {"message": "Feature importance not available for this model"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
