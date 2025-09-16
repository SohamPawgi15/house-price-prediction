"""
Unit tests for FastAPI application.
"""

import pytest
import json
import sys
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from api.main import app, HouseFeatures, PredictionResponse, HealthResponse


class TestAPI:
    """Test suite for FastAPI application."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_house_features(self):
        """Sample house features for testing."""
        return {
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

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

        # Check values
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Check response matches HealthResponse model
        health_response = HealthResponse(**data)
        assert health_response.status == "healthy"
        assert health_response.version == "1.0.0"

    @patch("src.api.main.model")
    @patch("src.api.main.preprocessor")
    def test_predict_endpoint_success(self, mock_preprocessor, mock_model, client, sample_house_features):
        """Test successful prediction."""
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = [250000.0]
        mock_model_instance.fitted_base_models = {"model1": MagicMock(), "model2": MagicMock()}
        mock_model_instance.fitted_base_models["model1"].predict.return_value = [245000.0]
        mock_model_instance.fitted_base_models["model2"].predict.return_value = [255000.0]
        mock_model_instance.get_model_weights.return_value = {"model1": 0.6, "model2": 0.4}

        # Mock globals
        mock_model = mock_model_instance
        mock_preprocessor = MagicMock()

        with patch("src.api.main.get_model") as mock_get_model:
            mock_get_model.return_value = (mock_model_instance, mock_preprocessor)

            response = client.post("/predict", json=sample_house_features)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "predicted_price" in data
        assert "confidence_interval" in data
        assert "model_used" in data
        assert "prediction_id" in data

        # Check values
        assert data["predicted_price"] == 250000.0
        assert data["confidence_interval"] is not None
        assert "lower" in data["confidence_interval"]
        assert "upper" in data["confidence_interval"]

    def test_predict_endpoint_missing_fields(self, client):
        """Test prediction with missing required fields."""
        incomplete_features = {
            "MSSubClass": 60,
            "MSZoning": "RL",
            # Missing required fields
        }

        response = client.post("/predict", json=incomplete_features)

        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_invalid_values(self, client):
        """Test prediction with invalid field values."""
        invalid_features = {
            "MSSubClass": 60,
            "MSZoning": "RL",
            "LotArea": 8450,
            "Neighborhood": "CollgCr",
            "HouseStyle": "2Story",
            "OverallQual": 15,  # Invalid: should be 1-10
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

        response = client.post("/predict", json=invalid_features)

        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_no_model(self, client, sample_house_features):
        """Test prediction when no model is loaded."""
        with patch("src.api.main.get_model") as mock_get_model:
            mock_get_model.side_effect = Exception("Model not loaded")

            response = client.post("/predict", json=sample_house_features)

        assert response.status_code == 500

    @patch("src.api.main.model")
    @patch("src.api.main.preprocessor")
    def test_model_info_endpoint(self, mock_preprocessor, mock_model, client):
        """Test model info endpoint."""
        # Setup mock
        mock_model_instance = MagicMock()
        mock_model_instance.__class__.__name__ = "StackingEnsemble"
        mock_model_instance.fitted_base_models = {"model1": MagicMock(), "model2": MagicMock()}
        mock_model_instance.get_model_weights.return_value = {"model1": 0.6, "model2": 0.4}

        with patch("src.api.main.get_model") as mock_get_model:
            mock_get_model.return_value = (mock_model_instance, mock_preprocessor)

            response = client.get("/model/info")

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "model_type" in data
        assert "sklearn_version" in data
        assert "base_models" in data
        assert "model_weights" in data

        # Check values
        assert data["model_type"] == "StackingEnsemble"
        assert isinstance(data["base_models"], list)
        assert isinstance(data["model_weights"], dict)

    @patch("src.api.main.model")
    @patch("src.api.main.preprocessor")
    def test_feature_importance_endpoint(self, mock_preprocessor, mock_model, client):
        """Test feature importance endpoint."""
        # Setup mock with feature importance
        import pandas as pd

        mock_model_instance = MagicMock()
        mock_model_instance.feature_importance_df = pd.DataFrame(
            {
                "feature": ["feature1", "feature2", "feature3"] * 3,
                "importance": [0.5, 0.3, 0.2, 0.4, 0.35, 0.25, 0.6, 0.25, 0.15],
                "model": ["model1"] * 3 + ["model2"] * 3 + ["model3"] * 3,
            }
        )

        with patch("src.api.main.get_model") as mock_get_model:
            mock_get_model.return_value = (mock_model_instance, mock_preprocessor)

            response = client.get("/model/feature-importance")

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "feature_importance" in data
        assert "total_features" in data

        # Check that feature importance is returned
        assert isinstance(data["feature_importance"], dict)
        assert data["total_features"] > 0

    @patch("src.api.main.model")
    @patch("src.api.main.preprocessor")
    def test_feature_importance_endpoint_no_importance(self, mock_preprocessor, mock_model, client):
        """Test feature importance endpoint when no importance available."""
        # Setup mock without feature importance
        mock_model_instance = MagicMock()
        mock_model_instance.feature_importance_df = None

        with patch("src.api.main.get_model") as mock_get_model:
            mock_get_model.return_value = (mock_model_instance, mock_preprocessor)

            response = client.get("/model/feature-importance")

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "message" in data
        assert "Feature importance not available" in data["message"]


class TestHouseFeaturesModel:
    """Test the HouseFeatures Pydantic model."""

    def test_valid_house_features(self):
        """Test valid house features."""
        valid_data = {
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

        # Should not raise error
        features = HouseFeatures(**valid_data)

        assert features.MSSubClass == 60
        assert features.OverallQual == 7
        assert features.YearBuilt == 2003

    def test_invalid_overall_qual(self):
        """Test invalid OverallQual value."""
        invalid_data = {
            "MSSubClass": 60,
            "MSZoning": "RL",
            "LotArea": 8450,
            "Neighborhood": "CollgCr",
            "HouseStyle": "2Story",
            "OverallQual": 15,  # Invalid: should be 1-10
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

        with pytest.raises(ValueError):
            HouseFeatures(**invalid_data)

    def test_invalid_remodel_year(self):
        """Test invalid remodel year (before build year)."""
        invalid_data = {
            "MSSubClass": 60,
            "MSZoning": "RL",
            "LotArea": 8450,
            "Neighborhood": "CollgCr",
            "HouseStyle": "2Story",
            "OverallQual": 7,
            "OverallCond": 5,
            "YearBuilt": 2003,
            "YearRemodAdd": 2000,  # Invalid: before build year
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

        with pytest.raises(ValueError):
            HouseFeatures(**invalid_data)

    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        minimal_data = {
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
            "GrLivArea": 1710,
            "FullBath": 2,
            "BedroomAbvGr": 3,
            "TotRmsAbvGrd": 8,
            "MoSold": 2,
            "YrSold": 2008,
            # Many optional fields omitted
        }

        # Should not raise error
        features = HouseFeatures(**minimal_data)

        # Check that optional fields have default values
        assert features.SecondFlrSF == 0  # Should use alias "2ndFlrSF"
        assert features.HalfBath == 0
        assert features.Fireplaces == 0


class TestResponseModels:
    """Test response Pydantic models."""

    def test_prediction_response(self):
        """Test PredictionResponse model."""
        data = {
            "predicted_price": 250000.0,
            "confidence_interval": {"lower": 240000.0, "upper": 260000.0},
            "model_used": "StackingEnsemble",
            "prediction_id": "test-id-123",
        }

        response = PredictionResponse(**data)

        assert response.predicted_price == 250000.0
        assert response.confidence_interval["lower"] == 240000.0
        assert response.model_used == "StackingEnsemble"
        assert response.prediction_id == "test-id-123"

    def test_health_response(self):
        """Test HealthResponse model."""
        data = {"status": "healthy", "model_loaded": True, "version": "1.0.0"}

        response = HealthResponse(**data)

        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.version == "1.0.0"


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for the API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_full_prediction_workflow(self, client):
        """Test complete prediction workflow."""
        # This would test with actual model if available
        sample_features = {
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

        # Check health first
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # If model is not loaded, the prediction should fail gracefully
        prediction_response = client.post("/predict", json=sample_features)

        # Should either succeed (200) or fail gracefully (503/500)
        assert prediction_response.status_code in [200, 500, 503]


if __name__ == "__main__":
    pytest.main([__file__])
