"""
Unit tests for ensemble models and training pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
from sklearn.metrics import mean_squared_error

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from models.ensemble_models import StackingEnsemble, ModelTrainer, train_complete_pipeline


class TestStackingEnsemble:
    """Test suite for StackingEnsemble."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "feature3": np.random.normal(0, 1, 100),
            }
        )
        y = pd.Series(X["feature1"] * 2 + X["feature2"] * 3 + np.random.normal(0, 0.1, 100))
        return X, y

    @pytest.fixture
    def base_models(self):
        """Create simple base models for testing."""
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.ensemble import RandomForestRegressor

        return {
            "linear": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
            "rf": RandomForestRegressor(n_estimators=10, random_state=42),
        }

    def test_init(self, base_models):
        """Test StackingEnsemble initialization."""
        ensemble = StackingEnsemble(base_models)

        assert ensemble.base_models == base_models
        assert ensemble.cv_folds == 5
        assert ensemble.random_state == 42
        assert isinstance(ensemble.meta_model, type(ensemble.meta_model))

    def test_fit(self, base_models, sample_data):
        """Test fitting the stacking ensemble."""
        X, y = sample_data
        ensemble = StackingEnsemble(base_models, cv_folds=3)

        # Fit the ensemble
        ensemble.fit(X, y)

        # Check that base models are fitted
        assert len(ensemble.fitted_base_models) == len(base_models)
        for name in base_models.keys():
            assert name in ensemble.fitted_base_models

        # Check that meta-model is fitted
        assert ensemble.fitted_meta_model is not None

    def test_predict(self, base_models, sample_data):
        """Test prediction with stacking ensemble."""
        X, y = sample_data
        ensemble = StackingEnsemble(base_models, cv_folds=3)

        # Fit and predict
        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        # Check prediction shape and type
        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)

        # Check that predictions are reasonable
        assert not np.isnan(predictions).any()
        assert np.isfinite(predictions).all()

    def test_generate_meta_features(self, base_models, sample_data):
        """Test meta-feature generation."""
        X, y = sample_data
        ensemble = StackingEnsemble(base_models, cv_folds=3)

        meta_features = ensemble._generate_meta_features(X, y)

        # Check meta-features shape
        assert meta_features.shape == (len(X), len(base_models))

        # Check that meta-features are valid
        assert not np.isnan(meta_features).any()
        assert np.isfinite(meta_features).all()

    def test_get_model_weights(self, base_models, sample_data):
        """Test getting model weights from meta-model."""
        X, y = sample_data
        ensemble = StackingEnsemble(base_models, cv_folds=3)

        ensemble.fit(X, y)
        weights = ensemble.get_model_weights()

        # Check that weights are returned
        assert isinstance(weights, dict)
        assert len(weights) == len(base_models)

        # Check that all base models have weights
        for name in base_models.keys():
            assert name in weights

    def test_feature_importance_calculation(self, base_models, sample_data):
        """Test feature importance calculation."""
        X, y = sample_data
        ensemble = StackingEnsemble(base_models, cv_folds=3)

        ensemble.fit(X, y)

        # Check that feature importance is calculated for tree-based models
        if ensemble.feature_importance_df is not None:
            assert isinstance(ensemble.feature_importance_df, pd.DataFrame)
            assert "model" in ensemble.feature_importance_df.columns
            assert "feature" in ensemble.feature_importance_df.columns
            assert "importance" in ensemble.feature_importance_df.columns


class TestModelTrainer:
    """Test suite for ModelTrainer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 50),
                "feature2": np.random.normal(0, 1, 50),
                "feature3": np.random.normal(0, 1, 50),
            }
        )
        y = pd.Series(X["feature1"] * 2 + X["feature2"] * 3 + np.random.normal(0, 0.1, 50))
        return X, y

    def test_init(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(random_state=123)

        assert trainer.random_state == 123
        assert trainer.models == {}
        assert trainer.best_model is None
        assert trainer.model_scores == {}

    def test_get_base_models(self):
        """Test getting base models."""
        trainer = ModelTrainer()
        models = trainer.get_base_models()

        # Check that all expected models are present
        expected_models = ["linear_regression", "ridge", "elastic_net", "random_forest", "xgboost", "lightgbm", "catboost"]

        for model_name in expected_models:
            assert model_name in models

        # Check that models are sklearn-compatible
        for model in models.values():
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")

    @patch("sklearn.model_selection.cross_val_score")
    def test_train_individual_models(self, mock_cv_score, sample_data):
        """Test training individual models."""
        X, y = sample_data
        trainer = ModelTrainer()

        # Mock cross-validation scores
        mock_cv_score.return_value = np.array([-0.1, -0.2, -0.15, -0.18, -0.12])

        # Train only a subset for testing speed
        with patch.object(trainer, "get_base_models") as mock_get_models:
            from sklearn.linear_model import LinearRegression, Ridge

            mock_get_models.return_value = {"linear": LinearRegression(), "ridge": Ridge()}

            models = trainer.train_individual_models(X, y)

            # Check that models are trained
            assert len(models) == 2
            assert "linear" in models
            assert "ridge" in models

            # Check that scores are recorded
            assert len(trainer.model_scores) == 2
            assert "linear" in trainer.model_scores
            assert "ridge" in trainer.model_scores

    def test_train_stacking_ensemble(self, sample_data):
        """Test training stacking ensemble."""
        X, y = sample_data
        trainer = ModelTrainer()

        # Use simple models for testing
        with patch.object(trainer, "get_base_models") as mock_get_models:
            from sklearn.linear_model import LinearRegression, Ridge

            mock_get_models.return_value = {"linear": LinearRegression(), "ridge": Ridge()}

            ensemble = trainer.train_stacking_ensemble(X, y)

            # Check that ensemble is created and trained
            assert isinstance(ensemble, StackingEnsemble)
            assert ensemble in trainer.models.values()
            assert trainer.best_model == ensemble

            # Check that ensemble can make predictions
            predictions = ensemble.predict(X)
            assert len(predictions) == len(y)

    def test_evaluate_model(self, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        trainer = ModelTrainer()

        # Create and fit a simple model
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(X, y)

        # Evaluate model
        metrics = trainer.evaluate_model(model, X, y)

        # Check that all metrics are calculated
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

        # Check that metrics are reasonable
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["r2"] <= 1

    def test_get_model_summary(self, sample_data):
        """Test getting model summary."""
        X, y = sample_data
        trainer = ModelTrainer()

        # Add some mock scores
        trainer.model_scores = {
            "model1": {"cv_rmse": 0.15, "cv_rmse_std": 0.02},
            "model2": {"cv_rmse": 0.12, "cv_rmse_std": 0.01},
        }

        summary = trainer.get_model_summary()

        # Check summary format
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert "Model" in summary.columns
        assert "CV_RMSE" in summary.columns

        # Check that summary is sorted by RMSE
        assert summary.iloc[0]["CV_RMSE"] <= summary.iloc[1]["CV_RMSE"]

    @patch("joblib.dump")
    @patch("os.makedirs")
    def test_save_models(self, mock_makedirs, mock_dump):
        """Test saving models."""
        trainer = ModelTrainer()

        # Add mock models
        from sklearn.linear_model import LinearRegression

        trainer.models = {"model1": LinearRegression(), "model2": LinearRegression()}

        trainer.save_models("test_dir/")

        # Check that directory is created
        mock_makedirs.assert_called_once_with("test_dir/", exist_ok=True)

        # Check that models are saved
        assert mock_dump.call_count == 2

    @patch("joblib.load")
    def test_load_model(self, mock_load):
        """Test loading model."""
        trainer = ModelTrainer()
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        loaded_model = trainer.load_model("test_path.joblib")

        mock_load.assert_called_once_with("test_path.joblib")
        assert loaded_model == mock_model


class TestTrainingPipeline:
    """Test the complete training pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 30),
                "feature2": np.random.normal(0, 1, 30),
                "feature3": np.random.normal(0, 1, 30),
            }
        )
        y = pd.Series(X["feature1"] * 2 + X["feature2"] * 3 + np.random.normal(0, 0.1, 30))
        return X, y

    def test_train_complete_pipeline(self, sample_data):
        """Test the complete training pipeline."""
        X, y = sample_data

        # Use simplified models for testing
        with patch("src.models.ensemble_models.ModelTrainer.get_base_models") as mock_get_models:
            from sklearn.linear_model import LinearRegression, Ridge

            mock_get_models.return_value = {"linear": LinearRegression(), "ridge": Ridge()}

            trainer, best_model = train_complete_pipeline(X, y)

            # Check that trainer is returned
            assert isinstance(trainer, ModelTrainer)

            # Check that best model is stacking ensemble
            assert isinstance(best_model, StackingEnsemble)

            # Check that models are trained
            assert len(trainer.models) > 0

            # Check that best model can make predictions
            predictions = best_model.predict(X)
            assert len(predictions) == len(y)


@pytest.mark.integration
class TestModelsIntegration:
    """Integration tests for models with real-like data."""

    def test_model_performance_realistic(self):
        """Test model performance with realistic data."""
        # Create more realistic house price data
        np.random.seed(42)
        n_samples = 200

        X = pd.DataFrame(
            {
                "GrLivArea": np.random.normal(1500, 500, n_samples),
                "OverallQual": np.random.randint(1, 11, n_samples),
                "YearBuilt": np.random.randint(1950, 2020, n_samples),
                "LotArea": np.random.normal(9000, 3000, n_samples),
                "TotalBsmtSF": np.random.normal(1000, 400, n_samples),
            }
        )

        # Create realistic target based on features
        y = (
            X["GrLivArea"] * 50
            + X["OverallQual"] * 10000
            + (2020 - X["YearBuilt"]) * -100
            + X["LotArea"] * 5
            + X["TotalBsmtSF"] * 30
            + np.random.normal(0, 5000, n_samples)
        )

        y = pd.Series(y)

        # Train simplified pipeline
        with patch("src.models.ensemble_models.ModelTrainer.get_base_models") as mock_get_models:
            from sklearn.linear_model import LinearRegression, Ridge
            from sklearn.ensemble import RandomForestRegressor

            mock_get_models.return_value = {
                "linear": LinearRegression(),
                "ridge": Ridge(),
                "rf": RandomForestRegressor(n_estimators=10, random_state=42),
            }

            trainer, best_model = train_complete_pipeline(X, y)

            # Test performance
            predictions = best_model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, predictions))

            # Should achieve reasonable performance
            assert rmse < y.std()  # RMSE should be less than target standard deviation

            # Check that model weights are available
            weights = best_model.get_model_weights()
            assert isinstance(weights, dict)
            assert len(weights) > 0


if __name__ == "__main__":
    pytest.main([__file__])
