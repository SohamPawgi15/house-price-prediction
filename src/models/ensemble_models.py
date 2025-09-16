"""
Advanced ensemble modeling pipeline for house price prediction.
Implements multiple base models and stacking meta-learner.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Advanced stacking ensemble with multiple base models and meta-learner.
    """

    def __init__(
        self,
        base_models: Dict[str, BaseEstimator],
        meta_model: BaseEstimator = None,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.base_models = base_models
        self.meta_model = meta_model or Ridge(alpha=0.1)
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.fitted_base_models = {}
        self.fitted_meta_model = None
        self.feature_importance_df = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StackingEnsemble":
        """Fit the stacking ensemble."""
        logger.info("Training stacking ensemble...")

        # Generate meta-features using cross-validation
        meta_features = self._generate_meta_features(X, y)

        # Train final base models on full dataset
        logger.info("Training final base models...")
        for name, model in self.base_models.items():
            logger.info(f"Training {name}...")
            fitted_model = clone(model)
            fitted_model.fit(X, y)
            self.fitted_base_models[name] = fitted_model

        # Train meta-model on meta-features
        logger.info("Training meta-model...")
        self.fitted_meta_model = clone(self.meta_model)
        self.fitted_meta_model.fit(meta_features, y)

        # Calculate feature importance
        self._calculate_feature_importance(X)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the stacking ensemble."""
        # Generate meta-features from base models
        meta_features = np.column_stack([model.predict(X) for model in self.fitted_base_models.values()])

        # Meta-model prediction
        return self.fitted_meta_model.predict(meta_features)

    def _generate_meta_features(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Generate meta-features using cross-validation."""
        logger.info("Generating meta-features with cross-validation...")

        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            logger.info(f"Processing fold {fold_idx + 1}/{self.cv_folds}")

            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train = y.iloc[train_idx]

            for model_idx, (name, model) in enumerate(self.base_models.items()):
                # Train model on fold training data
                fitted_model = clone(model)
                fitted_model.fit(X_fold_train, y_fold_train)

                # Predict on fold validation data
                predictions = fitted_model.predict(X_fold_val)
                meta_features[val_idx, model_idx] = predictions

        return meta_features

    def _calculate_feature_importance(self, X: pd.DataFrame) -> None:
        """Calculate feature importance from tree-based models."""
        importance_data = []

        for name, model in self.fitted_base_models.items():
            if hasattr(model, "feature_importances_"):
                for idx, importance in enumerate(model.feature_importances_):
                    importance_data.append(
                        {
                            "model": name,
                            "feature": X.columns[idx] if hasattr(X, "columns") else f"feature_{idx}",
                            "importance": importance,
                        }
                    )

        if importance_data:
            self.feature_importance_df = pd.DataFrame(importance_data)

    def get_model_weights(self) -> Dict[str, float]:
        """Get the weights of base models from meta-model."""
        if hasattr(self.fitted_meta_model, "coef_"):
            weights = self.fitted_meta_model.coef_
            return dict(zip(self.base_models.keys(), weights))
        return {}


class ModelTrainer:
    """
    Complete model training pipeline with multiple algorithms.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.model_scores = {}

    def get_base_models(self) -> Dict[str, BaseEstimator]:
        """Get all base models for ensemble."""
        models = {
            "linear_regression": LinearRegression(),
            "ridge": Ridge(alpha=10.0, random_state=self.random_state),
            "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.1, random_state=self.random_state, max_iter=5000),
            "random_forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "xgboost": xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0,
            ),
            "lightgbm": lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=-1,
            ),
            "catboost": CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                bootstrap_type="Bernoulli",
                subsample=0.8,
                random_seed=self.random_state,
                verbose=False,
            ),
        }

        return models

    def train_individual_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, BaseEstimator]:
        """Train individual models and evaluate performance."""
        logger.info("Training individual models...")

        base_models = self.get_base_models()
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for name, model in base_models.items():
            logger.info(f"Training {name}...")

            # Cross-validation scoring
            cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="neg_root_mean_squared_error", n_jobs=-1)

            # Fit model on full training data
            model.fit(X_train, y_train)
            self.models[name] = model

            # Store scores
            self.model_scores[name] = {"cv_rmse": -cv_scores.mean(), "cv_rmse_std": cv_scores.std()}

            logger.info(f"{name} - CV RMSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return self.models

    def train_stacking_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series) -> StackingEnsemble:
        """Train stacking ensemble model."""
        logger.info("Training stacking ensemble...")

        base_models = self.get_base_models()

        # Create stacking ensemble
        stacking_model = StackingEnsemble(
            base_models=base_models, meta_model=Ridge(alpha=0.1), cv_folds=5, random_state=self.random_state
        )

        # Train the ensemble
        stacking_model.fit(X_train, y_train)

        # Evaluate ensemble
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            stacking_model, X_train, y_train, cv=kfold, scoring="neg_root_mean_squared_error", n_jobs=-1
        )

        self.model_scores["stacking_ensemble"] = {"cv_rmse": -cv_scores.mean(), "cv_rmse_std": cv_scores.std()}

        logger.info(f"Stacking Ensemble - CV RMSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Store as best model
        self.models["stacking_ensemble"] = stacking_model
        self.best_model = stacking_model

        return stacking_model

    def evaluate_model(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test set."""
        predictions = model.predict(X_test)

        return {
            "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
        }

    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all model performances."""
        summary_data = []

        for name, scores in self.model_scores.items():
            summary_data.append({"Model": name, "CV_RMSE": scores["cv_rmse"], "CV_RMSE_Std": scores["cv_rmse_std"]})

        df = pd.DataFrame(summary_data)
        return df.sort_values("CV_RMSE")

    def save_models(self, model_dir: str = "models/") -> None:
        """Save trained models to disk."""
        import os

        os.makedirs(model_dir, exist_ok=True)

        for name, model in self.models.items():
            model_path = os.path.join(model_dir, f"{name}_model.joblib")
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")

    def load_model(self, model_path: str) -> BaseEstimator:
        """Load a saved model."""
        return joblib.load(model_path)


def train_complete_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[ModelTrainer, StackingEnsemble]:
    """Train the complete modeling pipeline."""
    logger.info("Starting complete model training pipeline...")

    # Initialize trainer
    trainer = ModelTrainer(random_state=42)

    # Train individual models
    trainer.train_individual_models(X_train, y_train)

    # Train stacking ensemble
    stacking_model = trainer.train_stacking_ensemble(X_train, y_train)

    # Print model summary
    summary = trainer.get_model_summary()
    logger.info("Model Performance Summary:")
    logger.info(f"\n{summary.to_string(index=False)}")

    return trainer, stacking_model


if __name__ == "__main__":
    # Test the modeling pipeline
    from src.data.preprocessor import load_and_preprocess_data

    # Load and preprocess data
    train_path = "house-prices-advanced-regression-techniques/train.csv"
    test_path = "house-prices-advanced-regression-techniques/test.csv"

    X_train, X_test, y_train = load_and_preprocess_data(train_path, test_path)

    # Train models
    trainer, best_model = train_complete_pipeline(X_train, y_train)

    # Save models
    trainer.save_models()

    print(f"Training completed. Best model: {type(best_model).__name__}")
    print(f"Model weights: {best_model.get_model_weights()}")
