"""
Model evaluation and performance metrics module.
Provides comprehensive evaluation tools for regression models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from sklearn.model_selection import learning_curve, validation_curve
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation class for regression tasks.
    """

    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.evaluation_results = {}

    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Evaluate all models and return performance summary."""
        logger.info("Evaluating all models...")

        results = []

        for name, model in self.models.items():
            try:
                metrics = self.evaluate_single_model(model, X_test, y_test)
                metrics["Model"] = name
                results.append(metrics)
                self.evaluation_results[name] = metrics

            except Exception as e:
                logger.error(f"Error evaluating model {name}: {str(e)}")
                continue

        if not results:
            logger.warning("No models could be evaluated")
            return pd.DataFrame()

        # Create DataFrame and sort by RMSE
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("RMSE")

        # Reorder columns
        cols = ["Model", "RMSE", "MAE", "R2", "MAPE", "Median_AE"]
        results_df = results_df[cols]

        return results_df

    def evaluate_single_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate a single model and return metrics."""
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        median_ae = median_absolute_error(y_test, y_pred)

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Additional metrics
        std_residuals = np.std(y_test - y_pred)
        max_error = np.max(np.abs(y_test - y_pred))

        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape,
            "Median_AE": median_ae,
            "Std_Residuals": std_residuals,
            "Max_Error": max_error,
        }

    def plot_predictions_vs_actual(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model",
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """Plot predictions vs actual values."""
        y_pred = model.predict(X_test)

        plt.figure(figsize=figsize)

        # Scatter plot
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)

        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

        # Add metrics to plot
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        plt.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}\nRMSE = {rmse:.3f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{model_name}: Predictions vs Actual Values")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_residuals(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model",
        figsize: Tuple[int, int] = (15, 5),
    ) -> None:
        """Plot residual analysis."""
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
        axes[0].axhline(y=0, color="r", linestyle="--")
        axes[0].set_xlabel("Predicted Values")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Predicted")
        axes[0].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor="black")
        axes[1].set_xlabel("Residuals")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Distribution of Residuals")
        axes[1].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title("Q-Q Plot")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(f"{model_name}: Residual Analysis", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str = "Model",
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """Plot feature importance for tree-based models."""
        if not hasattr(model, "feature_importances_"):
            logger.warning(f"Model {model_name} does not have feature_importances_ attribute")
            return

        # Get feature importance
        importance = model.feature_importances_

        # Create DataFrame and sort
        importance_df = pd.DataFrame({"feature": feature_names[: len(importance)], "importance": importance}).sort_values(
            "importance", ascending=False
        )

        # Plot top N features
        top_features = importance_df.head(top_n)

        plt.figure(figsize=figsize)
        sns.barplot(data=top_features, x="importance", y="feature", palette="viridis")
        plt.title(f"{model_name}: Top {top_n} Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()

    def plot_learning_curves(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "Model",
        cv: int = 5,
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        """Plot learning curves to analyze bias-variance tradeoff."""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring="neg_root_mean_squared_error"
        )

        # Convert to positive RMSE
        train_scores = -train_scores
        val_scores = -val_scores

        # Calculate means and standard deviations
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=figsize)

        # Plot learning curves
        plt.plot(train_sizes, train_mean, "o-", color="blue", label="Training RMSE")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")

        plt.plot(train_sizes, val_mean, "o-", color="red", label="Validation RMSE")
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="red")

        plt.xlabel("Training Set Size")
        plt.ylabel("RMSE")
        plt.title(f"{model_name}: Learning Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def compare_models_performance(self, evaluation_results: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Create comprehensive model comparison plots."""
        if evaluation_results.empty:
            logger.warning("No evaluation results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # RMSE comparison
        axes[0, 0].barh(evaluation_results["Model"], evaluation_results["RMSE"], color="skyblue")
        axes[0, 0].set_xlabel("RMSE")
        axes[0, 0].set_title("Root Mean Squared Error")
        axes[0, 0].grid(True, alpha=0.3)

        # R² comparison
        axes[0, 1].barh(evaluation_results["Model"], evaluation_results["R2"], color="lightgreen")
        axes[0, 1].set_xlabel("R² Score")
        axes[0, 1].set_title("R² Score")
        axes[0, 1].grid(True, alpha=0.3)

        # MAE comparison
        axes[1, 0].barh(evaluation_results["Model"], evaluation_results["MAE"], color="salmon")
        axes[1, 0].set_xlabel("MAE")
        axes[1, 0].set_title("Mean Absolute Error")
        axes[1, 0].grid(True, alpha=0.3)

        # MAPE comparison
        axes[1, 1].barh(evaluation_results["Model"], evaluation_results["MAPE"], color="gold")
        axes[1, 1].set_xlabel("MAPE (%)")
        axes[1, 1].set_title("Mean Absolute Percentage Error")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def generate_evaluation_report(self, X_test: pd.DataFrame, y_test: pd.Series, output_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report."""
        # Evaluate all models
        results_df = self.evaluate_all_models(X_test, y_test)

        if results_df.empty:
            return "No models could be evaluated."

        # Generate report
        report = []
        report.append("=" * 80)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 40)
        report.append(results_df.to_string(index=False, float_format="%.4f"))
        report.append("")

        # Best model analysis
        best_model_name = results_df.iloc[0]["Model"]
        best_metrics = results_df.iloc[0]

        report.append("BEST MODEL ANALYSIS")
        report.append("-" * 40)
        report.append(f"Best Model: {best_model_name}")
        report.append(f"RMSE: {best_metrics['RMSE']:.4f}")
        report.append(f"R² Score: {best_metrics['R2']:.4f}")
        report.append(f"MAE: {best_metrics['MAE']:.4f}")
        report.append(f"MAPE: {best_metrics['MAPE']:.2f}%")
        report.append("")

        # Model ranking
        report.append("MODEL RANKING")
        report.append("-" * 40)
        for i, row in results_df.iterrows():
            rank = i + 1
            report.append(f"{rank}. {row['Model']} (RMSE: {row['RMSE']:.4f})")
        report.append("")

        # Performance insights
        report.append("PERFORMANCE INSIGHTS")
        report.append("-" * 40)

        # Best R² score
        best_r2 = results_df["R2"].max()
        best_r2_model = results_df.loc[results_df["R2"].idxmax(), "Model"]
        report.append(f"• Highest R² Score: {best_r2:.4f} ({best_r2_model})")

        # Lowest MAPE
        best_mape = results_df["MAPE"].min()
        best_mape_model = results_df.loc[results_df["MAPE"].idxmin(), "Model"]
        report.append(f"• Lowest MAPE: {best_mape:.2f}% ({best_mape_model})")

        # RMSE range
        rmse_range = results_df["RMSE"].max() - results_df["RMSE"].min()
        report.append(f"• RMSE Range: {rmse_range:.4f}")

        report.append("")
        report.append("=" * 80)

        # Join report
        report_text = "\n".join(report)

        # Save to file if path provided
        if output_path:
            try:
                with open(output_path, "w") as f:
                    f.write(report_text)
                logger.info(f"Evaluation report saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save report: {str(e)}")

        return report_text


def evaluate_ensemble_model(ensemble_model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Specialized evaluation for ensemble models with base model analysis.
    """
    logger.info("Evaluating ensemble model...")

    evaluation_data = {}

    # Ensemble prediction
    ensemble_pred = ensemble_model.predict(X_test)

    # Ensemble metrics
    evaluation_data["ensemble_metrics"] = {
        "rmse": np.sqrt(mean_squared_error(y_test, ensemble_pred)),
        "mae": mean_absolute_error(y_test, ensemble_pred),
        "r2": r2_score(y_test, ensemble_pred),
    }

    # Base model predictions and metrics
    if hasattr(ensemble_model, "fitted_base_models"):
        base_predictions = {}
        base_metrics = {}

        for name, model in ensemble_model.fitted_base_models.items():
            pred = model.predict(X_test)
            base_predictions[name] = pred

            base_metrics[name] = {
                "rmse": np.sqrt(mean_squared_error(y_test, pred)),
                "mae": mean_absolute_error(y_test, pred),
                "r2": r2_score(y_test, pred),
            }

        evaluation_data["base_predictions"] = base_predictions
        evaluation_data["base_metrics"] = base_metrics

        # Model weights
        if hasattr(ensemble_model, "get_model_weights"):
            evaluation_data["model_weights"] = ensemble_model.get_model_weights()

    return evaluation_data


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    import numpy as np

    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)])
    y = pd.Series(X.sum(axis=1) + np.random.randn(100) * 0.1)

    # Create sample models
    models = {"RandomForest": RandomForestRegressor(n_estimators=10, random_state=42), "LinearRegression": LinearRegression()}

    # Fit models
    for model in models.values():
        model.fit(X, y)

    # Evaluate
    evaluator = ModelEvaluator(models)
    results = evaluator.evaluate_all_models(X, y)

    print("Evaluation Results:")
    print(results)

    # Generate report
    report = evaluator.generate_evaluation_report(X, y)
    print("\n" + report)
