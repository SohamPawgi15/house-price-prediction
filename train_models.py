#!/usr/bin/env python3
"""
Main training script for house price prediction models.
This script orchestrates the complete model training pipeline.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.preprocessor import load_and_preprocess_data
from models.ensemble_models import train_complete_pipeline
import joblib
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train house price prediction models'
    )
    
    parser.add_argument(
        '--train-path',
        type=str,
        default='house-prices-advanced-regression-techniques/train.csv',
        help='Path to training data CSV file'
    )
    
    parser.add_argument(
        '--test-path',
        type=str,
        default='house-prices-advanced-regression-techniques/test.csv',
        help='Path to test data CSV file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    
    parser.add_argument(
        '--quick-train',
        action='store_true',
        help='Use reduced parameters for quick training (for testing)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing models without training'
    )
    
    return parser.parse_args()


def validate_data_files(train_path: str, test_path: str) -> bool:
    """Validate that data files exist and are readable."""
    logger.info("Validating data files...")
    
    if not os.path.exists(train_path):
        logger.error(f"Training data file not found: {train_path}")
        return False
    
    if not os.path.exists(test_path):
        logger.error(f"Test data file not found: {test_path}")
        return False
    
    try:
        # Quick read test
        train_df = pd.read_csv(train_path, nrows=5)
        test_df = pd.read_csv(test_path, nrows=5)
        
        # Check required columns
        required_cols = ['Id', 'MSSubClass', 'MSZoning', 'LotArea', 'OverallQual']
        for col in required_cols:
            if col not in train_df.columns:
                logger.error(f"Required column '{col}' not found in training data")
                return False
            if col not in test_df.columns and col != 'SalePrice':
                logger.error(f"Required column '{col}' not found in test data")
                return False
        
        if 'SalePrice' not in train_df.columns:
            logger.error("Target column 'SalePrice' not found in training data")
            return False
        
        logger.info("Data files validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error reading data files: {str(e)}")
        return False


def create_output_directory(output_dir: str) -> bool:
    """Create output directory if it doesn't exist."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory ready: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to create output directory: {str(e)}")
        return False


def validate_existing_models(output_dir: str) -> dict:
    """Validate existing trained models."""
    logger.info("Validating existing models...")
    
    if not os.path.exists(output_dir):
        logger.warning(f"Models directory does not exist: {output_dir}")
        return {}
    
    model_files = [f for f in os.listdir(output_dir) if f.endswith('.joblib')]
    
    if not model_files:
        logger.warning("No model files found")
        return {}
    
    valid_models = {}
    
    for model_file in model_files:
        model_path = os.path.join(output_dir, model_file)
        try:
            model = joblib.load(model_path)
            model_name = model_file.replace('_model.joblib', '')
            
            # Basic validation - check if model has predict method
            if hasattr(model, 'predict'):
                # Test with dummy data
                dummy_data = pd.DataFrame(np.random.randn(5, 10))
                try:
                    predictions = model.predict(dummy_data)
                    if len(predictions) == 5:
                        valid_models[model_name] = {
                            'path': model_path,
                            'type': type(model).__name__,
                            'status': 'valid'
                        }
                        logger.info(f"✓ {model_name}: {type(model).__name__}")
                    else:
                        logger.warning(f"✗ {model_name}: Invalid prediction output")
                except Exception as pred_error:
                    logger.warning(f"✗ {model_name}: Prediction test failed - {str(pred_error)}")
            else:
                logger.warning(f"✗ {model_name}: No predict method found")
                
        except Exception as e:
            logger.warning(f"✗ {model_file}: Failed to load - {str(e)}")
    
    logger.info(f"Found {len(valid_models)} valid models out of {len(model_files)} files")
    return valid_models


def save_training_report(trainer, output_dir: str, training_time: float):
    """Save training report with model performance metrics."""
    try:
        report = {
            'training_completed': True,
            'training_time_seconds': training_time,
            'models_trained': list(trainer.models.keys()),
            'model_performance': {}
        }
        
        # Add performance metrics
        for name, scores in trainer.model_scores.items():
            report['model_performance'][name] = {
                'cv_rmse': float(scores['cv_rmse']),
                'cv_rmse_std': float(scores['cv_rmse_std'])
            }
        
        # Get model summary
        summary_df = trainer.get_model_summary()
        report['best_model'] = summary_df.iloc[0]['Model']
        report['best_cv_rmse'] = float(summary_df.iloc[0]['CV_RMSE'])
        
        # Save report
        import json
        from datetime import datetime
        
        report['timestamp'] = datetime.now().isoformat()
        
        report_path = os.path.join(output_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Failed to save training report: {str(e)}")


def main():
    """Main training function."""
    args = parse_arguments()
    
    logger.info("=" * 60)
    logger.info("HOUSE PRICE PREDICTION MODEL TRAINING")
    logger.info("=" * 60)
    
    # Validate inputs
    if not validate_data_files(args.train_path, args.test_path):
        logger.error("Data validation failed. Exiting.")
        sys.exit(1)
    
    if not create_output_directory(args.output_dir):
        logger.error("Failed to create output directory. Exiting.")
        sys.exit(1)
    
    # Validate existing models if requested
    if args.validate_only:
        logger.info("Validation mode - checking existing models only")
        valid_models = validate_existing_models(args.output_dir)
        
        if valid_models:
            logger.info("Model validation summary:")
            for name, info in valid_models.items():
                logger.info(f"  {name}: {info['type']} - {info['status']}")
        else:
            logger.warning("No valid models found")
        
        return
    
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        X_train, X_test, y_train = load_and_preprocess_data(
            args.train_path, 
            args.test_path
        )
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Target distribution: mean={y_train.mean():.0f}, std={y_train.std():.0f}")
        
        # Train models
        logger.info("Starting model training...")
        import time
        start_time = time.time()
        
        trainer, best_model = train_complete_pipeline(X_train, y_train)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save models
        logger.info("Saving trained models...")
        trainer.save_models(args.output_dir)
        
        # Save training report
        save_training_report(trainer, args.output_dir, training_time)
        
        # Print final summary
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        
        summary_df = trainer.get_model_summary()
        logger.info("Model Performance Ranking:")
        logger.info(f"\n{summary_df.to_string(index=False)}")
        
        best_model_name = summary_df.iloc[0]['Model']
        best_rmse = summary_df.iloc[0]['CV_RMSE']
        
        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"Best CV RMSE: {best_rmse:.4f}")
        
        if hasattr(best_model, 'get_model_weights'):
            weights = best_model.get_model_weights()
            if weights:
                logger.info("Ensemble Weights:")
                for name, weight in weights.items():
                    logger.info(f"  {name}: {weight:.4f}")
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Models saved to: {args.output_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
