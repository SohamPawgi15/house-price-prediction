"""
Unit tests for data preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.preprocessor import HousePricePreprocessor, load_and_preprocess_data


class TestHousePricePreprocessor:
    """Test suite for HousePricePreprocessor."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample house price data for testing."""
        return pd.DataFrame({
            'MSSubClass': [60, 20, 70],
            'MSZoning': ['RL', 'RL', 'RM'],
            'LotFrontage': [65.0, np.nan, 75.0],
            'LotArea': [8450, 9600, 10000],
            'Street': ['Pave', 'Pave', 'Pave'],
            'OverallQual': [7, 6, 8],
            'OverallCond': [5, 8, 5],
            'YearBuilt': [2003, 1976, 2000],
            'YearRemodAdd': [2003, 1976, 2000],
            'GrLivArea': [1710, 1262, 1800],
            '1stFlrSF': [856, 1262, 900],
            '2ndFlrSF': [854, 0, 900],
            'TotalBsmtSF': [856, 1262, 800],
            'FullBath': [2, 2, 2],
            'HalfBath': [1, 0, 1],
            'BsmtFullBath': [1, 0, 1],
            'BsmtHalfBath': [0, 1, 0],
            'GarageCars': [2.0, 2.0, np.nan],
            'GarageArea': [548.0, 460.0, np.nan],
            'MasVnrArea': [196.0, np.nan, 0.0],
            'Neighborhood': ['CollgCr', 'Veenker', 'NoRidge'],
            'Exterior1st': ['VinylSd', 'MetalSd', 'VinylSd'],
            'BsmtQual': ['Gd', 'Gd', np.nan],
            'FireplaceQu': [np.nan, 'TA', np.nan]
        })
    
    @pytest.fixture
    def sample_target(self):
        """Create sample target values."""
        return pd.Series([208500, 181500, 250000])
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return HousePricePreprocessor(
            outlier_threshold=3.0,
            log_transform_features=['LotArea', 'GrLivArea'],
            create_interactions=True
        )
    
    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = HousePricePreprocessor()
        
        assert preprocessor.outlier_threshold == 3.0
        assert preprocessor.log_transform_features == []
        assert preprocessor.create_interactions is True
        assert preprocessor.label_encoders == {}
    
    def test_fit(self, preprocessor, sample_data, sample_target):
        """Test fitting preprocessor."""
        preprocessor.fit(sample_data, sample_target)
        
        # Check that feature types are identified
        assert len(preprocessor.numerical_features) > 0
        assert len(preprocessor.categorical_features) > 0
        
        # Check that label encoders are fitted
        assert len(preprocessor.label_encoders) == len(preprocessor.categorical_features)
        
        # Verify specific features are categorized correctly
        assert 'MSSubClass' in preprocessor.numerical_features
        assert 'MSZoning' in preprocessor.categorical_features
    
    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling."""
        preprocessor.numerical_features = ['LotFrontage', 'MasVnrArea', 'GarageCars', 'GarageArea']
        preprocessor.categorical_features = ['MSZoning', 'BsmtQual', 'FireplaceQu']
        
        result = preprocessor._handle_missing_values(sample_data)
        
        # Check that numerical missing values are filled
        assert not result['LotFrontage'].isnull().any()
        assert not result['MasVnrArea'].isnull().any()
        assert not result['GarageCars'].isnull().any()
        
        # Check that categorical missing values are filled
        assert not result['MSZoning'].isnull().any()
        assert not result['BsmtQual'].isnull().any()
        assert not result['FireplaceQu'].isnull().any()
    
    def test_feature_engineering(self, preprocessor, sample_data):
        """Test feature engineering."""
        result = preprocessor._engineer_features(sample_data)
        
        # Check that new features are created
        expected_features = ['TotalSF', 'TotalBathrooms', 'HouseAge', 'RemodAge', 'IsRemodeled']
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Verify calculations
        assert result['TotalSF'].iloc[0] == sample_data['TotalBsmtSF'].iloc[0] + sample_data['1stFlrSF'].iloc[0] + sample_data['2ndFlrSF'].iloc[0]
        assert result['TotalBathrooms'].iloc[0] == 4  # 2 + 1 + 1 + 0
        assert result['HouseAge'].iloc[0] == 2024 - 2003
    
    def test_log_transforms(self, preprocessor, sample_data):
        """Test log transformations."""
        preprocessor.log_transform_features = ['LotArea', 'GrLivArea']
        
        result = preprocessor._apply_log_transforms(sample_data)
        
        # Check that log features are created
        assert 'LotArea_log' in result.columns
        assert 'GrLivArea_log' in result.columns
        
        # Verify log transformation
        expected_log_value = np.log1p(sample_data['LotArea'].iloc[0])
        assert abs(result['LotArea_log'].iloc[0] - expected_log_value) < 1e-6
    
    def test_create_interactions(self, preprocessor, sample_data):
        """Test interaction term creation."""
        # Add required features for interactions
        sample_data['TotalSF'] = sample_data['TotalBsmtSF'] + sample_data['1stFlrSF'] + sample_data['2ndFlrSF']
        sample_data['TotalBathrooms'] = sample_data['FullBath'] + sample_data['HalfBath'] + sample_data['BsmtFullBath'] + sample_data['BsmtHalfBath']
        
        result = preprocessor._create_interactions(sample_data)
        
        # Check that interaction features are created
        expected_interactions = ['OverallQual_GrLivArea_interaction', 'OverallQual_TotalSF_interaction']
        
        for interaction in expected_interactions:
            assert interaction in result.columns
    
    def test_transform_complete(self, preprocessor, sample_data, sample_target):
        """Test complete transformation pipeline."""
        # Fit first
        preprocessor.fit(sample_data, sample_target)
        
        # Transform
        result = preprocessor.transform(sample_data)
        
        # Check that result is DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check that no missing values remain
        assert not result.isnull().any().any()
        
        # Check that new features are present
        assert 'HouseAge' in result.columns
        assert 'TotalSF' in result.columns
    
    def test_outlier_removal(self, preprocessor, sample_data):
        """Test outlier removal."""
        # Add outlier data
        outlier_data = sample_data.copy()
        outlier_data['SalePrice'] = [208500, 181500, 250000]
        outlier_data.loc[0, 'GrLivArea'] = 5000
        outlier_data.loc[0, 'SalePrice'] = 100000  # Outlier: large area, low price
        
        result = preprocessor._remove_outliers(outlier_data)
        
        # Check that outlier is removed
        assert len(result) < len(outlier_data)
    
    def test_encode_categorical(self, preprocessor, sample_data):
        """Test categorical encoding."""
        # Fit label encoders first
        preprocessor.categorical_features = ['MSZoning', 'Neighborhood']
        preprocessor.fit(sample_data)
        
        result = preprocessor._encode_categorical(sample_data)
        
        # Check that categorical features are encoded as numbers
        assert result['MSZoning'].dtype in [np.int64, np.float64]
        assert result['Neighborhood'].dtype in [np.int64, np.float64]
    
    def test_unseen_categories(self, preprocessor, sample_data):
        """Test handling of unseen categories."""
        # Fit on original data
        preprocessor.fit(sample_data)
        
        # Create test data with unseen category
        test_data = sample_data.copy()
        test_data.loc[0, 'MSZoning'] = 'NewZone'  # Unseen category
        
        # Should not raise error
        result = preprocessor.transform(test_data)
        
        # Unseen category should be encoded as 0
        assert result['MSZoning'].iloc[0] == 0


class TestDataLoading:
    """Test data loading functions."""
    
    @patch('pandas.read_csv')
    def test_load_and_preprocess_data(self, mock_read_csv):
        """Test data loading and preprocessing function."""
        # Mock data
        train_data = pd.DataFrame({
            'Id': [1, 2, 3],
            'MSSubClass': [60, 20, 70],
            'MSZoning': ['RL', 'RL', 'RM'],
            'LotArea': [8450, 9600, 10000],
            'SalePrice': [208500, 181500, 250000]
        })
        
        test_data = pd.DataFrame({
            'Id': [1461, 1462, 1463],
            'MSSubClass': [60, 20, 70],
            'MSZoning': ['RL', 'RL', 'RM'],
            'LotArea': [8450, 9600, 10000]
        })
        
        mock_read_csv.side_effect = [train_data, test_data]
        
        # Test function
        with patch('src.data.preprocessor.HousePricePreprocessor') as mock_preprocessor:
            mock_instance = mock_preprocessor.return_value
            mock_instance.fit.return_value = None
            mock_instance.transform.side_effect = [
                train_data.drop(['SalePrice'], axis=1),  # X_train_processed
                test_data  # X_test_processed
            ]
            
            X_train, X_test, y_train = load_and_preprocess_data('train.csv', 'test.csv')
            
            # Verify function calls
            assert mock_read_csv.call_count == 2
            mock_instance.fit.assert_called_once()
            assert mock_instance.transform.call_count == 2
            
            # Verify return values
            assert len(X_train) == 3
            assert len(X_test) == 3
            assert len(y_train) == 3


@pytest.mark.integration
class TestPreprocessorIntegration:
    """Integration tests for preprocessor."""
    
    def test_real_data_compatibility(self):
        """Test preprocessor with real data structure."""
        # This would test with actual house prices dataset
        # Skip if data files don't exist
        train_path = "house-prices-advanced-regression-techniques/train.csv"
        test_path = "house-prices-advanced-regression-techniques/test.csv"
        
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            pytest.skip("Real data files not available")
        
        try:
            X_train, X_test, y_train = load_and_preprocess_data(train_path, test_path)
            
            # Basic checks
            assert X_train.shape[0] > 0
            assert X_test.shape[0] > 0
            assert len(y_train) == X_train.shape[0]
            assert not X_train.isnull().any().any()
            assert not X_test.isnull().any().any()
            
        except Exception as e:
            pytest.fail(f"Real data preprocessing failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__])
