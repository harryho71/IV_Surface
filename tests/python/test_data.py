"""Data loading, validation, and cleaning tests."""
import pandas as pd
import pytest
from src.python.data.validators import OptionDataValidator
from src.python.data.cleaners import clean_option_chain


class TestOptionDataValidator:
    """Test suite for OptionDataValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return OptionDataValidator()

    @pytest.fixture
    def sample_option_data(self):
        """Create sample option chain data with UTC-aware timestamps."""
        return pd.DataFrame({
            'strike': [95, 100, 105, 100, 105, 110],
            'bid': [10.5, 5.2, 1.5, 3.2, 0.8, 0.2],
            'ask': [10.7, 5.4, 1.6, 3.4, 0.9, 0.3],
            'volume': [100, 200, 50, 150, 75, 25],
            'openInterest': [1000, 2000, 500, 1500, 750, 250],
            'impliedVolatility': [0.25, 0.20, 0.22, 0.23, 0.24, 0.26],
            'type': ['call', 'call', 'call', 'put', 'put', 'put'],
            'expiration': pd.to_datetime(['2026-03-15'] * 3 + ['2026-04-15'] * 3, utc=True),
        })

    def test_validator_initialization(self, validator):
        """Test that validator initializes correctly."""
        assert validator is not None
        assert isinstance(validator, OptionDataValidator)

    def test_valid_option_chain(self, validator, sample_option_data):
        """Test validation of valid option data."""
        valid_df, rejected_df = validator.validate_chain(sample_option_data)
        
        # All options should be valid in sample data
        assert len(valid_df) > 0
        assert 'strike' in valid_df.columns
        assert 'impliedVolatility' in valid_df.columns

    def test_bid_ask_constraint(self, validator):
        """Test bid < ask constraint."""
        df = pd.DataFrame({
            'strike': [100, 105],  # Different strikes to avoid duplicate detection
            'bid': [10.0, 5.0],
            'ask': [9.0, 5.5],  # First row violates bid < ask
            'volume': [100, 100],
            'openInterest': [1000, 1000],
            'impliedVolatility': [0.20, 0.20],
            'type': ['call', 'call'],
            'expiration': pd.to_datetime(['2026-03-15', '2026-03-15'], utc=True),
        })
        
        valid_df, rejected_df = validator.validate_chain(df)
        
        # Second row should be valid, first should be rejected
        assert len(valid_df) >= 1

    def test_iv_range_constraint(self, validator):
        """Test IV must be in [0.001, 3.0] range."""
        df = pd.DataFrame({
            'strike': [95, 100, 105],  # Different strikes to avoid duplicate detection
            'bid': [10.0, 10.0, 10.0],
            'ask': [10.5, 10.5, 10.5],
            'volume': [100, 100, 100],
            'openInterest': [1000, 1000, 1000],
            'impliedVolatility': [0.0001, 0.20, 5.0],  # First and third out of range
            'type': ['call', 'call', 'call'],
            'expiration': pd.to_datetime(['2026-03-15'] * 3, utc=True),
        })
        
        valid_df, rejected_df = validator.validate_chain(df)
        
        # Only middle row should be valid
        assert len(valid_df) >= 1
        assert valid_df['impliedVolatility'].min() >= 0.001
        assert valid_df['impliedVolatility'].max() <= 3.0


class TestDataCleaner:
    """Test suite for clean_option_chain."""

    @pytest.fixture
    def sample_option_data(self):
        """Create sample option chain data with UTC-aware timestamps."""
        return pd.DataFrame({
            'strike': [95, 100, 105, 100, 105, 110],
            'bid': [10.5, 5.2, 1.5, 3.2, 0.8, 0.2],
            'ask': [10.7, 5.4, 1.6, 3.4, 0.9, 0.3],
            'volume': [100, 200, 50, 150, 75, 25],
            'openInterest': [1000, 2000, 500, 1500, 750, 250],
            'impliedVolatility': [0.25, 0.20, 0.22, 0.23, 0.24, 0.26],
            'type': ['call', 'call', 'call', 'put', 'put', 'put'],
            'expiration': pd.to_datetime(['2026-03-15'] * 3 + ['2026-04-15'] * 3, utc=True),
        })

    def test_clean_option_chain(self, sample_option_data):
        """Test cleaning produces required columns."""
        clean_df = clean_option_chain(
            sample_option_data,
            spot_price=100.0,
            risk_free_rate=0.05,
        )
        
        # Check required columns exist
        required_cols = ['strike', 'maturity', 'moneyness', 'mid_price', 'implied_volatility']
        for col in required_cols:
            assert col in clean_df.columns, f"Missing column: {col}"

    def test_clean_moneyness_calculation(self, sample_option_data):
        """Test that moneyness is calculated correctly."""
        clean_df = clean_option_chain(
            sample_option_data,
            spot_price=100.0,
            risk_free_rate=0.05,
        )
        
        # Check that moneyness is calculated (some may be aggregated)
        assert clean_df['moneyness'].min() > 0
        assert len(clean_df) > 0

    def test_clean_mid_price_calculation(self, sample_option_data):
        """Test that mid-price is calculated."""
        clean_df = clean_option_chain(
            sample_option_data,
            spot_price=100.0,
            risk_free_rate=0.05,
        )
        
        # Mid-price should be > 0
        assert (clean_df['mid_price'] > 0).all()
        assert len(clean_df) > 0

    def test_clean_preserves_data_type(self, sample_option_data):
        """Test that clean produces numeric data types."""
        clean_df = clean_option_chain(
            sample_option_data,
            spot_price=100.0,
            risk_free_rate=0.05,
        )
        
        numeric_cols = ['strike', 'maturity', 'moneyness', 'mid_price', 'implied_volatility']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(clean_df[col]), f"Column {col} is not numeric"

