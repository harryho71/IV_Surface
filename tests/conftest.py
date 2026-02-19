"""pytest configuration and shared fixtures."""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; must be set before pyplot is imported

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir():
    """Return test data directory."""
    return PROJECT_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    """Return temporary output directory for test artifacts."""
    return tmp_path_factory.mktemp("output")
