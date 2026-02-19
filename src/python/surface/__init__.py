"""Surface construction subpackage."""

# eSSVI — Extended SSVI surface with maturity-dependent correlation
from .essvi import ESSVISurface, ESSVIParameters, ESSVICalibrationResult, essvi_to_ssvi_params

# Automated arbitrage grid checker
from .validation import SurfaceValidator, ValidationSummary

# Benchmark structure pricer
from .benchmark import BenchmarkStructurePricer, BenchmarkReport, BenchmarkResult

# Greeks calculator, validator and dashboard
from .greeks import GreeksCalculator, GreeksValidator, plot_greeks_dashboard

# Parameter dynamics monitoring (PCA)
from .parameter_monitoring import ParameterDynamicsMonitor, PCAMonitorReport

# Backtesting framework
from .backtesting import Backtester, BacktestReport, StressScenario

# Calibration audit trail
from .audit import CalibrationLog, CalibrationRecord

# Parameter version control
from .audit import ParameterVersionControl, ParameterVersion

# Alerting system
from .alerting import AlertSystem, Alert

# Override manager
from .overrides import OverrideManager, Override

# Bid/Ask surface generator
from .bid_ask import BidAskSurfaceGenerator, SpreadReport

# Daily calibration pipeline
from .pipeline import DailyCalibrationPipeline, PipelineConfig, PipelineReport

# Trader adjustment layer
from .overrides import TraderAdjustments, TraderAdjustmentRecord

__all__ = [
    # eSSVI
    "ESSVISurface",
    "ESSVIParameters",
    "ESSVICalibrationResult",
    "essvi_to_ssvi_params",
    # 5.1
    "SurfaceValidator",
    "ValidationSummary",
    # 5.2
    "BenchmarkStructurePricer",
    "BenchmarkReport",
    "BenchmarkResult",
    # 5.3 + 5.6
    "GreeksCalculator",
    "GreeksValidator",
    "plot_greeks_dashboard",
    # 5.4
    "ParameterDynamicsMonitor",
    "PCAMonitorReport",
    # 5.5
    "Backtester",
    "BacktestReport",
    "StressScenario",
    # 6.1
    "CalibrationLog",
    "CalibrationRecord",
    # 6.2
    "ParameterVersionControl",
    "ParameterVersion",
    # 6.3
    "AlertSystem",
    "Alert",
    # 6.4
    "OverrideManager",
    "Override",
    # 6.5
    "BidAskSurfaceGenerator",
    "SpreadReport",
    # 6.6
    "DailyCalibrationPipeline",
    "PipelineConfig",
    "PipelineReport",
    # 6.7
    "TraderAdjustments",
    "TraderAdjustmentRecord",
]
