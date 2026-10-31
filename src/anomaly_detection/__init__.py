"""JEEVAN-DRISHTI Phase 7: Independent Disease Anomaly Detection Engine & WHO EWARS-Inspired Thresholds.

Implements:
1. Seasonal baseline deviation (`Seasonal_Baseline_Deviation`).
2. Rolling statistical anomaly detection (`Rolling_Statistical_Detector`).
3. Robust MAD Z-score (`Robust_MAD_ZScore`).
4. Multivariate Isolation Forest (`Isolation_Forest_Multivariate`).
5. WHO EWARS-inspired early warning thresholds (`WHOEWARSThresholdEngine`):
   - Fixed disease-specific thresholds
   - Moving-average thresholds (rolling 4w)
   - Historical seasonal thresholds
   - Lagged alert persistence indicators
   - Consecutive weekly increase surge detection

Detects sudden case increases, unusual district-level patterns, unexpected seasonal
deviations, and unusual mortality patterns while accounting for reporting delays,
seasonality, missing observations, and reporting coverage changes.

Note: An anomaly or EWARS alert is an elevated signal requiring investigation, not proof of an outbreak.
"""

from src.anomaly_detection.ewars_thresholds import (
    EWARS_GOVERNANCE_DISCLAIMER,
    EWARS_MODULE_VERSION,
    EWARSAlertOutput,
    EWARSThresholdRecord,
    WHOEWARSThresholdEngine,
)

ANOMALY_MODEL_VERSION = "phase7-v1.0.0"
ANOMALY_DISCLAIMER = (
    "An anomaly is an elevated signal requiring investigation, not proof of an outbreak."
)

__all__ = [
    "ANOMALY_MODEL_VERSION",
    "ANOMALY_DISCLAIMER",
    "EWARS_MODULE_VERSION",
    "EWARS_GOVERNANCE_DISCLAIMER",
    "EWARSAlertOutput",
    "EWARSThresholdRecord",
    "WHOEWARSThresholdEngine",
]
