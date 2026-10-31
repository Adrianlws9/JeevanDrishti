"""JEEVAN-DRISHTI Phase 8: Climate-Disease Intelligence Engine.

Analyses statistical, lagged, seasonal, regression-adjusted, and ML/TreeSHAP/PDP
relationships between environmental variables (rainfall, temperature, relative humidity)
and disease surveillance trends in Maharashtra (2023-2025 weekly scope + 2009-2022 historical context).

Strict Scientific Rule:
Correlation, regression coefficients, and ML feature importance / SHAP values measure
observational association and predictive utility — never proof of causation.
"""

CLIMATE_INTELLIGENCE_VERSION = "phase8-v1.0.0"
CAUSALITY_DISCLAIMER = (
    "Observed correlation, regression association, or ML/SHAP feature importance reflects "
    "statistical co-movement and predictive utility only; it does NOT establish causation."
)
