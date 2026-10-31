"""
Pydantic Schemas for JEEVAN-DRISHTI API
=======================================
Phase 17: FastAPI Backend & Analytical Engine Integration

Provides common metadata and request/response models adhering strictly to:
- Distinct prediction types: OBSERVED, FORECAST, ANOMALY_SIGNAL, SIMULATION
- Standardized metadata (District, Disease, Period, Model Version, Limitations)
- Pydantic v2 validation
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class PredictionType(str, Enum):
    OBSERVED = "OBSERVED"
    FORECAST = "FORECAST"
    RISK_CLASSIFICATION = "RISK_CLASSIFICATION"
    ANOMALY_SIGNAL = "ANOMALY_SIGNAL"
    SIMULATION = "SIMULATION"
    EWARS_THRESHOLD_SIGNAL = "EWARS_THRESHOLD_SIGNAL"


class AnalyticalMetadata(BaseModel):
    """Standardized metadata attached to every analytical engine response."""
    district: Optional[str] = Field(None, description="Standardized district name")
    disease: Optional[str] = Field(None, description="Standardized disease name")
    reporting_period: Optional[str] = Field(None, description="Epidemiological reporting period (e.g., 2025-W32)")
    source_observation_date: Optional[str] = Field(None, description="Date of source surveillance report")
    generated_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="API response generation timestamp")
    model_version: str = Field("v1.0.0", description="Analytical engine or registry model version")
    data_quality_status: str = Field("VALIDATED_IDSP_SURVEILLANCE", description="Quality validation state of underlying data")
    prediction_type: PredictionType = Field(PredictionType.OBSERVED, description="Classification of analytical result")
    limitations: str = Field(
        "Decision-support only. Requires epidemiological verification; does not constitute autonomous clinical authority.",
        description="Public health and operational limitations"
    )

    model_config = ConfigDict(from_attributes=True)


# -----------------------------------------------------------------------------
# Overview & System Status
# -----------------------------------------------------------------------------

class OverviewResponse(BaseModel):
    platform_name: str
    version: str
    status: str
    surveillance_territory: str
    total_active_districts: int
    unsupported_districts: List[str]
    total_weekly_observations: int
    active_outbreak_alerts_count: int
    registered_models_count: int
    monitored_diseases_count: int
    system_drift_status: str
    governance_notice: str
    metadata: AnalyticalMetadata


# -----------------------------------------------------------------------------
class DistrictItem(BaseModel):
    district_id: str
    district_name: str
    division: str
    is_active: bool
    notes: Optional[str] = None


class DistrictListResponse(BaseModel):
    total_districts: int
    active_count: int
    districts: List[DistrictItem]


class DistrictDetailResponse(BaseModel):
    district: DistrictItem
    historical_outbreak_weeks: int
    total_observed_cases: float
    total_observed_deaths: float
    recent_risk_tier: Optional[str] = None
    recent_forecast_cases: Optional[float] = None
    is_supported: bool
    unsupported_reason: Optional[str] = None
    metadata: AnalyticalMetadata


# -----------------------------------------------------------------------------
class DiseaseItem(BaseModel):
    disease_id: str
    disease_name: str
    category: str
    is_priority: bool
    transmission_route: Optional[str] = None


class DiseaseListResponse(BaseModel):
    total_diseases: int
    diseases: List[DiseaseItem]


class DiseaseTrendPoint(BaseModel):
    reporting_period: str
    calendar_year: int
    week_number: int
    observed_cases: float
    observed_deaths: float


class DiseaseTrendResponse(BaseModel):
    disease_name: str
    total_historical_weeks: int
    peak_weekly_cases: float
    historical_cfr_pct: float
    trends: List[DiseaseTrendPoint]
    metadata: AnalyticalMetadata


# -----------------------------------------------------------------------------
class DistrictRiskItem(BaseModel):
    district_id: str
    district_name: str
    disease_name: str
    reporting_period: str
    risk_tier: str
    risk_score: float
    predicted_probability: float
    contributing_factors: List[str]
    model_version: str


class RiskOverviewResponse(BaseModel):
    reporting_period: str
    elevated_districts_count: int
    watchlist_districts_count: int
    normal_districts_count: int
    district_risks: List[DistrictRiskItem]
    metadata: AnalyticalMetadata


# -----------------------------------------------------------------------------
class DistrictRiskDetailResponse(BaseModel):
    district_name: str
    disease_name: str
    is_supported: bool
    current_risk: Optional[DistrictRiskItem] = None
    unsupported_reason: Optional[str] = None
    metadata: AnalyticalMetadata


# -----------------------------------------------------------------------------
class ForecastHorizonItem(BaseModel):
    horizon_days: int
    horizon_weeks: int
    predicted_cases: float
    pi_lower_80: float
    pi_upper_80: float
    pi_lower_90: float
    pi_upper_90: float
    trend_direction: str


class DistrictForecastResponse(BaseModel):
    district_name: str
    disease_name: str
    reporting_period: str
    is_supported: bool
    unsupported_reason: Optional[str] = None
    horizons: List[ForecastHorizonItem]
    model_version: str
    metadata: AnalyticalMetadata


# -----------------------------------------------------------------------------
class ForecastOverviewResponse(BaseModel):
    reporting_period: str
    forecast_horizon_days: int
    total_forecasted_cases_statewide: float
    district_forecasts: List[Dict[str, Any]]
    metadata: AnalyticalMetadata


# -----------------------------------------------------------------------------
class AnomalyItem(BaseModel):
    anomaly_id: str
    district_id: str
    district_name: str
    disease_name: str
    reporting_period: str
    anomaly_score: float
    is_anomaly: bool
    severity_tier: str
    signal_types: List[str]
    seasonal_zscore: Optional[float] = None
    rolling_zscore: Optional[float] = None


class AnomalyListResponse(BaseModel):
    total_anomalies: int
    anomalies: List[AnomalyItem]
    metadata: AnalyticalMetadata


# -----------------------------------------------------------------------------
class AnomalyDetailResponse(BaseModel):
    anomaly: AnomalyItem
    historical_baseline: Dict[str, Any]
    triggering_features: List[str]
    metadata: AnalyticalMetadata


# -----------------------------------------------------------------------------
class AlertItem(BaseModel):
    alert_id: str
    district_name: str
    disease_name: str
    reporting_period: str
    alert_type: str
    severity: str
    trigger_reason: str
    observed_cases: float
    threshold_value: Optional[float] = None
    alert_status: str
    human_verified: bool
    created_at: str


class AlertListResponse(BaseModel):
    total_alerts: int
    pending_review_count: int
    alerts: List[AlertItem]
    metadata: AnalyticalMetadata


# -----------------------------------------------------------------------------
class AlertReviewAction(str, Enum):
    CONFIRMED_OUTBREAK = "CONFIRMED_OUTBREAK"
    DISCARDED_FALSE_ALERT = "DISCARDED_FALSE_ALERT"
    MONITORED_WATCHLIST = "MONITORED_WATCHLIST"


class AlertReviewRequest(BaseModel):
    reviewer_id: str = Field(..., description="ID or official email of public-health reviewer")
    action_taken: AlertReviewAction = Field(..., description="Review verdict")
    comments: str = Field(..., description="Field notes and verification justification", min_length=5)


class AlertReviewResponse(BaseModel):
    review_id: str
    alert_id: str
    status: str
    action_taken: str
    reviewer_id: str
    review_timestamp: str
    message: str


# -----------------------------------------------------------------------------
# Climate & Geospatial Intelligence
# -----------------------------------------------------------------------------

class ClimateIndicatorItem(BaseModel):
    district_name: str
    reporting_period: str
    rainfall_accum_4w_mm: float
    rainfall_deficit_spi_proxy: float
    temperature_mean_c: float
    temp_diurnal_range_c: float
    climate_epidemic_risk_tier: str


class ClimateIntelligenceResponse(BaseModel):
    reporting_period: str
    statewide_monsoon_phase: str
    high_climate_risk_districts: List[str]
    district_climate_indicators: List[ClimateIndicatorItem]
    metadata: AnalyticalMetadata


# -----------------------------------------------------------------------------
class HotspotDistrictItem(BaseModel):
    district_name: str
    division: str
    disease_name: str
    moran_quadrant: str
    gi_star_zscore: float
    hotspot_classification: str
    spatial_lag_cases: float


class GeospatialHotspotsResponse(BaseModel):
    reporting_period: str
    global_morans_i: float
    p_value: float
    spatial_clustering_significant: bool
    hotspot_districts: List[HotspotDistrictItem]
    metadata: AnalyticalMetadata


# -----------------------------------------------------------------------------
class ResourceDemandItem(BaseModel):
    district_name: str
    disease_name: str
    reporting_period: str
    predicted_cases: float
    supported_bed_days_estimate: float
    data_limitation_notice: str
    ml_bed_forecasting_supported: bool = False
    medicine_stock_supported: bool = False


class ResourceDemandResponse(BaseModel):
    reporting_period: str
    operational_disclaimer: str
    total_estimated_bed_days: float
    district_resource_estimates: List[ResourceDemandItem]
    unsupported_categories: List[str]
    metadata: AnalyticalMetadata


# -----------------------------------------------------------------------------
class FeatureContribution(BaseModel):
    feature_name: str
    feature_value: float
    contribution_score: float
    direction: str


class ExplainabilityResponse(BaseModel):
    prediction_id: str
    district_name: str
    disease_name: str
    model_name: str
    prediction_type: str
    top_contributing_features: List[FeatureContribution]
    clinical_plain_language_summary: str
    metadata: AnalyticalMetadata


class ModelRegistryItem(BaseModel):
    model_id: str
    model_name: str
    model_type: str
    disease: str
    supported_districts_count: int
    operational_status: str
    primary_metric_name: Optional[str] = None
    primary_metric_value: Optional[float] = None
    artifact_path: str
    version: str


class ModelRegistryListResponse(BaseModel):
    total_models: int
    models: List[ModelRegistryItem]
    drift_status: str
    governance_notice: str


class ModelMetricsResponse(BaseModel):
    total_models_evaluated: int
    metrics_catalog: List[Dict[str, Any]]
    baseline_comparisons: List[Dict[str, Any]]
    overfitting_gaps: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# Data Quality & Governance
# -----------------------------------------------------------------------------

class DataQualityResponse(BaseModel):
    dataset_version: str
    total_verified_outbreak_weeks: int
    surveillance_completeness_pct: float
    missing_data_audit: List[Dict[str, Any]]
    deduplication_status: str
    data_leakage_safeguards_active: bool
    unsupported_variables_documented: List[str]


# -----------------------------------------------------------------------------
# Scenario Simulation
# -----------------------------------------------------------------------------

class ScenarioType(str, Enum):
    VECTOR_CONTROL_INTENSIFICATION = "VECTOR_CONTROL_INTENSIFICATION"
    EARLY_ACTIVE_SURVEILLANCE = "EARLY_ACTIVE_SURVEILLANCE"
    DELAYED_INTERVENTION = "DELAYED_INTERVENTION"
    MULTI_DISTRICT_COORDINATION = "MULTI_DISTRICT_COORDINATION"


class ScenarioRunRequest(BaseModel):
    scenario_type: ScenarioType
    target_district: str
    disease: str = "Dengue"
    intervention_start_week: int = 1
    duration_weeks: int = 4
    assumed_efficacy_tier: str = Field("MODERATE", description="CONSERVATIVE, MODERATE, or AGGRESSIVE")


class ScenarioRunResponse(BaseModel):
    scenario_id: str
    scenario_type: str
    target_district: str
    disease: str
    status: str
    projected_baseline_cases: float
    projected_intervention_cases: float
    hypothetical_cases_averted: float
    explicit_assumptions: List[str]
    governance_caveat: str
    metadata: AnalyticalMetadata


# -----------------------------------------------------------------------------
# Refresh & Ingestion Schemas (Phase 18)
# -----------------------------------------------------------------------------

class RefreshSummaryResponse(BaseModel):
    """Ultra-lightweight snapshot for 2-second dashboard polling."""
    last_dashboard_refresh: str = Field(..., description="Wall-clock timestamp of this 2-second poll response")
    source_observation_date: str = Field(..., description="Epidemiological date of latest IDSP observation")
    source_frequency_label: str = Field(..., description="Reporting cadence of underlying public-health data")
    refresh_interval_seconds: float = Field(2.0, description="Dashboard UI polling interval in seconds")
    data_status_label: str = Field(..., description="Truthful human-readable display label")
    system_status: str = Field("HEALTHY", description="Operational state: HEALTHY, PROCESSING, or IDLE")
    last_successful_ingestion: str = Field(..., description="Timestamp of latest successful data ingestion")
    last_processing_completion: str = Field(..., description="Timestamp of latest feature/preprocessing completion")
    last_prediction_generation: str = Field(..., description="Timestamp of latest model prediction generation")
    model_version: str = Field(..., description="Active analytical engine composite version")
    total_districts: int = Field(36, description="Total districts tracked in Maharashtra")
    total_diseases: int = Field(28, description="Total diseases cataloged")
    total_observations: int = Field(..., description="Total historical surveillance records in database")
    active_alerts_count: int = Field(..., description="Currently active outbreak alerts requiring review")
    high_risk_districts_count: int = Field(..., description="Districts currently classified in High Risk tier")
    anomalies_flagged_count: int = Field(..., description="Active statistical and threshold anomalies")
    recent_alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Latest alerts feed")
    truthful_clarification: str = Field(
        "Dashboard UI refreshes every 2 seconds. Surveillance source data is updated weekly via IDSP. "
        "Predictive models are cached and recomputed only when verified new surveillance records arrive.",
        description="Public health and operational cadence clarification"
    )

    model_config = ConfigDict(from_attributes=True)


class IngestionResponse(BaseModel):
    """Structured response from the 11-stage automated ingestion pipeline."""
    ingestion_id: str
    filename: str
    file_hash_sha256: str
    status: str = Field(..., description="SUCCESS, DUPLICATE_SKIPPED, VALIDATION_FAILED, or PROCESSING_FAILED")
    stage_reached: str
    records_total: int
    records_valid: int
    records_invalid: int
    records_duplicates: int
    new_records_inserted: int
    predictions_updated: bool
    validation_errors: Optional[List[str]] = None
    execution_duration_sec: float
    started_at: str
    completed_at: str
    message: str


class AuditLogItem(BaseModel):
    id: int
    ingestion_id: str
    filename: str
    file_hash_sha256: str
    file_size_bytes: int
    source_type: str
    records_total: int
    records_valid: int
    records_invalid: int
    records_duplicates: int
    new_records_inserted: int
    status: str
    stage_completed: str
    validation_errors: Optional[str] = None
    predictions_recomputed: bool
    duration_seconds: float
    started_at: str
    completed_at: str


class AuditLogListResponse(BaseModel):
    total_logs: int
    logs: List[AuditLogItem]
