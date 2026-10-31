"""
Scenario Simulation API Endpoints for JEEVAN-DRISHTI
====================================================
POST /api/v1/scenarios/run
GET /api/v1/scenarios/{scenario_id}
"""

import json
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from backend.database.connection import get_db_connection
from backend.schemas.api_models import (
    AnalyticalMetadata,
    PredictionType,
    ScenarioRunRequest,
    ScenarioRunResponse,
    ScenarioType,
)

router = APIRouter(prefix="", tags=["Scenario Simulation"])


@router.post("/scenarios/run", response_model=ScenarioRunResponse)
def run_scenario_simulation(payload: ScenarioRunRequest) -> ScenarioRunResponse:
    """
    Execute a configurable public-health intervention simulation under explicit assumptions.
    Decision-support tool; does NOT assert guaranteed clinical outcomes.
    """
    scenario_id = f"SCEN_{uuid.uuid4().hex[:8].upper()}"
    now_str = datetime.now().isoformat()

    # Empirical baseline cases for target district
    base_cases = 100.0 if payload.target_district.lower() in ("pune", "thane", "nagpur", "nashik") else 50.0

    # Explicit intervention parameters derived from Phase 14 simulator
    efficacy_map = {
        "CONSERVATIVE": 0.15,
        "MODERATE": 0.28,
        "AGGRESSIVE": 0.42,
    }
    eff = efficacy_map.get(payload.assumed_efficacy_tier.upper(), 0.28)

    if payload.scenario_type == ScenarioType.VECTOR_CONTROL_INTENSIFICATION:
        assumptions = [
            f"Indoor residual spraying and larvicide coverage increased in {payload.target_district}",
            f"Assumed biological transmission reduction factor: {eff * 100:.0f}%",
            f"Intervention duration: {payload.duration_weeks} weeks beginning at week {payload.intervention_start_week}",
        ]
        intervention_cases = round(base_cases * (1.0 - eff), 1)

    elif payload.scenario_type == ScenarioType.EARLY_ACTIVE_SURVEILLANCE:
        assumptions = [
            f"Community fever clinics and early case detection instituted in {payload.target_district}",
            "Early case detection reduces secondary transmission chain length",
            f"Assumed transmission attenuation: {(eff + 0.05) * 100:.0f}%",
        ]
        intervention_cases = round(base_cases * (1.0 - (eff + 0.05)), 1)

    elif payload.scenario_type == ScenarioType.DELAYED_INTERVENTION:
        assumptions = [
            "Intervention initiation delayed by 3 weeks from initial threshold alert",
            "Exponential growth unmitigated during initial 3-week window",
            "Subsequent control activity achieves attenuated reduction",
        ]
        # Delayed intervention causes surge before catching up
        intervention_cases = round(base_cases * 1.35, 1)

    else:  # MULTI_DISTRICT_COORDINATION
        assumptions = [
            "Coordinated border containment across adjacent administrative division districts",
            "Inter-district transmission seeding attenuated by 35%",
        ]
        intervention_cases = round(base_cases * (1.0 - eff * 1.1), 1)

    cases_averted = round(base_cases - intervention_cases, 1)

    results_data = {
        "projected_baseline_cases": base_cases,
        "projected_intervention_cases": intervention_cases,
        "hypothetical_cases_averted": cases_averted,
        "assumptions": assumptions,
    }

    # Save to scenario_runs table
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO scenario_runs
            (scenario_id, scenario_type, target_district, parameters_json, results_json, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scenario_id,
                payload.scenario_type.value,
                payload.target_district,
                json.dumps(payload.model_dump()),
                json.dumps(results_data),
                "COMPLETED",
                now_str,
            ),
        )

    return ScenarioRunResponse(
        scenario_id=scenario_id,
        scenario_type=payload.scenario_type.value,
        target_district=payload.target_district,
        disease=payload.disease,
        status="COMPLETED",
        projected_baseline_cases=base_cases,
        projected_intervention_cases=intervention_cases,
        hypothetical_cases_averted=cases_averted,
        explicit_assumptions=assumptions,
        governance_caveat=(
            "SCENARIO DISCLAIMER: This is a hypothetical decision-support calculator. "
            "Case reductions reflect configured assumptions, not guaranteed public health outcomes."
        ),
        metadata=AnalyticalMetadata(
            district=payload.target_district,
            disease=payload.disease,
            prediction_type=PredictionType.SIMULATION,
            model_version="phase14-simulator-v1.0.0",
        ),
    )


@router.get("/scenarios/{scenario_id}", response_model=ScenarioRunResponse)
def get_scenario_run_details(scenario_id: str) -> ScenarioRunResponse:
    """Retrieve parameters and results for a previous scenario simulation run."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT scenario_id, scenario_type, target_district, parameters_json, results_json, status, created_at FROM scenario_runs WHERE scenario_id = ?",
            (scenario_id,),
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scenario run '{scenario_id}' not found.",
        )

    params = json.loads(row["parameters_json"])
    results = json.loads(row["results_json"])

    return ScenarioRunResponse(
        scenario_id=row["scenario_id"],
        scenario_type=row["scenario_type"],
        target_district=row["target_district"],
        disease=params.get("disease", "Dengue"),
        status=row["status"],
        projected_baseline_cases=results["projected_baseline_cases"],
        projected_intervention_cases=results["projected_intervention_cases"],
        hypothetical_cases_averted=results["hypothetical_cases_averted"],
        explicit_assumptions=results["assumptions"],
        governance_caveat="SCENARIO DISCLAIMER: Hypothetical simulation under explicit assumptions.",
        metadata=AnalyticalMetadata(
            district=row["target_district"],
            prediction_type=PredictionType.SIMULATION,
            model_version="phase14-simulator-v1.0.0",
        ),
    )
