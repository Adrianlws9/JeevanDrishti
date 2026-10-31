"""
Ingestion Validation & Quality Assurance Engine
===============================================
Phase 18: Automated Data Ingestion & 2-Second Dashboard Refresh

Provides SHA-256 duplicate checking, schema compliance validation,
and data quality auditing against Maharashtra public-health standards.
"""

import hashlib
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd

from backend.database.connection import get_db_connection
from backend.ingestion.config import IngestionConfig
from jeevan_pipeline.standardize import standardize_district, standardize_disease


class IngestionValidator:
    """Validates uploaded surveillance files and enforces data hygiene rules."""

    @staticmethod
    def calculate_sha256(content: bytes) -> str:
        """Compute SHA-256 cryptographic digest of file bytes."""
        hasher = hashlib.sha256()
        hasher.update(content)
        return hasher.hexdigest()

    @staticmethod
    def is_duplicate_file(file_hash: str) -> Tuple[bool, Optional[str]]:
        """
        Check if file hash has already been successfully ingested into database.
        Returns (is_duplicate, previous_ingestion_id_or_file).
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # 1. Check ingestion_audit_logs
            cursor.execute(
                "SELECT ingestion_id, status FROM ingestion_audit_logs WHERE file_hash_sha256 = ? AND status = 'SUCCESS' LIMIT 1",
                (file_hash,),
            )
            row = cursor.fetchone()
            if row:
                return True, f"Ingestion ID {row['ingestion_id']}"

            # 2. Check data_sources table
            cursor.execute(
                "SELECT source_name FROM data_sources WHERE sha256_checksum = ? LIMIT 1",
                (file_hash,),
            )
            row_ds = cursor.fetchone()
            if row_ds:
                return True, f"Data Source: {row_ds['source_name']}"

        return False, None

    @staticmethod
    def identify_column_mapping(df: pd.DataFrame) -> Tuple[bool, Dict[str, str], List[str]]:
        """
        Match DataFrame columns against candidate names for required public-health dimensions.
        Returns (is_valid, column_mapping, missing_columns).
        """
        mapping = {}
        missing = []
        col_lower = {str(c).strip().lower(): str(c) for c in df.columns}

        # Check district
        dist_match = None
        for cand in IngestionConfig.REQUIRED_COLUMNS_CANDIDATES["district"]:
            if cand.lower() in col_lower:
                dist_match = col_lower[cand.lower()]
                break
        if dist_match:
            mapping["district"] = dist_match
        else:
            missing.append("district")

        # Check disease
        dis_match = None
        for cand in IngestionConfig.REQUIRED_COLUMNS_CANDIDATES["disease"]:
            if cand.lower() in col_lower:
                dis_match = col_lower[cand.lower()]
                break
        if dis_match:
            mapping["disease"] = dis_match
        else:
            missing.append("disease")

        # Check cases
        cases_match = None
        for cand in IngestionConfig.REQUIRED_COLUMNS_CANDIDATES["cases"]:
            if cand.lower() in col_lower:
                cases_match = col_lower[cand.lower()]
                break
        if cases_match:
            mapping["cases"] = cases_match
        else:
            missing.append("cases (or observed_cases)")

        # Check period or year/week
        period_match = None
        for cand in IngestionConfig.REQUIRED_COLUMNS_CANDIDATES["period"]:
            if cand.lower() in col_lower:
                period_match = col_lower[cand.lower()]
                break
        if period_match:
            mapping["period"] = period_match
        else:
            # Check year and week combination
            year_match, week_match = None, None
            for cand in IngestionConfig.REQUIRED_COLUMNS_CANDIDATES["year"]:
                if cand.lower() in col_lower:
                    year_match = col_lower[cand.lower()]
                    break
            for cand in IngestionConfig.REQUIRED_COLUMNS_CANDIDATES["week"]:
                if cand.lower() in col_lower:
                    week_match = col_lower[cand.lower()]
                    break
            if year_match and week_match:
                mapping["year"] = year_match
                mapping["week"] = week_match
            else:
                missing.append("reporting_period (or calendar_year + week_number)")

        # Check optional deaths
        deaths_match = None
        for cand in IngestionConfig.REQUIRED_COLUMNS_CANDIDATES["deaths"]:
            if cand.lower() in col_lower:
                deaths_match = col_lower[cand.lower()]
                break
        if deaths_match:
            mapping["deaths"] = deaths_match

        is_valid = len(missing) == 0
        return is_valid, mapping, missing

    @classmethod
    def validate_and_clean_dataset(
        cls, df_raw: pd.DataFrame
    ) -> Tuple[pd.DataFrame, int, int, int, List[str]]:
        """
        Validate schema, cleanse values, standardize names, and discard invalid rows.
        Returns: (df_clean, total_records, valid_records, invalid_records, validation_errors)
        """
        total_records = len(df_raw)
        validation_errors = []

        if total_records == 0:
            return pd.DataFrame(), 0, 0, 0, ["The uploaded file contains 0 data rows."]

        is_valid_schema, mapping, missing_cols = cls.identify_column_mapping(df_raw)
        if not is_valid_schema:
            return (
                pd.DataFrame(),
                total_records,
                0,
                total_records,
                [f"Missing mandatory public-health columns: {', '.join(missing_cols)}"],
            )

        valid_rows = []
        invalid_count = 0

        period_regex = re.compile(r"^\d{4}-W(?:0[1-9]|[1-4][0-9]|5[0-3])$")

        for idx, row in df_raw.iterrows():
            row_errors = []

            # District standardization
            raw_district = str(row[mapping["district"]]).strip()
            std_district, dist_type = standardize_district(raw_district)
            if not std_district:
                row_errors.append(f"Unrecognized district: '{raw_district}'")

            # Disease standardization
            raw_disease = str(row[mapping["disease"]]).strip()
            std_disease, is_suspected, mapped = standardize_disease(raw_disease)
            if not std_disease:
                row_errors.append(f"Unrecognized disease: '{raw_disease}'")

            # Cases validation
            raw_cases = row[mapping["cases"]]
            try:
                cases = float(raw_cases)
                if cases < 0:
                    row_errors.append(f"Negative case count: {cases}")
                elif cases > IngestionConfig.MAX_PLAUSIBLE_WEEKLY_CASES:
                    row_errors.append(f"Implausibly high case count: {cases}")
            except (ValueError, TypeError):
                row_errors.append(f"Invalid non-numeric case value: '{raw_cases}'")
                cases = 0.0

            # Deaths validation
            deaths = 0.0
            if "deaths" in mapping and pd.notna(row[mapping["deaths"]]):
                try:
                    deaths = float(row[mapping["deaths"]])
                    if deaths < 0:
                        row_errors.append(f"Negative deaths count: {deaths}")
                    elif deaths > cases:
                        row_errors.append(f"Deaths ({deaths}) exceed cases ({cases})")
                except (ValueError, TypeError):
                    row_errors.append(f"Invalid non-numeric deaths value: '{row[mapping['deaths']]}'")

            # Reporting period normalization
            period_str = ""
            cal_year = None
            cal_week = None
            if "period" in mapping and pd.notna(row[mapping["period"]]):
                raw_period = str(row[mapping["period"]]).strip()
                if period_regex.match(raw_period):
                    period_str = raw_period
                    parts = raw_period.split("-W")
                    cal_year = int(parts[0])
                    cal_week = int(parts[1])
                else:
                    # Attempt simple conversion if format is like 2025-32
                    m = re.match(r"^(\d{4})[-_](\d{1,2})$", raw_period)
                    if m:
                        cal_year = int(m.group(1))
                        cal_week = int(m.group(2))
                        period_str = f"{cal_year}-W{cal_week:02d}"
                    else:
                        row_errors.append(f"Invalid reporting_period format: '{raw_period}'. Expected 'YYYY-Www'.")
            elif "year" in mapping and "week" in mapping:
                try:
                    cal_year = int(row[mapping["year"]])
                    cal_week = int(row[mapping["week"]])
                    if 2000 <= cal_year <= 2035 and 1 <= cal_week <= 53:
                        period_str = f"{cal_year}-W{cal_week:02d}"
                    else:
                        row_errors.append(f"Year ({cal_year}) or week ({cal_week}) out of bounds")
                except (ValueError, TypeError):
                    row_errors.append("Invalid numeric year/week specification")

            if row_errors:
                invalid_count += 1
                if len(validation_errors) < 10:
                    validation_errors.append(f"Row {idx + 1}: {'; '.join(row_errors)}")
            else:
                valid_rows.append(
                    {
                        "district_name": std_district,
                        "disease_name": std_disease,
                        "reporting_period": period_str,
                        "calendar_year": cal_year,
                        "week_number": cal_week,
                        "observed_cases": cases,
                        "observed_deaths": deaths,
                    }
                )

        if invalid_count > 10:
            validation_errors.append(f"... and {invalid_count - 10} additional invalid records.")

        df_clean = pd.DataFrame(valid_rows)
        valid_records = len(df_clean)

        return df_clean, total_records, valid_records, invalid_count, validation_errors
