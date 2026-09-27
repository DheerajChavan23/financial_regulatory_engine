"""Lakehouse analytics transformation engine using DuckDB.

Transforms cleansed Silver Parquet datasets into an enterprise Gold Star Schema,
producing dimension tables (dim_date, dim_geography, dim_policy, dim_claimant,
dim_incident_classification), the central fact_claims table, and a consolidated
analytical mart optimized for BI reporting and supervisory regulatory inspection.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import duckdb

# Ensure project root is in sys.path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings

logger = logging.getLogger("lakehouse_analytics")


class LakehouseAnalyticsEngine:
    """DuckDB-powered analytics engine modeling Silver lakehouse data into Gold Star Schema."""

    def __init__(self, database_path: Optional[str] = None) -> None:
        """Initialize in-memory or persistent DuckDB connection.

        Args:
            database_path: Optional path for persistent DuckDB file. Defaults to ':memory:'.
        """
        self.settings = get_settings()
        self.db_path = database_path or ":memory:"
        self.con = duckdb.connect(self.db_path)
        logger.info("Initialized LakehouseAnalyticsEngine (DuckDB database: '%s')", self.db_path)

    def build_gold_star_schema(
        self,
        silver_parquet_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Transform Silver Parquet records into Gold Star Schema dimension and fact tables.

        Args:
            silver_parquet_path: Path to Silver Parquet table.
            output_dir: Destination folder for Gold Star Schema Parquet files.

        Returns:
            Dict[str, Any]: Execution manifest with output file paths and table row counts.

        Raises:
            FileNotFoundError: If the source Silver Parquet file does not exist.
        """
        silver_path = silver_parquet_path or (self.settings.storage.silver_lakehouse_dir / "claims_silver.parquet")
        if not silver_path.exists():
            raise FileNotFoundError(f"Silver Parquet table not found at: {silver_path}")

        target_dir = output_dir or (self.settings.storage.gold_lakehouse_dir / "star_schema")
        target_dir.mkdir(parents=True, exist_ok=True)

        normalized_silver_path = str(silver_path.resolve()).replace("\\", "/")

        # Register Silver source view in DuckDB
        self.con.execute(f"CREATE OR REPLACE VIEW raw_silver AS SELECT * FROM read_parquet('{normalized_silver_path}')")
        silver_count = self.con.execute("SELECT COUNT(*) FROM raw_silver").fetchone()[0]
        logger.info("Loaded raw_silver view with %d records from %s", silver_count, silver_path.name)

        manifest: Dict[str, Any] = {
            "source_silver_file": str(silver_path.resolve()),
            "source_record_count": silver_count,
            "transformation_timestamp": datetime.now(timezone.utc).isoformat(),
            "tables": {},
        }

        # ---------------------------------------------------------------------
        # 1. Dimension: dim_date
        # ---------------------------------------------------------------------
        self.con.execute(
            """
            CREATE OR REPLACE TABLE dim_date AS
            WITH distinct_dates AS (
                SELECT CAST(filing_date AS DATE) AS full_date FROM raw_silver
                UNION
                SELECT CAST(incident_date AS DATE) AS full_date FROM raw_silver
            )
            SELECT
                CAST(strftime(full_date, '%Y%m%d') AS INTEGER) AS date_key,
                full_date,
                EXTRACT(YEAR FROM full_date) AS year,
                EXTRACT(QUARTER FROM full_date) AS quarter,
                EXTRACT(MONTH FROM full_date) AS month,
                strftime(full_date, '%B') AS month_name,
                EXTRACT(DAY FROM full_date) AS day_of_month,
                EXTRACT(DOW FROM full_date) AS day_of_week,
                strftime(full_date, '%A') AS day_name,
                CASE WHEN EXTRACT(DOW FROM full_date) IN (0, 6) THEN TRUE ELSE FALSE END AS is_weekend
            FROM distinct_dates
            ORDER BY full_date
            """
        )
        dim_date_path = target_dir / "gold_dim_date.parquet"
        self._export_table_to_parquet("dim_date", dim_date_path, manifest)

        # ---------------------------------------------------------------------
        # 2. Dimension: dim_geography (with Irish provincial mapping)
        # ---------------------------------------------------------------------
        self.con.execute(
            """
            CREATE OR REPLACE TABLE dim_geography AS
            WITH distinct_geo AS (
                SELECT DISTINCT
                    loss_county,
                    eircode_routing_key
                FROM raw_silver
            )
            SELECT
                ROW_NUMBER() OVER (ORDER BY loss_county, eircode_routing_key) AS geography_key,
                loss_county,
                CASE
                    WHEN loss_county IN ('Co. Dublin', 'Co. Kildare', 'Co. Meath', 'Co. Wicklow', 'Co. Wexford', 
                                         'Co. Carlow', 'Co. Kilkenny', 'Co. Laois', 'Co. Offaly', 'Co. Westmeath', 
                                         'Co. Longford', 'Co. Louth') THEN 'Leinster'
                    WHEN loss_county IN ('Co. Cork', 'Co. Kerry', 'Co. Limerick', 'Co. Tipperary', 
                                         'Co. Waterford', 'Co. Clare') THEN 'Munster'
                    WHEN loss_county IN ('Co. Galway', 'Co. Mayo', 'Co. Roscommon', 'Co. Sligo', 'Co. Leitrim') THEN 'Connacht'
                    WHEN loss_county IN ('Co. Donegal', 'Co. Cavan', 'Co. Monaghan', 'Co. Armagh', 
                                         'Co. Antrim', 'Co. Down', 'Co. Derry', 'Co. Fermanagh', 'Co. Tyrone') THEN 'Ulster'
                    ELSE 'Other / National'
                END AS province,
                eircode_routing_key,
                'Ireland' AS country
            FROM distinct_geo
            ORDER BY geography_key
            """
        )
        dim_geo_path = target_dir / "gold_dim_geography.parquet"
        self._export_table_to_parquet("dim_geography", dim_geo_path, manifest)

        # ---------------------------------------------------------------------
        # 3. Dimension: dim_incident_classification
        # ---------------------------------------------------------------------
        self.con.execute(
            """
            CREATE OR REPLACE TABLE dim_incident_classification AS
            WITH distinct_types AS (
                SELECT DISTINCT
                    incident_type,
                    dora_classification
                FROM raw_silver
            )
            SELECT
                ROW_NUMBER() OVER (ORDER BY incident_type, dora_classification) AS incident_type_key,
                incident_type,
                dora_classification,
                CASE
                    WHEN dora_classification LIKE '%Major ICT%' THEN 'Tier 1 - Major Incident'
                    WHEN dora_classification LIKE '%Significant%' THEN 'Tier 2 - Significant Disruption'
                    ELSE 'Tier 3 - Standard Loss'
                END AS dora_regulatory_tier,
                CASE
                    WHEN incident_type LIKE '%Cyber%' OR incident_type LIKE '%ICT%' THEN TRUE
                    ELSE FALSE
                END AS is_ict_related
            FROM distinct_types
            ORDER BY incident_type_key
            """
        )
        dim_inc_path = target_dir / "gold_dim_incident_classification.parquet"
        self._export_table_to_parquet("dim_incident_classification", dim_inc_path, manifest)

        # ---------------------------------------------------------------------
        # 4. Dimension: dim_policy
        # ---------------------------------------------------------------------
        self.con.execute(
            """
            CREATE OR REPLACE TABLE dim_policy AS
            WITH distinct_policies AS (
                SELECT DISTINCT
                    policy_number,
                    governance_status
                FROM raw_silver
            )
            SELECT
                ROW_NUMBER() OVER (ORDER BY policy_number) AS policy_key,
                policy_number,
                governance_status,
                'ACTIVE_UNDERWRITTEN' AS contract_status,
                'Emerald Shield Assurance DAC' AS underwriting_entity
            FROM distinct_policies
            ORDER BY policy_key
            """
        )
        dim_pol_path = target_dir / "gold_dim_policy.parquet"
        self._export_table_to_parquet("dim_policy", dim_pol_path, manifest)

        # ---------------------------------------------------------------------
        # 5. Dimension: dim_claimant (GDPR pseudonymized)
        # ---------------------------------------------------------------------
        self.con.execute(
            """
            CREATE OR REPLACE TABLE dim_claimant AS
            WITH distinct_claimants AS (
                SELECT DISTINCT
                    claimant_ppsn_hash,
                    claimant_name_masked,
                    claimant_email_masked,
                    claimant_phone_masked
                FROM raw_silver
            )
            SELECT
                ROW_NUMBER() OVER (ORDER BY claimant_ppsn_hash) AS claimant_key,
                claimant_ppsn_hash,
                claimant_name_masked,
                SPLIT_PART(claimant_email_masked, '@', 2) AS email_domain,
                SUBSTRING(claimant_phone_masked, 1, 7) AS phone_country_prefix,
                'CONFIDENTIAL_INDIVIDUAL' AS claimant_category
            FROM distinct_claimants
            ORDER BY claimant_key
            """
        )
        dim_clm_path = target_dir / "gold_dim_claimant.parquet"
        self._export_table_to_parquet("dim_claimant", dim_clm_path, manifest)

        # ---------------------------------------------------------------------
        # 6. Central Fact Table: fact_claims
        # ---------------------------------------------------------------------
        self.con.execute(
            """
            CREATE OR REPLACE TABLE fact_claims AS
            SELECT
                ROW_NUMBER() OVER (ORDER BY s.claim_id) AS claim_key,
                s.claim_id,
                -- Foreign Key Joins
                fd.date_key AS filing_date_key,
                id.date_key AS incident_date_key,
                g.geography_key,
                ic.incident_type_key,
                p.policy_key,
                c.claimant_key,
                -- Financial Measures
                s.total_loss_amount AS gross_loss_amount,
                s.total_deductible AS deductible_amount,
                s.total_net_claimed AS net_claimed_amount,
                s.currency,
                -- Operational Metrics
                s.days_to_notify,
                s.quality_score,
                s.num_line_items,
                -- Statutory Compliance Indicators
                CASE WHEN s.days_to_notify <= 90 THEN 1 ELSE 0 END AS is_timely_sla_compliant,
                CASE WHEN s.dora_classification LIKE '%Major ICT%' THEN 1 ELSE 0 END AS is_major_dora_incident,
                -- Audit Provenance
                s.ingestion_id,
                s.source_filename,
                s.source_file_sha256,
                CURRENT_TIMESTAMP AS gold_load_timestamp
            FROM raw_silver s
            JOIN dim_date fd ON CAST(s.filing_date AS DATE) = fd.full_date
            JOIN dim_date id ON CAST(s.incident_date AS DATE) = id.full_date
            JOIN dim_geography g ON s.loss_county = g.loss_county AND s.eircode_routing_key = g.eircode_routing_key
            JOIN dim_incident_classification ic ON s.incident_type = ic.incident_type AND s.dora_classification = ic.dora_classification
            JOIN dim_policy p ON s.policy_number = p.policy_number
            JOIN dim_claimant c ON s.claimant_ppsn_hash = c.claimant_ppsn_hash
            ORDER BY claim_key
            """
        )
        fact_claims_path = target_dir / "gold_fact_claims.parquet"
        self._export_table_to_parquet("fact_claims", fact_claims_path, manifest)

        # ---------------------------------------------------------------------
        # 7. Consolidated Analytical Mart View / Parquet Table
        # ---------------------------------------------------------------------
        mart_path = self.settings.storage.gold_lakehouse_dir / "gold_claims_analytical_mart.parquet"
        normalized_mart_path = str(mart_path.resolve()).replace("\\", "/")
        self.con.execute(
            f"""
            CREATE OR REPLACE TABLE gold_claims_analytical_mart AS
            SELECT
                f.claim_key,
                f.claim_id,
                p.policy_number,
                fd.full_date AS filing_date,
                id.full_date AS incident_date,
                f.days_to_notify,
                f.is_timely_sla_compliant,
                ic.incident_type,
                ic.dora_classification,
                ic.dora_regulatory_tier,
                ic.is_ict_related,
                g.loss_county,
                g.province,
                g.eircode_routing_key,
                c.claimant_ppsn_hash,
                c.claimant_name_masked,
                c.email_domain,
                f.gross_loss_amount,
                f.deductible_amount,
                f.net_claimed_amount,
                f.currency,
                f.quality_score,
                f.num_line_items,
                f.is_major_dora_incident,
                f.ingestion_id,
                f.source_filename,
                f.gold_load_timestamp
            FROM fact_claims f
            JOIN dim_date fd ON f.filing_date_key = fd.date_key
            JOIN dim_date id ON f.incident_date_key = id.date_key
            JOIN dim_geography g ON f.geography_key = g.geography_key
            JOIN dim_incident_classification ic ON f.incident_type_key = ic.incident_type_key
            JOIN dim_policy p ON f.policy_key = p.policy_key
            JOIN dim_claimant c ON f.claimant_key = c.claimant_key
            ORDER BY f.claim_key
            """
        )
        self.con.execute(f"COPY gold_claims_analytical_mart TO '{normalized_mart_path}' (FORMAT 'parquet', CODEC 'SNAPPY')")
        mart_count = self.con.execute("SELECT COUNT(*) FROM gold_claims_analytical_mart").fetchone()[0]
        manifest["tables"]["gold_claims_analytical_mart"] = {
            "path": str(mart_path.resolve()),
            "rows": mart_count,
            "type": "consolidated_mart",
        }
        logger.info("Saved Gold analytical mart with %d records to: %s", mart_count, mart_path.name)

        # Write manifest file
        manifest_path = target_dir / "star_schema_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)

        return manifest

    def _export_table_to_parquet(self, table_name: str, output_path: Path, manifest: Dict[str, Any]) -> None:
        """Export a DuckDB internal table to a Parquet file and register in manifest."""
        normalized_output = str(output_path.resolve()).replace("\\", "/")
        self.con.execute(f"COPY {table_name} TO '{normalized_output}' (FORMAT 'parquet', CODEC 'SNAPPY')")
        row_count = self.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        manifest["tables"][table_name] = {
            "path": str(output_path.resolve()),
            "rows": row_count,
            "type": "star_schema_table",
        }
        logger.info("Exported Gold table '%s' (%d rows) -> %s", table_name, row_count, output_path.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    print("=" * 85)
    print("Gold Lakehouse Star Schema Transformation (DuckDB OLAP Engine)")
    print("=" * 85)

    engine = LakehouseAnalyticsEngine()
    result = engine.build_gold_star_schema()

    print(f"\nStar Schema Transformation Complete:")
    print(f"Source Records Processed: {result['source_record_count']}")
    print("-" * 85)
    print(f"{'Table Name':<32} | {'Row Count':<10} | {'Destination File'}")
    print("-" * 85)
    for tbl_name, meta in result["tables"].items():
        fname = Path(meta["path"]).name
        print(f"{tbl_name:<32} | {meta['rows']:<10} | {fname}")

    print("-" * 85)
    # Quick SQL preview through DuckDB
    preview_df = engine.con.execute(
        """
        SELECT 
            claim_id,
            province,
            incident_type,
            dora_regulatory_tier,
            net_claimed_amount,
            is_timely_sla_compliant
        FROM gold_claims_analytical_mart
        LIMIT 5
        """
    ).df()
    print("\nGold Analytical Mart Sample Preview:")
    print(preview_df.to_string())
    print("=" * 85)
