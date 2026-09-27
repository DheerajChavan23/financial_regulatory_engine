"""Business Intelligence (BI) export engine with DAX metrics for Power BI and Tableau.

Generates tabular model specifications, entity-relationship schemas, CSV/Parquet data feeds,
and an extensive catalog of production DAX (Data Analysis Expressions) measures covering
financial liquidation, DORA Article 18 exposure, and CBI statutory SLA compliance.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd

# Ensure project root is in sys.path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings

logger = logging.getLogger("bi_exporter")


@dataclass
class DAXMeasureDefinition:
    """Specification of an enterprise DAX calculation measure."""

    name: str
    dax_formula: str
    category: str
    format_string: str
    description: str


class BIExporter:
    """Exporter orchestrating Gold lakehouse datasets and DAX measures for BI platforms."""

    def __init__(self, star_schema_dir: Optional[Path] = None, export_dir: Optional[Path] = None) -> None:
        """Initialize BI exporter with source Gold Star Schema and target export paths.

        Args:
            star_schema_dir: Folder containing Gold Star Schema Parquet tables.
            export_dir: Destination folder for BI exports and DAX scripts.
        """
        app_settings = get_settings()
        self.star_schema_dir: Path = star_schema_dir or (app_settings.storage.gold_lakehouse_dir / "star_schema")
        self.export_dir: Path = export_dir or (app_settings.storage.gold_lakehouse_dir / "bi_export")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.measures_catalog: List[DAXMeasureDefinition] = self._build_dax_measures_catalog()

    def _build_dax_measures_catalog(self) -> List[DAXMeasureDefinition]:
        """Compile a comprehensive catalog of production DAX measures."""
        return [
            # Financial KPIs
            DAXMeasureDefinition(
                name="Total Gross Loss",
                dax_formula="SUM('fact_claims'[gross_loss_amount])",
                category="Financial KPIs",
                format_string="€#,##0.00",
                description="Cumulative gross financial loss before deductibles across all audited claims.",
            ),
            DAXMeasureDefinition(
                name="Total Deductibles Applied",
                dax_formula="SUM('fact_claims'[deductible_amount])",
                category="Financial KPIs",
                format_string="€#,##0.00",
                description="Cumulative policyholder deductibles subtracted from gross loss.",
            ),
            DAXMeasureDefinition(
                name="Total Net Claim Amount",
                dax_formula="SUM('fact_claims'[net_claimed_amount])",
                category="Financial KPIs",
                format_string="€#,##0.00",
                description="Total net indemnity compensation requested for financial liquidation.",
            ),
            DAXMeasureDefinition(
                name="Average Net Claim Value",
                dax_formula="AVERAGE('fact_claims'[net_claimed_amount])",
                category="Financial KPIs",
                format_string="€#,##0.00",
                description="Mean net financial liquidation value per claim.",
            ),
            DAXMeasureDefinition(
                name="Max Single Claim Exposure",
                dax_formula="MAX('fact_claims'[net_claimed_amount])",
                category="Financial KPIs",
                format_string="€#,##0.00",
                description="Highest individual net claim liquidation exposure in the active portfolio.",
            ),
            # Operational & Quality KPIs
            DAXMeasureDefinition(
                name="Total Claims Count",
                dax_formula="COUNTROWS('fact_claims')",
                category="Operational Metrics",
                format_string="#,##0",
                description="Total count of successfully ingested and promoted Gold claim dossiers.",
            ),
            DAXMeasureDefinition(
                name="Average Notice Latency Days",
                dax_formula="AVERAGE('fact_claims'[days_to_notify])",
                category="Operational Metrics",
                format_string="0.0",
                description="Average elapsed calendar days between incident occurrence and claim filing.",
            ),
            DAXMeasureDefinition(
                name="Average Data Quality Score",
                dax_formula="AVERAGE('fact_claims'[quality_score])",
                category="Operational Metrics",
                format_string="0.0%",
                description="Mean data governance quality score across mandatory schema rules.",
            ),
            DAXMeasureDefinition(
                name="Total Line Items Count",
                dax_formula="SUM('fact_claims'[num_line_items])",
                category="Operational Metrics",
                format_string="#,##0",
                description="Total itemized schedule entries parsed across all claim dossiers.",
            ),
            DAXMeasureDefinition(
                name="Average Items Per Claim",
                dax_formula="DIVIDE([Total Line Items Count], [Total Claims Count], 0)",
                category="Operational Metrics",
                format_string="0.0",
                description="Average number of itemized expense categories per claim form.",
            ),
            # Regulatory & DORA Compliance Measures
            DAXMeasureDefinition(
                name="SLA Compliant Claims Count",
                dax_formula="CALCULATE(COUNTROWS('fact_claims'), 'fact_claims'[is_timely_sla_compliant] = 1)",
                category="DORA & Regulatory Compliance",
                format_string="#,##0",
                description="Count of claims lodged within statutory CBI notification window (<= 90 days).",
            ),
            DAXMeasureDefinition(
                name="Statutory SLA Compliance Rate",
                dax_formula="DIVIDE([SLA Compliant Claims Count], [Total Claims Count], 0)",
                category="DORA & Regulatory Compliance",
                format_string="0.0%",
                description="Percentage of claims meeting Central Bank of Ireland timely notification SLAs.",
            ),
            DAXMeasureDefinition(
                name="Major DORA Incidents Count",
                dax_formula="CALCULATE(COUNTROWS('fact_claims'), 'fact_claims'[is_major_dora_incident] = 1)",
                category="DORA & Regulatory Compliance",
                format_string="#,##0",
                description="Number of claims classified under EU DORA Article 18 as Major ICT Incidents.",
            ),
            DAXMeasureDefinition(
                name="Major DORA Total Exposure",
                dax_formula="CALCULATE([Total Net Claim Amount], 'fact_claims'[is_major_dora_incident] = 1)",
                category="DORA & Regulatory Compliance",
                format_string="€#,##0.00",
                description="Total liquidation exposure stemming exclusively from Major DORA ICT incidents.",
            ),
            DAXMeasureDefinition(
                name="Major DORA Exposure Share",
                dax_formula="DIVIDE([Major DORA Total Exposure], [Total Net Claim Amount], 0)",
                category="DORA & Regulatory Compliance",
                format_string="0.0%",
                description="Share of total financial indemnity exposure attributable to Major DORA incidents.",
            ),
            DAXMeasureDefinition(
                name="ICT Related Claims Exposure",
                dax_formula=(
                    "CALCULATE([Total Net Claim Amount], "
                    "FILTER('dim_incident_classification', 'dim_incident_classification'[is_ict_related] = TRUE()))"
                ),
                category="DORA & Regulatory Compliance",
                format_string="€#,##0.00",
                description="Aggregate financial claims generated by technological, cyber, or ICT perils.",
            ),
            # Time Intelligence DAX Measures
            DAXMeasureDefinition(
                name="Claims Net Amount MTD",
                dax_formula="TOTALMTD([Total Net Claim Amount], 'dim_date'[full_date])",
                category="Time Intelligence",
                format_string="€#,##0.00",
                description="Month-to-date cumulative net claims liquidation total.",
            ),
            DAXMeasureDefinition(
                name="Claims Net Amount YTD",
                dax_formula="TOTALYTD([Total Net Claim Amount], 'dim_date'[full_date])",
                category="Time Intelligence",
                format_string="€#,##0.00",
                description="Year-to-date cumulative net claims liquidation total.",
            ),
            DAXMeasureDefinition(
                name="Claims Count MTD",
                dax_formula="TOTALMTD([Total Claims Count], 'dim_date'[full_date])",
                category="Time Intelligence",
                format_string="#,##0",
                description="Month-to-date cumulative count of claims filed.",
            ),
        ]

    def export_dax_script(self) -> Path:
        """Write formatted DAX measures script file for Power BI model import.

        Returns:
            Path: Path to dax_measures.dax file.
        """
        output_path = self.export_dir / "dax_measures.dax"
        lines: List[str] = [
            "// ============================================================================",
            "// ENTERPRISE REGULATORY & OPERATIONAL INGESTION ENGINE - DAX MEASURES CATALOG",
            "// Target Platform: Power BI Desktop / Microsoft Fabric / SSAS Tabular",
            f"// Generated Timestamp: {datetime.now(timezone.utc).isoformat()}",
            "// ============================================================================\n",
        ]

        # Group by category
        categories: Dict[str, List[DAXMeasureDefinition]] = {}
        for m in self.measures_catalog:
            categories.setdefault(m.category, []).append(m)

        for cat, measures in categories.items():
            lines.append(f"// --- CATEGORY: {cat.upper()} ---")
            for m in measures:
                lines.append(f"// Description: {m.description}")
                lines.append(f"// Format: {m.format_string}")
                lines.append(f"[{m.name}] = \n    {m.dax_formula}\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info("Exported %d DAX measures to: %s", len(self.measures_catalog), output_path.name)
        return output_path

    def export_tabular_model_schema(self) -> Path:
        """Generate Power BI Tabular Object Model JSON specification with relationships and measures.

        Returns:
            Path: Path to powerbi_tabular_model.json file.
        """
        schema = {
            "name": "FinancialIngestionLakehouseModel",
            "compatibilityLevel": 1550,
            "createdDate": datetime.now(timezone.utc).isoformat(),
            "description": "Star Schema Tabular Model for Financial Claims Ingestion & DORA Compliance Audit.",
            "tables": [
                {
                    "name": "fact_claims",
                    "sourceFile": "fact_claims.parquet",
                    "description": "Central financial and operational claims liquidation fact table.",
                    "measures": [
                        {
                            "name": m.name,
                            "expression": m.dax_formula,
                            "displayFolder": m.category,
                            "formatString": m.format_string,
                            "description": m.description,
                        }
                        for m in self.measures_catalog
                    ],
                },
                {"name": "dim_date", "sourceFile": "dim_date.parquet", "description": "Calendar dimension."},
                {"name": "dim_geography", "sourceFile": "dim_geography.parquet", "description": "Irish geographic hierarchy."},
                {"name": "dim_policy", "sourceFile": "dim_policy.parquet", "description": "Insurance policy dimension."},
                {"name": "dim_claimant", "sourceFile": "dim_claimant.parquet", "description": "Pseudonymized claimant dimension."},
                {
                    "name": "dim_incident_classification",
                    "sourceFile": "dim_incident_classification.parquet",
                    "description": "DORA regulatory incident classification hierarchy.",
                },
            ],
            "relationships": [
                {
                    "fromTable": "fact_claims",
                    "fromColumn": "filing_date_key",
                    "toTable": "dim_date",
                    "toColumn": "date_key",
                    "name": "Rel_Fact_Date_Filing",
                },
                {
                    "fromTable": "fact_claims",
                    "fromColumn": "geography_key",
                    "toTable": "dim_geography",
                    "toColumn": "geography_key",
                    "name": "Rel_Fact_Geography",
                },
                {
                    "fromTable": "fact_claims",
                    "fromColumn": "incident_type_key",
                    "toTable": "dim_incident_classification",
                    "toColumn": "incident_type_key",
                    "name": "Rel_Fact_IncidentClassification",
                },
                {
                    "fromTable": "fact_claims",
                    "fromColumn": "policy_key",
                    "toTable": "dim_policy",
                    "toColumn": "policy_key",
                    "name": "Rel_Fact_Policy",
                },
                {
                    "fromTable": "fact_claims",
                    "fromColumn": "claimant_key",
                    "toTable": "dim_claimant",
                    "toColumn": "claimant_key",
                    "name": "Rel_Fact_Claimant",
                },
            ],
        }

        output_path = self.export_dir / "powerbi_tabular_model.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

        logger.info("Exported Tabular Model schema to: %s", output_path.name)
        return output_path

    def export_csv_and_parquet_feeds(self) -> Dict[str, Path]:
        """Convert Gold Parquet tables into clean CSV and Parquet feeds for direct BI tool consumption.

        Returns:
            Dict[str, Path]: Dictionary mapping table names to their exported CSV file paths.
        """
        table_files = list(self.star_schema_dir.glob("gold_*.parquet"))
        # Also include consolidated mart if present
        mart_file = self.star_schema_dir.parent / "gold_claims_analytical_mart.parquet"
        if mart_file.exists():
            table_files.append(mart_file)

        exported_feeds: Dict[str, Path] = {}
        for pfile in table_files:
            # Table name without 'gold_' prefix for standard BI naming
            clean_name = pfile.stem.replace("gold_", "")
            df = pd.read_parquet(pfile)

            csv_path = self.export_dir / f"{clean_name}.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
            exported_feeds[clean_name] = csv_path
            logger.info("Exported BI feed: %s (%d rows)", csv_path.name, len(df))

        return exported_feeds

    def compute_kpi_summary(self) -> Dict[str, Any]:
        """Compute the live values of the DAX measures locally using DuckDB.

        Returns:
            Dict[str, Any]: Real-time evaluated KPI metric cards.
        """
        mart_path = self.star_schema_dir.parent / "gold_claims_analytical_mart.parquet"
        normalized_path = str(mart_path.resolve()).replace("\\", "/")

        con = duckdb.connect(":memory:")
        con.execute(f"CREATE VIEW mart AS SELECT * FROM read_parquet('{normalized_path}')")

        metrics_query = """
        SELECT
            COUNT(*) AS total_claims,
            ROUND(SUM(gross_loss_amount), 2) AS total_gross_loss,
            ROUND(SUM(deductible_amount), 2) AS total_deductibles,
            ROUND(SUM(net_claimed_amount), 2) AS total_net_claimed,
            ROUND(AVG(net_claimed_amount), 2) AS avg_claim_value,
            ROUND(MAX(net_claimed_amount), 2) AS max_single_claim,
            ROUND(AVG(days_to_notify), 1) AS avg_latency_days,
            ROUND(AVG(quality_score) * 100, 1) AS avg_quality_pct,
            SUM(num_line_items) AS total_items,
            SUM(CASE WHEN is_timely_sla_compliant = 1 THEN 1 ELSE 0 END) AS sla_compliant_claims,
            SUM(CASE WHEN is_major_dora_incident = 1 THEN 1 ELSE 0 END) AS major_dora_claims,
            ROUND(SUM(CASE WHEN is_major_dora_incident = 1 THEN net_claimed_amount ELSE 0 END), 2) AS major_dora_exposure,
            ROUND(SUM(CASE WHEN is_ict_related = TRUE THEN net_claimed_amount ELSE 0 END), 2) AS ict_related_exposure
        FROM mart
        """
        row = con.execute(metrics_query).fetchone()

        total_claims = row[0]
        total_gross = row[1]
        total_ded = row[2]
        total_net = row[3]
        avg_val = row[4]
        max_val = row[5]
        avg_lat = row[6]
        avg_qual = row[7]
        total_items = row[8]
        sla_comp = row[9]
        dora_claims = row[10]
        dora_exp = row[11]
        ict_exp = row[12]

        sla_pct = round((sla_comp / total_claims) * 100.0, 1) if total_claims else 0.0
        dora_ratio = round((dora_exp / total_net) * 100.0, 1) if total_net else 0.0
        ict_ratio = round((ict_exp / total_net) * 100.0, 1) if total_net else 0.0

        summary = {
            "Total Claims": total_claims,
            "Total Gross Loss": f"EUR {total_gross:,.2f}",
            "Total Deductibles Applied": f"EUR {total_ded:,.2f}",
            "Total Net Claim Amount": f"EUR {total_net:,.2f}",
            "Average Net Claim Value": f"EUR {avg_val:,.2f}",
            "Max Single Claim Exposure": f"EUR {max_val:,.2f}",
            "Average Notice Latency": f"{avg_lat} days",
            "Average Data Quality Score": f"{avg_qual}%",
            "Statutory SLA Compliance Rate": f"{sla_pct}%",
            "Major DORA Incidents": dora_claims,
            "Major DORA Total Exposure": f"EUR {dora_exp:,.2f} ({dora_ratio}% of portfolio)",
            "ICT Related Claims Exposure": f"EUR {ict_exp:,.2f} ({ict_ratio}% of portfolio)",
        }

        # Write KPI summary JSON
        kpi_path = self.export_dir / "kpi_executive_summary.json"
        with open(kpi_path, "w", encoding="utf-8") as kf:
            json.dump(summary, kf, indent=2)

        return summary

    def run_full_export(self) -> Dict[str, Any]:
        """Execute complete BI export workflow: DAX, tabular model, data feeds, and KPI report.

        Returns:
            Dict[str, Any]: Consolidated export summary and artifact manifest.
        """
        dax_file = self.export_dax_script()
        model_file = self.export_tabular_model_schema()
        feeds = self.export_csv_and_parquet_feeds()
        kpi_report = self.compute_kpi_summary()

        return {
            "dax_measures_count": len(self.measures_catalog),
            "dax_script_path": str(dax_file.resolve()),
            "tabular_model_path": str(model_file.resolve()),
            "exported_feeds": {k: str(v.resolve()) for k, v in feeds.items()},
            "kpi_summary": kpi_report,
            "export_directory": str(self.export_dir.resolve()),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    print("=" * 85)
    print("Power BI & Tableau BI Exporter with DAX Measures Catalog")
    print("=" * 85)

    exporter = BIExporter()
    summary = exporter.run_full_export()

    print(f"\nBI Export Completed Successfully:")
    print(f"Export Destination:      {summary['export_directory']}")
    print(f"Total DAX Measures:      {summary['dax_measures_count']}")
    print(f"DAX Script File:         {Path(summary['dax_script_path']).name}")
    print(f"Tabular Model Schema:    {Path(summary['tabular_model_path']).name}")
    print(f"Data Feeds Generated:    {len(summary['exported_feeds'])} tables")
    print("-" * 85)

    print("\nExecutive KPI Summary (Evaluated Live via DAX Logic):")
    for kpi, val in summary["kpi_summary"].items():
        print(f"  • {kpi:<32}: {val}")

    print("-" * 85)
    print("\nSample DAX Measures Preview (from dax_measures.dax):")
    with open(summary["dax_script_path"], "r", encoding="utf-8") as df:
        lines = df.readlines()
    for l in lines[9:27]:
        print("  " + l.rstrip())
    print("=" * 85)
