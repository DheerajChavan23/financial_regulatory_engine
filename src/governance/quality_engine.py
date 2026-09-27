"""Data quality and governance engine for Silver Lakehouse promotion.

Validates schema constraints, statutory business rules, notification SLAs, and financial
reconciliations. Routes fully conformant, PII-pseudonymized records into columnar
Silver Parquet storage while isolating non-conformant/anomalous dossiers into a quarantine audit pool.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure project root is in sys.path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.extraction.schemas import BronzeLakehouseEnvelope
from src.governance.pii_redactor import GovernedClaimRecord, PIIRedactor

logger = logging.getLogger("quality_engine")


@dataclass
class QualityCheckResult:
    """Outcome of an individual data quality rule evaluation."""

    rule_name: str
    passed: bool
    description: str
    details: Optional[str] = None


@dataclass
class QualityEvaluationReport:
    """Composite quality audit report for a claim dossier."""

    claim_id: str
    is_valid: bool
    quality_score: float
    passed_rules: List[str]
    failed_rules: List[str]
    check_results: List[QualityCheckResult]


class DataQualityEngine:
    """Enterprise data quality evaluation engine enforcing financial and regulatory rules."""

    def __init__(
        self,
        min_quality_score: Optional[float] = None,
        max_notification_days: int = 90,
    ) -> None:
        """Initialize engine with configurable quality score threshold and SLA limits.

        Args:
            min_quality_score: Minimum ratio (0.0 - 1.0) of passed checks required for Silver promotion.
            max_notification_days: Statutory notice window threshold under DORA/CBI (default: 90).
        """
        settings = get_settings()
        self.min_quality_score: float = min_quality_score or settings.governance.min_quality_score
        self.max_notification_days: int = max_notification_days
        self.allowed_currencies: List[str] = settings.governance.allowed_currencies

    def evaluate_record(self, record: GovernedClaimRecord) -> QualityEvaluationReport:
        """Execute a comprehensive battery of quality and business integrity checks.

        Evaluates:
        1. Mandatory Policy Identifier (RULE_POLICY_NUMBER_PRESENT)
        2. Positive Financial Amounts (RULE_POSITIVE_AMOUNTS)
        3. Statutory SLA Notification Window (RULE_TIMELY_NOTIFICATION)
        4. Permitted ISO Currency (RULE_ALLOWED_CURRENCY)
        5. Valid National Tax Identifier Format (RULE_VALID_PPSN_FORMAT)
        6. Itemized Financial Reconciliation (RULE_LEDGER_RECONCILED)
        7. Mandatory Metadata Completeness (RULE_METADATA_COMPLETENESS)

        Args:
            record: PII-redacted GovernedClaimRecord instance.

        Returns:
            QualityEvaluationReport: Detailed assessment with boolean verdict and score.
        """
        checks: List[QualityCheckResult] = []

        # 1. Mandatory Policy Number Check
        has_policy = bool(record.policy_number and record.policy_number.strip())
        checks.append(
            QualityCheckResult(
                rule_name="RULE_POLICY_NUMBER_PRESENT",
                passed=has_policy,
                description="Insurance policy identifier must be populated and non-null.",
                details=f"Policy number: '{record.policy_number}'",
            )
        )

        # 2. Positive Financial Amount Check
        positive_amounts = (
            record.total_net_claimed > 0
            and record.total_loss_amount > 0
            and all(item.get("amount", 0) > 0 for item in record.line_items)
            and all(item.get("net_claimed", 0) > 0 for item in record.line_items)
        )
        checks.append(
            QualityCheckResult(
                rule_name="RULE_POSITIVE_AMOUNTS",
                passed=positive_amounts,
                description="Financial claim loss amounts and net requested values must be strictly positive.",
                details=f"Total Net Claimed: {record.total_net_claimed}, Total Loss: {record.total_loss_amount}",
            )
        )

        # 3. Notification Latency SLA Check
        timely = record.days_to_notify <= self.max_notification_days
        checks.append(
            QualityCheckResult(
                rule_name="RULE_TIMELY_NOTIFICATION",
                passed=timely,
                description=f"Incident notification must not exceed statutory limit of {self.max_notification_days} days.",
                details=f"Elapsed days to notice: {record.days_to_notify} (Max allowed: {self.max_notification_days})",
            )
        )

        # 4. Permitted Currency Check
        valid_currency = record.currency in self.allowed_currencies
        checks.append(
            QualityCheckResult(
                rule_name="RULE_ALLOWED_CURRENCY",
                passed=valid_currency,
                description=f"Currency must be an authorized ISO code: {self.allowed_currencies}.",
                details=f"Document currency: '{record.currency}'",
            )
        )

        # 5. PPSN Format Check
        checks.append(
            QualityCheckResult(
                rule_name="RULE_VALID_PPSN_FORMAT",
                passed=record.ppsn_format_valid,
                description="Irish Personal Public Service Number must adhere to statutory 7-digit + letter syntax.",
                details=f"PPSN Valid Format: {record.ppsn_format_valid}",
            )
        )

        # 6. Itemized Ledger Reconciliation
        if record.line_items:
            sum_line_items = round(sum(item.get("net_claimed", 0.0) for item in record.line_items), 2)
            reconciled = abs(sum_line_items - record.total_net_claimed) < 0.05
        else:
            sum_line_items = 0.0
            reconciled = False
        checks.append(
            QualityCheckResult(
                rule_name="RULE_LEDGER_RECONCILED",
                passed=reconciled,
                description="Sum of itemized net claim values must reconcile with stated aggregate liquidation total.",
                details=f"Line items sum: {sum_line_items}, Aggregate total: {record.total_net_claimed}",
            )
        )

        # 7. Metadata Completeness Check
        complete_meta = bool(
            record.claim_id
            and record.filing_date
            and record.incident_date
            and record.incident_type
            and record.loss_county != "Unknown County"
        )
        checks.append(
            QualityCheckResult(
                rule_name="RULE_METADATA_COMPLETENESS",
                passed=complete_meta,
                description="Core operational metadata (dates, claim type, regional county) must be non-empty.",
                details=f"Metadata completeness verified: {complete_meta}",
            )
        )

        passed_rules = [c.rule_name for c in checks if c.passed]
        failed_rules = [c.rule_name for c in checks if not c.passed]

        quality_score = round(len(passed_rules) / len(checks), 4)

        # Strict gating: All critical rules (policy, positive amount, SLA) must pass, plus score >= threshold
        critical_rules = {"RULE_POLICY_NUMBER_PRESENT", "RULE_POSITIVE_AMOUNTS", "RULE_TIMELY_NOTIFICATION"}
        critical_passed = critical_rules.issubset(set(passed_rules))
        is_valid = critical_passed and (quality_score >= self.min_quality_score)

        return QualityEvaluationReport(
            claim_id=record.claim_id,
            is_valid=is_valid,
            quality_score=quality_score,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            check_results=checks,
        )


def process_bronze_to_silver(
    bronze_dir: Optional[Path] = None,
    silver_dir: Optional[Path] = None,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Govern all Bronze Lakehouse envelopes and partition into Silver Parquet and Quarantine pools.

    Args:
        bronze_dir: Ingestion directory with Bronze JSON envelopes (defaults to 02_bronze_lakehouse).
        silver_dir: Destination directory for Silver Parquet files (defaults to 03_silver_lakehouse).

    Returns:
        Tuple[Path, Path, Dict[str, Any]]: Paths to (silver_parquet, quarantine_json) and execution summary.
    """
    settings = get_settings()
    src_dir = bronze_dir or settings.storage.bronze_lakehouse_dir
    dst_dir = silver_dir or settings.storage.silver_lakehouse_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    redactor = PIIRedactor()
    quality_engine = DataQualityEngine()

    bronze_files = sorted(src_dir.glob("bronze_*.json"))
    if not bronze_files:
        logger.warning("No Bronze JSON files found at: %s", src_dir)
        return dst_dir, dst_dir, {"total_processed": 0, "promoted": 0, "quarantined": 0}

    silver_rows: List[Dict[str, Any]] = []
    quarantine_rows: List[Dict[str, Any]] = []

    for b_file in bronze_files:
        try:
            with open(b_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            envelope = BronzeLakehouseEnvelope.model_validate(data)

            # Step 1: PII Redaction and Pseudonymization
            governed_record = redactor.govern_envelope(envelope)

            # Step 2: Quality Engine Evaluation
            report = quality_engine.evaluate_record(governed_record)

            # Build standardized tabular record for Lakehouse
            base_dict = {
                "claim_id": governed_record.claim_id,
                "policy_number": governed_record.policy_number,
                "filing_date": governed_record.filing_date,
                "incident_date": governed_record.incident_date,
                "days_to_notify": governed_record.days_to_notify,
                "dora_classification": governed_record.dora_classification,
                "incident_type": governed_record.incident_type,
                "claimant_ppsn_hash": governed_record.claimant_ppsn_hash,
                "claimant_name_masked": governed_record.claimant_name_masked,
                "claimant_email_masked": governed_record.claimant_email_masked,
                "claimant_phone_masked": governed_record.claimant_phone_masked,
                "loss_county": governed_record.loss_county,
                "eircode_routing_key": governed_record.eircode_routing_key,
                "currency": governed_record.currency,
                "total_loss_amount": governed_record.total_loss_amount,
                "total_deductible": governed_record.total_deductible,
                "total_net_claimed": governed_record.total_net_claimed,
                "num_line_items": len(governed_record.line_items),
                "incident_description_clean": governed_record.incident_description_clean,
                "statutory_attestation": governed_record.statutory_attestation,
                "quality_score": report.quality_score,
                "source_filename": governed_record.source_filename,
                "source_file_sha256": governed_record.source_file_sha256,
                "ingestion_id": governed_record.ingestion_id,
                "lakehouse_promotion_timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if report.is_valid:
                base_dict["governance_status"] = "VALID_PROMOTED"
                silver_rows.append(base_dict)
            else:
                base_dict["governance_status"] = "QUARANTINED"
                base_dict["failed_rules"] = report.failed_rules
                base_dict["failure_reasons"] = [
                    f"{r.rule_name}: {r.details}" for r in report.check_results if not r.passed
                ]
                quarantine_rows.append(base_dict)

        except Exception as exc:
            logger.error("Failed processing Bronze record '%s': %s", b_file.name, exc)

    # 1. Write Silver Lakehouse Parquet
    silver_parquet_path = dst_dir / "claims_silver.parquet"
    if silver_rows:
        df_silver = pd.DataFrame(silver_rows)
        # Enforce exact column types
        df_silver["days_to_notify"] = df_silver["days_to_notify"].astype("int64")
        df_silver["total_loss_amount"] = df_silver["total_loss_amount"].astype("float64")
        df_silver["total_deductible"] = df_silver["total_deductible"].astype("float64")
        df_silver["total_net_claimed"] = df_silver["total_net_claimed"].astype("float64")
        df_silver["quality_score"] = df_silver["quality_score"].astype("float64")
        df_silver["num_line_items"] = df_silver["num_line_items"].astype("int64")

        table_silver = pa.Table.from_pandas(df_silver, preserve_index=False)
        pq.write_table(
            table_silver,
            silver_parquet_path,
            compression=settings.lakehouse.compression_codec,
        )
        logger.info("Saved %d promoted records to Silver Parquet: %s", len(silver_rows), silver_parquet_path)
    else:
        logger.warning("No records passed quality validation for Silver promotion.")

    # 2. Write Quarantine Records
    quarantine_json_path = dst_dir / "claims_quarantine.json"
    with open(quarantine_json_path, "w", encoding="utf-8") as qf:
        json.dump(quarantine_rows, qf, indent=2)

    # Also save quarantine Parquet for analytical inspection
    quarantine_parquet_path = dst_dir / "claims_quarantine.parquet"
    if quarantine_rows:
        df_quar = pd.DataFrame(quarantine_rows)
        # Convert lists to strings for Parquet compatibility
        df_quar["failed_rules"] = df_quar["failed_rules"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        df_quar["failure_reasons"] = df_quar["failure_reasons"].apply(
            lambda x: " | ".join(x) if isinstance(x, list) else str(x)
        )
        table_quar = pa.Table.from_pandas(df_quar, preserve_index=False)
        pq.write_table(table_quar, quarantine_parquet_path, compression="snappy")

    stats = {
        "total_processed": len(bronze_files),
        "promoted_to_silver": len(silver_rows),
        "quarantined_anomalies": len(quarantine_rows),
        "silver_parquet_path": str(silver_parquet_path.resolve()),
        "quarantine_json_path": str(quarantine_json_path.resolve()),
    }
    return silver_parquet_path, quarantine_json_path, stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    print("=" * 85)
    print("Silver Lakehouse Promotion & Quality Validation Pipeline")
    print("=" * 85)

    silver_path, quar_path, summary = process_bronze_to_silver()

    print(f"\nProcessing Summary:")
    print(f"Total Bronze Dossiers Analyzed:  {summary['total_processed']}")
    print(f"Promoted to Silver Parquet:      {summary['promoted_to_silver']}")
    print(f"Quarantined Anomalies:           {summary['quarantined_anomalies']}")
    print("-" * 85)
    print(f"Silver Parquet Destination:      {silver_path.name} ({silver_path.stat().st_size} bytes)")
    print(f"Quarantine Dossier Log:          {quar_path.name} ({quar_path.stat().st_size} bytes)")

    # Read and inspect Silver Parquet
    df_silver = pd.read_parquet(silver_path)
    print(f"\nSilver Parquet Table Preview (Rows: {len(df_silver)}, Columns: {len(df_silver.columns)}):")
    cols_preview = ["claim_id", "policy_number", "days_to_notify", "total_net_claimed", "claimant_ppsn_hash", "loss_county", "quality_score"]
    print(df_silver[cols_preview].head(5).to_string())

    # Inspect Quarantine reasons
    with open(quar_path, "r", encoding="utf-8") as qf:
        quarantine_data = json.load(qf)

    print("\nQuarantined Records & Failure Rule Audit:")
    for q in quarantine_data:
        print(f"-> Claim: {q['claim_id']:<18} | Net: EUR {q['total_net_claimed']:>10,.2f} | Failed: {q['failed_rules']}")
    print("=" * 85)
