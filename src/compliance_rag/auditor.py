"""Compliance RAG auditor cross-referencing claims against Central Bank of Ireland and DORA guidelines.

Leverages ChromaDB vector embeddings to dynamically retrieve statutory articles and sections,
evaluate claim dossiers against regulatory criteria, and produce defensible, auditable
compliance reports with authoritative legal citations.
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
from src.compliance_rag.vector_store import ComplianceVectorStore

logger = logging.getLogger("compliance_auditor")


@dataclass
class ComplianceFinding:
    """An individual regulatory check evaluation with statutory citations."""

    area: str
    verdict: str  # COMPLIANT, BREACH, WARNING
    regulatory_section: str
    statutory_citation: str
    observation: str
    similarity_score: float


@dataclass
class ComplianceAuditReport:
    """Consolidated compliance audit dossier for a financial or operational claim."""

    claim_id: str
    overall_verdict: str  # COMPLIANT, REGULATORY_BREACH, REQUIRES_REVIEW
    policy_number: Optional[str]
    total_net_claimed: float
    days_to_notify: int
    dora_classification: str
    findings: List[ComplianceFinding]
    regulatory_citations_cited: List[str]
    audit_timestamp: str
    supervisory_summary: str


class ComplianceAuditor:
    """Automated supervisory auditor integrating ChromaDB semantic retrieval with business logic."""

    def __init__(self, vector_store: Optional[ComplianceVectorStore] = None) -> None:
        """Initialize auditor with persistent vector store.

        Args:
            vector_store: Instance of ComplianceVectorStore. If None, instantiates default.
        """
        self.vector_store: ComplianceVectorStore = vector_store or ComplianceVectorStore()
        # Guarantee corpus is indexed
        self.vector_store.index_corpus()
        self.settings = get_settings()

    def audit_record(self, record: Dict[str, Any]) -> ComplianceAuditReport:
        """Conduct an end-to-end statutory compliance audit of a claim record via RAG retrieval.

        Args:
            record: Dictionary representation of claim record (from Silver or Quarantine).

        Returns:
            ComplianceAuditReport: Structured audit report with regulatory citations.
        """
        claim_id: str = record.get("claim_id", "UNKNOWN_CLAIM")
        policy_num: Optional[str] = record.get("policy_number")
        net_claimed: float = float(record.get("total_net_claimed", 0.0))
        days_to_notify: int = int(record.get("days_to_notify", 0))
        dora_tier: str = record.get("dora_classification", "Standard Operational Claim (Non-ICT)")

        findings: List[ComplianceFinding] = []
        citations_cited: List[str] = []

        # -------------------------------------------------------------------------
        # Area 1: Statutory Notification SLA (DORA Article 19 & CBI Guidelines)
        # -------------------------------------------------------------------------
        sla_query = "operational claims submission deadline 90 days statutory SLA notification latency"
        sla_hits = self.vector_store.search_guidelines(sla_query, top_k=1, section_number=3)
        if sla_hits:
            hit = sla_hits[0]
            sec_name = hit["metadata"].get("section", "SECTION 3: STATUTORY NOTIFICATION TIMELINES")
            sim = hit["similarity_score"]
            excerpt = hit["text"][:220].replace("\n", " ").strip()
        else:
            sec_name = "SECTION 3: STATUTORY NOTIFICATION TIMELINES"
            sim = 0.0
            excerpt = "Claims must be submitted within 90 days from incident date."

        if days_to_notify > 90:
            findings.append(
                ComplianceFinding(
                    area="STATUTORY_NOTIFICATION_SLA",
                    verdict="BREACH",
                    regulatory_section=sec_name,
                    statutory_citation=f"{sec_name} (Similarity: {sim:.1%}): {excerpt}",
                    observation=(
                        f"Claim filed {days_to_notify} days post-incident, violating the statutory 90-day "
                        f"notification window mandated by Central Bank of Ireland and DORA Article 19."
                    ),
                    similarity_score=sim,
                )
            )
            citations_cited.append("CBI-DORA Section 3.2 (90-Day SLA Limit)")
        else:
            findings.append(
                ComplianceFinding(
                    area="STATUTORY_NOTIFICATION_SLA",
                    verdict="COMPLIANT",
                    regulatory_section=sec_name,
                    statutory_citation=f"{sec_name}: {excerpt}",
                    observation=f"Claim notification latency of {days_to_notify} days satisfies the statutory SLA.",
                    similarity_score=sim,
                )
            )

        # -------------------------------------------------------------------------
        # Area 2: Mandatory Policy Identification (CBI Consumer Protection Code)
        # -------------------------------------------------------------------------
        policy_query = "mandatory policy identification number contractual validity consumer protection"
        policy_hits = self.vector_store.search_guidelines(policy_query, top_k=1, section_number=4)
        if policy_hits:
            hit = policy_hits[0]
            sec_name = hit["metadata"].get("section", "SECTION 4: MANDATORY POLICY IDENTIFICATION")
            sim = hit["similarity_score"]
            excerpt = hit["text"][:220].replace("\n", " ").strip()
        else:
            sec_name = "SECTION 4: MANDATORY POLICY IDENTIFICATION"
            sim = 0.0
            excerpt = "Every claim must bear an active, verified Policy Number."

        if not policy_num or policy_num.strip() in ("", "None", "null", "[MISSING]"):
            findings.append(
                ComplianceFinding(
                    area="MANDATORY_POLICY_NUMBER",
                    verdict="BREACH",
                    regulatory_section=sec_name,
                    statutory_citation=f"{sec_name} (Similarity: {sim:.1%}): {excerpt}",
                    observation=(
                        "Policy number is absent or placeholder. Financial liquidation cannot proceed "
                        "against unverified contractual relationships under the CBI Consumer Protection Code."
                    ),
                    similarity_score=sim,
                )
            )
            citations_cited.append("CBI-DORA Section 4.1 & 4.2 (Mandatory Policy Identification)")
        else:
            findings.append(
                ComplianceFinding(
                    area="MANDATORY_POLICY_NUMBER",
                    verdict="COMPLIANT",
                    regulatory_section=sec_name,
                    statutory_citation=f"{sec_name}: {excerpt}",
                    observation=f"Valid insurance policy reference verified: '{policy_num}'.",
                    similarity_score=sim,
                )
            )

        # -------------------------------------------------------------------------
        # Area 3: Financial Valuation Integrity & Negative Claims Prohibition
        # -------------------------------------------------------------------------
        val_query = "negative liquidation prohibition positive financial loss valuation integrity"
        val_hits = self.vector_store.search_guidelines(val_query, top_k=1, section_number=5)
        if val_hits:
            hit = val_hits[0]
            sec_name = hit["metadata"].get("section", "SECTION 5: FINANCIAL INTEGRITY AND VALUATION")
            sim = hit["similarity_score"]
            excerpt = hit["text"][:220].replace("\n", " ").strip()
        else:
            sec_name = "SECTION 5: FINANCIAL INTEGRITY AND VALUATION"
            sim = 0.0
            excerpt = "Negative liquidation claims are strictly prohibited and considered irregular."

        if net_claimed < 0:
            findings.append(
                ComplianceFinding(
                    area="FINANCIAL_VALUATION_INTEGRITY",
                    verdict="BREACH",
                    regulatory_section=sec_name,
                    statutory_citation=f"{sec_name} (Similarity: {sim:.1%}): {excerpt}",
                    observation=(
                        f"Net liquidation requested amount of EUR {net_claimed:,.2f} is negative. "
                        f"Classified as critical accounting irregularity or unauthorized chargeback."
                    ),
                    similarity_score=sim,
                )
            )
            citations_cited.append("CBI-DORA Section 5.2 (Negative Liquidation Prohibition)")
        else:
            findings.append(
                ComplianceFinding(
                    area="FINANCIAL_VALUATION_INTEGRITY",
                    verdict="COMPLIANT",
                    regulatory_section=sec_name,
                    statutory_citation=f"{sec_name}: {excerpt}",
                    observation=f"Requested net indemnity of EUR {net_claimed:,.2f} is positive and compliant.",
                    similarity_score=sim,
                )
            )

        # -------------------------------------------------------------------------
        # Area 4: GDPR & Restricted PPSN Pseudonymization Compliance
        # -------------------------------------------------------------------------
        ppsn_hash = record.get("claimant_ppsn_hash")
        gdpr_query = "GDPR Article 4 restricted identifiers Irish PPSN cryptographic pseudonymization"
        gdpr_hits = self.vector_store.search_guidelines(gdpr_query, top_k=1, section_number=6)
        if gdpr_hits:
            hit = gdpr_hits[0]
            sec_name = hit["metadata"].get("section", "SECTION 6: DATA GOVERNANCE AND GDPR")
            sim = hit["similarity_score"]
            excerpt = hit["text"][:220].replace("\n", " ").strip()
        else:
            sec_name = "SECTION 6: DATA GOVERNANCE AND GDPR"
            sim = 0.0
            excerpt = "PPSNs must be pseudonymized using cryptographically salted digests."

        if ppsn_hash and ppsn_hash.startswith("PPSN_"):
            findings.append(
                ComplianceFinding(
                    area="GDPR_DATA_PRIVACY_PPSN",
                    verdict="COMPLIANT",
                    regulatory_section=sec_name,
                    statutory_citation=f"{sec_name}: {excerpt}",
                    observation="Irish PPSN cryptographically pseudonymized under GDPR Article 4(5).",
                    similarity_score=sim,
                )
            )
        else:
            findings.append(
                ComplianceFinding(
                    area="GDPR_DATA_PRIVACY_PPSN",
                    verdict="WARNING",
                    regulatory_section=sec_name,
                    statutory_citation=f"{sec_name}: {excerpt}",
                    observation="Pseudonymized PPSN token could not be verified in the record payload.",
                    similarity_score=sim,
                )
            )

        # Overall verdict synthesis
        breaches = [f for f in findings if f.verdict == "BREACH"]
        warnings = [f for f in findings if f.verdict == "WARNING"]

        if breaches:
            overall_verdict = "REGULATORY_BREACH"
            summary_desc = (
                f"AUDIT FAILED: Claim {claim_id} contains {len(breaches)} statutory breach(es): "
                f"{', '.join(b.area for b in breaches)}. Quarantined pending supervisory certification."
            )
        elif warnings:
            overall_verdict = "REQUIRES_REVIEW"
            summary_desc = f"AUDIT CONDITIONAL: Claim {claim_id} flagged for review ({len(warnings)} warning)."
        else:
            overall_verdict = "COMPLIANT"
            summary_desc = (
                f"AUDIT PASSED: Claim {claim_id} fully satisfies Central Bank of Ireland and DORA "
                f"statutory requirements across filing latency, policy validity, valuation, and privacy."
            )

        return ComplianceAuditReport(
            claim_id=claim_id,
            overall_verdict=overall_verdict,
            policy_number=policy_num,
            total_net_claimed=net_claimed,
            days_to_notify=days_to_notify,
            dora_classification=dora_tier,
            findings=findings,
            regulatory_citations_cited=citations_cited,
            audit_timestamp=datetime.now(timezone.utc).isoformat(),
            supervisory_summary=summary_desc,
        )

    def audit_lakehouse_pipeline(
        self,
        silver_parquet_path: Optional[Path] = None,
        quarantine_json_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Audit all records from Silver Parquet and Quarantine, producing a Gold Lakehouse audit report.

        Args:
            silver_parquet_path: Path to promoted Silver Parquet table.
            quarantine_json_path: Path to quarantined JSON records.
            output_dir: Target directory for Gold Lakehouse audit output.

        Returns:
            Tuple[Path, Dict[str, Any]]: Output JSON dossier path and summary statistics.
        """
        app_settings = get_settings()
        silver_path = silver_parquet_path or (app_settings.storage.silver_lakehouse_dir / "claims_silver.parquet")
        quar_path = quarantine_json_path or (app_settings.storage.silver_lakehouse_dir / "claims_quarantine.json")
        gold_dir = output_dir or app_settings.storage.gold_lakehouse_dir
        gold_dir.mkdir(parents=True, exist_ok=True)

        records_to_audit: List[Dict[str, Any]] = []

        # Load Silver Records
        if silver_path.exists():
            df_silver = pd.read_parquet(silver_path)
            records_to_audit.extend(df_silver.to_dict(orient="records"))

        # Load Quarantine Records
        if quar_path.exists():
            with open(quar_path, "r", encoding="utf-8") as qf:
                quar_records = json.load(qf)
            records_to_audit.extend(quar_records)

        reports: List[ComplianceAuditReport] = []
        stats = {
            "total_audited": len(records_to_audit),
            "compliant": 0,
            "regulatory_breaches": 0,
            "requires_review": 0,
            "citations_frequency": {},
        }

        for rec in records_to_audit:
            report = self.audit_record(rec)
            reports.append(report)

            if report.overall_verdict == "COMPLIANT":
                stats["compliant"] += 1
            elif report.overall_verdict == "REGULATORY_BREACH":
                stats["regulatory_breaches"] += 1
            else:
                stats["requires_review"] += 1

            for cit in report.regulatory_citations_cited:
                stats["citations_frequency"][cit] = stats["citations_frequency"].get(cit, 0) + 1

        # Write Gold Lakehouse compliance audit JSON
        gold_json_path = gold_dir / "compliance_audit_reports.json"
        with open(gold_json_path, "w", encoding="utf-8") as gf:
            json.dump([asdict(r) for r in reports], gf, indent=2)

        # Write Flat Parquet Summary Table for BI/Analytics
        summary_rows: List[Dict[str, Any]] = []
        for r in reports:
            summary_rows.append(
                {
                    "claim_id": r.claim_id,
                    "overall_verdict": r.overall_verdict,
                    "policy_number": r.policy_number,
                    "total_net_claimed": r.total_net_claimed,
                    "days_to_notify": r.days_to_notify,
                    "dora_classification": r.dora_classification,
                    "num_breaches": sum(1 for f in r.findings if f.verdict == "BREACH"),
                    "citations_count": len(r.regulatory_citations_cited),
                    "cited_articles": ", ".join(r.regulatory_citations_cited),
                    "audit_timestamp": r.audit_timestamp,
                    "supervisory_summary": r.supervisory_summary,
                }
            )

        gold_parquet_path = gold_dir / "compliance_audit_ledger.parquet"
        if summary_rows:
            df_gold = pd.DataFrame(summary_rows)
            df_gold["total_net_claimed"] = df_gold["total_net_claimed"].astype("float64")
            df_gold["days_to_notify"] = df_gold["days_to_notify"].astype("int64")
            df_gold["num_breaches"] = df_gold["num_breaches"].astype("int64")
            df_gold["citations_count"] = df_gold["citations_count"].astype("int64")

            table_gold = pa.Table.from_pandas(df_gold, preserve_index=False)
            pq.write_table(table_gold, gold_parquet_path, compression="snappy")
            logger.info("Saved Gold compliance audit ledger to: %s", gold_parquet_path)

        logger.info(
            "Completed compliance audit across %d dossiers: %d Compliant, %d Breaches",
            len(reports),
            stats["compliant"],
            stats["regulatory_breaches"],
        )
        return gold_json_path, stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    print("=" * 85)
    print("Compliance RAG Auditor Pipeline (ChromaDB + CBI / DORA Rules)")
    print("=" * 85)

    auditor = ComplianceAuditor()
    report_file, audit_stats = auditor.audit_lakehouse_pipeline()

    print(f"\nAudit Execution Finished:")
    print(f"Total Dossiers Inspected:      {audit_stats['total_audited']}")
    print(f"Statutorily Compliant:        {audit_stats['compliant']}")
    print(f"Regulatory Breaches Detected: {audit_stats['regulatory_breaches']}")
    print(f"Gold Audit Report Path:        {report_file.name}")
    print("-" * 85)

    print("\nRegulatory Citations Triggered:")
    for cit, cnt in audit_stats["citations_frequency"].items():
        print(f"  • {cit:<60}: {cnt} breach(es)")

    print("\nSample Audit Findings Preview:")
    with open(report_file, "r", encoding="utf-8") as rf:
        data = json.load(rf)

    for item in data[:4]:
        print(f"\n[Claim: {item['claim_id']}] -> Verdict: {item['overall_verdict']}")
        print(f"  Summary: {item['supervisory_summary']}")
        for f in item["findings"]:
            if f["verdict"] == "BREACH":
                print(f"  [BREACH] {f['area']}: {f['observation']}")
                print(f"           Citation: {f['statutory_citation'][:140]}...")
    print("=" * 85)
