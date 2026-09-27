"""Unit tests for PII redaction, cryptographic pseudonymization, and quality validation engine."""

from pathlib import Path
import sys

import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.governance.pii_redactor import PIIRedactor
from src.governance.quality_engine import DataQualityEngine, GovernedClaimRecord


@pytest.fixture
def redactor() -> PIIRedactor:
    """Fixture providing initialized PIIRedactor."""
    return PIIRedactor(salt="test_secret_salt_12345", mask_char="*")


@pytest.fixture
def quality_engine() -> DataQualityEngine:
    """Fixture providing DataQualityEngine."""
    return DataQualityEngine(min_quality_score=0.85, max_notification_days=90)


def test_ppsn_deterministic_pseudonymization(redactor: PIIRedactor) -> None:
    """Verify PPSN hashing is deterministic, salted, and valid format."""
    raw_ppsn = "1234567T"
    hash1, is_valid1 = redactor.hash_ppsn(raw_ppsn)
    hash2, is_valid2 = redactor.hash_ppsn(raw_ppsn)

    assert is_valid1 is True
    assert is_valid2 is True
    assert hash1 == hash2
    assert hash1.startswith("PPSN_")
    assert len(hash1) == 5 + 32  # Prefix + 32 hex chars


def test_ppsn_invalid_format_detection(redactor: PIIRedactor) -> None:
    """Verify invalid PPSNs are flagged."""
    invalid_ppsn = "INVALID_PPSN_123"
    _, is_valid = redactor.hash_ppsn(invalid_ppsn)
    assert is_valid is False


def test_contact_masking(redactor: PIIRedactor) -> None:
    """Verify name, email, and phone masking."""
    name = "Mr. Alister Kinnear"
    masked_name = redactor.mask_name(name)
    assert masked_name.startswith("Mr. A")
    assert "*" in masked_name

    email = "test.claimant@emeraldshield.ie"
    masked_email = redactor.mask_email(email)
    assert "@emeraldshield.ie" in masked_email
    assert "*" in masked_email

    phone = "+353 87 123 4567"
    masked_phone = redactor.mask_phone(phone)
    assert "+353 87" in masked_phone
    assert "*" in masked_phone


def test_clean_record_passes_quality_rules(quality_engine: DataQualityEngine) -> None:
    """Verify fully conformant record passes all 7 quality rules."""
    record = GovernedClaimRecord(
        claim_id="CLM-TEST-0001",
        policy_number="POL-IE-123456",
        status_banner="CLEAN INGESTION",
        filing_date="2026-02-15",
        incident_date="2026-02-01",
        days_to_notify=14,
        dora_classification="Major ICT Incident (DORA Article 18)",
        incident_type="ICT Operational System Outage",
        claimant_ppsn_hash="PPSN_abcd1234efgh5678",
        ppsn_format_valid=True,
        claimant_name_masked="Mr. T*** C******",
        claimant_email_masked="t***t@domain.ie",
        claimant_phone_masked="+353 87 *** **67",
        loss_county="Co. Dublin",
        eircode_routing_key="D02",
        incident_description_clean="Clean incident description.",
        currency="EUR",
        line_items=[
            {"category": "Item 1", "amount": 1000.0, "deductible": 100.0, "net_claimed": 900.0},
            {"category": "Item 2", "amount": 2000.0, "deductible": 200.0, "net_claimed": 1800.0},
        ],
        total_loss_amount=3000.0,
        total_deductible=300.0,
        total_net_claimed=2700.0,
        statutory_attestation="Attestation statement.",
        source_filename="test.pdf",
        source_file_sha256="dummy_sha256",
        ingestion_id="uuid_123",
        governance_timestamp="2026-02-15T12:00:00Z",
        redaction_actions=["PSEUDONYMIZED_IRISH_PPSN"],
    )

    report = quality_engine.evaluate_record(record)
    assert report.is_valid is True
    assert report.quality_score == 1.0
    assert len(report.failed_rules) == 0


def test_negative_amount_quarantine_trigger(quality_engine: DataQualityEngine) -> None:
    """Verify negative amount triggers RULE_POSITIVE_AMOUNTS failure."""
    record = GovernedClaimRecord(
        claim_id="CLM-TEST-NEG",
        policy_number="POL-IE-123456",
        status_banner="FLAGGED ANOMALY",
        filing_date="2026-02-15",
        incident_date="2026-02-01",
        days_to_notify=14,
        dora_classification="Standard Operational Claim",
        incident_type="Property Loss",
        claimant_ppsn_hash="PPSN_abcd1234efgh5678",
        ppsn_format_valid=True,
        claimant_name_masked="Mr. T***",
        claimant_email_masked="t***t@domain.ie",
        claimant_phone_masked="+353 87 *** **67",
        loss_county="Co. Cork",
        eircode_routing_key="T12",
        incident_description_clean="Negative amount test.",
        currency="EUR",
        line_items=[{"category": "Chargeback", "amount": -5000.0, "deductible": 0.0, "net_claimed": -5000.0}],
        total_loss_amount=-5000.0,
        total_deductible=0.0,
        total_net_claimed=-5000.0,
        statutory_attestation="Attestation.",
        source_filename="neg.pdf",
        source_file_sha256="dummy_sha256",
        ingestion_id="uuid_123",
        governance_timestamp="2026-02-15T12:00:00Z",
        redaction_actions=[],
    )

    report = quality_engine.evaluate_record(record)
    assert report.is_valid is False
    assert "RULE_POSITIVE_AMOUNTS" in report.failed_rules


def test_missing_policy_quarantine_trigger(quality_engine: DataQualityEngine) -> None:
    """Verify missing policy triggers RULE_POLICY_NUMBER_PRESENT failure."""
    record = GovernedClaimRecord(
        claim_id="CLM-TEST-NOPOL",
        policy_number=None,
        status_banner="FLAGGED ANOMALY",
        filing_date="2026-02-15",
        incident_date="2026-02-01",
        days_to_notify=14,
        dora_classification="Standard Operational Claim",
        incident_type="Property Loss",
        claimant_ppsn_hash="PPSN_abcd1234efgh5678",
        ppsn_format_valid=True,
        claimant_name_masked="Mr. T***",
        claimant_email_masked="t***t@domain.ie",
        claimant_phone_masked="+353 87 *** **67",
        loss_county="Co. Galway",
        eircode_routing_key="H91",
        incident_description_clean="Missing policy test.",
        currency="EUR",
        line_items=[{"category": "Loss", "amount": 1000.0, "deductible": 100.0, "net_claimed": 900.0}],
        total_loss_amount=1000.0,
        total_deductible=100.0,
        total_net_claimed=900.0,
        statutory_attestation="Attestation.",
        source_filename="nopol.pdf",
        source_file_sha256="dummy_sha256",
        ingestion_id="uuid_123",
        governance_timestamp="2026-02-15T12:00:00Z",
        redaction_actions=[],
    )

    report = quality_engine.evaluate_record(record)
    assert report.is_valid is False
    assert "RULE_POLICY_NUMBER_PRESENT" in report.failed_rules


def test_sla_latency_breach_trigger(quality_engine: DataQualityEngine) -> None:
    """Verify notification delay > 90 days triggers RULE_TIMELY_NOTIFICATION failure."""
    record = GovernedClaimRecord(
        claim_id="CLM-TEST-DELAY",
        policy_number="POL-IE-999999",
        status_banner="FLAGGED ANOMALY",
        filing_date="2026-02-15",
        incident_date="2025-04-01",
        days_to_notify=320,  # Violates 90 days SLA
        dora_classification="Standard Operational Claim",
        incident_type="Property Loss",
        claimant_ppsn_hash="PPSN_abcd1234efgh5678",
        ppsn_format_valid=True,
        claimant_name_masked="Mr. T***",
        claimant_email_masked="t***t@domain.ie",
        claimant_phone_masked="+353 87 *** **67",
        loss_county="Co. Waterford",
        eircode_routing_key="X91",
        incident_description_clean="Delayed filing test.",
        currency="EUR",
        line_items=[{"category": "Loss", "amount": 1000.0, "deductible": 100.0, "net_claimed": 900.0}],
        total_loss_amount=1000.0,
        total_deductible=100.0,
        total_net_claimed=900.0,
        statutory_attestation="Attestation.",
        source_filename="delayed.pdf",
        source_file_sha256="dummy_sha256",
        ingestion_id="uuid_123",
        governance_timestamp="2026-02-15T12:00:00Z",
        redaction_actions=[],
    )

    report = quality_engine.evaluate_record(record)
    assert report.is_valid is False
    assert "RULE_TIMELY_NOTIFICATION" in report.failed_rules
