"""PII redaction and pseudonymization engine for financial and operational claims.

Enforces compliance with European GDPR Article 4(5) (Pseudonymisation) and the Irish
Data Protection Act 2018 by cryptographically salting and hashing restricted Irish
Personal Public Service Numbers (PPSNs), masking sensitive personal identifiers (names,
emails, phone numbers, exact addresses), and sanitizing narrative incident descriptions.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.extraction.schemas import BronzeLakehouseEnvelope, ExtractedLineItem, RawClaimExtract

logger = logging.getLogger("pii_redactor")

# Irish PPSN format: 7 digits followed by 1 or 2 alphabetic characters (e.g. 1234567T, 7654321WA)
PPSN_REGEX = re.compile(r"\b(\d{7}[A-Za-z]{1,2})\b")
EMAIL_REGEX = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b")
PHONE_REGEX = re.compile(r"(\+?353[\s\-]?)?(0?[1-9]\d{1,2}[\s\-]?)?(\d{3}[\s\-]?\d{4})")


@dataclass
class GovernedClaimRecord:
    """Cleansed and pseudonymized claim record prepared for quality validation and Silver Lakehouse."""

    claim_id: str
    policy_number: Optional[str]
    status_banner: Optional[str]
    filing_date: str
    incident_date: str
    days_to_notify: int
    dora_classification: str
    incident_type: str
    claimant_ppsn_hash: Optional[str]
    ppsn_format_valid: bool
    claimant_name_masked: str
    claimant_email_masked: str
    claimant_phone_masked: str
    loss_county: str
    eircode_routing_key: str
    incident_description_clean: str
    currency: str
    line_items: List[Dict[str, Any]]
    total_loss_amount: float
    total_deductible: float
    total_net_claimed: float
    statutory_attestation: Optional[str]
    source_filename: str
    source_file_sha256: str
    ingestion_id: str
    governance_timestamp: str
    redaction_actions: List[str]


class PIIRedactor:
    """Enterprise governance engine implementing deterministic pseudonymization and data masking."""

    def __init__(self, salt: Optional[str] = None, mask_char: Optional[str] = None) -> None:
        """Initialize redactor with cryptographic salt and masking configuration.

        Args:
            salt: Secret salt for HMAC-SHA256 pseudonymization. Defaults to settings.governance.salt.
            mask_char: Single character used for masking PII strings. Defaults to settings.governance.mask_char.
        """
        settings = get_settings()
        self.salt: str = salt or settings.governance.salt
        self.mask_char: str = mask_char or settings.governance.mask_char

    def hash_ppsn(self, ppsn: Optional[str]) -> Tuple[Optional[str], bool]:
        """Perform deterministic HMAC-SHA256 pseudonymization on an Irish PPSN.

        Args:
            ppsn: Raw PPSN string (e.g., '1234567T').

        Returns:
            Tuple[Optional[str], bool]: (Salted SHA256 hex digest prefixed with 'PPSN_', is_valid_format).
        """
        if not ppsn:
            return None, False

        cleaned = ppsn.strip().upper()
        is_valid = bool(PPSN_REGEX.fullmatch(cleaned))

        digest = hmac.new(
            key=self.salt.encode("utf-8"),
            msg=cleaned.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

        return f"PPSN_{digest[:32]}", is_valid

    def mask_name(self, name: Optional[str]) -> str:
        """Mask claimant's name while preserving prefix and initials for record sanity.

        Example: 'Mr. Alister Kinnear' -> 'Mr. A****** K******'

        Args:
            name: Claimant's full legal name.

        Returns:
            str: Masked representation.
        """
        if not name or not name.strip():
            return "[NAME_REDACTED]"

        tokens = name.strip().split()
        masked_tokens: List[str] = []

        for token in tokens:
            if token.lower() in ("mr.", "mrs.", "ms.", "dr.", "prof."):
                masked_tokens.append(token)
            elif len(token) <= 2:
                masked_tokens.append(f"{token[0]}{self.mask_char}")
            else:
                masked_tokens.append(f"{token[0]}{self.mask_char * (len(token) - 1)}")

        return " ".join(masked_tokens)

    def mask_email(self, email: Optional[str]) -> str:
        """Mask user part of email address while preserving domain for institutional classification.

        Example: 'marcodwane@davis.org' -> 'm*******e@davis.org'

        Args:
            email: Raw email address string.

        Returns:
            str: Masked email string.
        """
        if not email or "@" not in email:
            return "[EMAIL_REDACTED]"

        user, domain = email.strip().split("@", 1)
        if len(user) <= 2:
            masked_user = f"{user[0]}{self.mask_char}"
        else:
            masked_user = f"{user[0]}{self.mask_char * (len(user) - 2)}{user[-1]}"

        return f"{masked_user}@{domain}"

    def mask_phone(self, phone: Optional[str]) -> str:
        """Mask telephone number preserving international prefix and final two digits.

        Example: '+353 83 125 5506' -> '+353 83 *** **06'

        Args:
            phone: Raw contact telephone number.

        Returns:
            str: Masked phone number string.
        """
        if not phone or not phone.strip():
            return "[PHONE_REDACTED]"

        cleaned = phone.strip()
        if len(cleaned) <= 6:
            return f"***{self.mask_char * 3}"

        return f"{cleaned[:7]} {self.mask_char * 3} {self.mask_char * 2}{cleaned[-2:]}"

    def extract_geo_jurisdiction(
        self,
        address: Optional[str],
        eircode: Optional[str],
    ) -> Tuple[str, str]:
        """Extract coarse county and Eircode routing key for geographic risk aggregation without PII.

        Args:
            address: Full street address line.
            eircode: Irish postal code (e.g. 'D14 PKX2').

        Returns:
            Tuple[str, str]: (Coarse County name, 3-character Eircode Routing Key).
        """
        county = "Unknown County"
        if address:
            county_match = re.search(r"Co\.\s*([A-Za-z\s]+?)(?:\[|$|\,)", address)
            if county_match:
                county = f"Co. {county_match.group(1).strip()}"

        routing_key = "UNK"
        if eircode and len(eircode.strip()) >= 3:
            routing_key = eircode.strip()[:3].upper()

        return county, routing_key

    def sanitize_narrative(self, narrative: Optional[str]) -> str:
        """Scrub embedded PII (PPSNs, emails, phone numbers) from narrative text fields.

        Args:
            narrative: Unstructured descriptive text.

        Returns:
            str: PII-sanitized narrative text.
        """
        if not narrative:
            return ""

        text = narrative
        # Mask embedded PPSNs
        text = PPSN_REGEX.sub("[PPSN_REDACTED]", text)
        # Mask embedded Emails
        text = EMAIL_REGEX.sub("[EMAIL_REDACTED]", text)
        # Mask embedded Phone numbers
        text = PHONE_REGEX.sub("[PHONE_REDACTED]", text)

        return text

    def govern_envelope(self, envelope: BronzeLakehouseEnvelope) -> GovernedClaimRecord:
        """Transform an immutable Bronze Lakehouse envelope into a governed, pseudonymized claim record.

        Args:
            envelope: Ingested BronzeLakehouseEnvelope instance.

        Returns:
            GovernedClaimRecord: Transformed and PII-scrubbed record.
        """
        payload: RawClaimExtract = envelope.extracted_payload
        actions: List[str] = []

        # 1. PPSN Pseudonymization
        ppsn_hash, ppsn_valid = self.hash_ppsn(payload.claimant_ppsn)
        if ppsn_hash:
            actions.append("PSEUDONYMIZED_IRISH_PPSN")

        # 2. Mask Contact Identifiers
        masked_name = self.mask_name(payload.claimant_name)
        if payload.claimant_name:
            actions.append("MASKED_CLAIMANT_NAME")

        masked_email = self.mask_email(payload.claimant_email)
        if payload.claimant_email:
            actions.append("MASKED_CLAIMANT_EMAIL")

        masked_phone = self.mask_phone(payload.claimant_phone)
        if payload.claimant_phone:
            actions.append("MASKED_CLAIMANT_PHONE")

        # 3. Coarse Geo Extraction
        county, routing_key = self.extract_geo_jurisdiction(payload.loss_location, payload.eircode)
        actions.append("GENERALIZED_GEOGRAPHIC_LOCATION")

        # 4. Scrub Narrative
        clean_narrative = self.sanitize_narrative(payload.incident_description)
        if clean_narrative != payload.incident_description:
            actions.append("SCRUBBED_NARRATIVE_PII")

        # Line items serialization
        line_items_dict: List[Dict[str, Any]] = [item.model_dump() for item in payload.line_items]

        return GovernedClaimRecord(
            claim_id=payload.claim_id,
            policy_number=payload.policy_number,
            status_banner=payload.status_banner,
            filing_date=payload.filing_date,
            incident_date=payload.incident_date,
            days_to_notify=payload.days_to_notify or 0,
            dora_classification=payload.dora_classification or "Standard Operational Claim (Non-ICT)",
            incident_type=payload.incident_type or "Operational Impact",
            claimant_ppsn_hash=ppsn_hash,
            ppsn_format_valid=ppsn_valid,
            claimant_name_masked=masked_name,
            claimant_email_masked=masked_email,
            claimant_phone_masked=masked_phone,
            loss_county=county,
            eircode_routing_key=routing_key,
            incident_description_clean=clean_narrative,
            currency=payload.currency,
            line_items=line_items_dict,
            total_loss_amount=payload.total_loss_amount,
            total_deductible=payload.total_deductible,
            total_net_claimed=payload.total_net_claimed,
            statutory_attestation=payload.statutory_attestation,
            source_filename=envelope.source_filename,
            source_file_sha256=envelope.source_file_sha256,
            ingestion_id=envelope.ingestion_id,
            governance_timestamp=datetime.now(timezone.utc).isoformat(),
            redaction_actions=actions,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 80)
    print("PII Redaction & Irish PPSN Pseudonymization Engine Test")
    print("=" * 80)

    redactor = PIIRedactor()

    # Test PPSN Hashing
    raw_ppsn = "5108603T"
    ppsn_hash, is_valid = redactor.hash_ppsn(raw_ppsn)
    print(f"Raw PPSN: {raw_ppsn} -> Hash: {ppsn_hash} (Valid format: {is_valid})")

    # Test deterministic property
    ppsn_hash2, _ = redactor.hash_ppsn(raw_ppsn)
    assert ppsn_hash == ppsn_hash2, "Deterministic hashing verification failed!"
    print("Deterministic pseudonymization verified: Same PPSN produces identical digest.")

    # Test Name and Contact Masking
    sample_name = "Mr. Alister Kinnear"
    sample_email = "marcodwane@davis.org"
    sample_phone = "+353 83 125 5506"
    print(f"Name Masking:  '{sample_name}' -> '{redactor.mask_name(sample_name)}'")
    print(f"Email Masking: '{sample_email}' -> '{redactor.mask_email(sample_email)}'")
    print(f"Phone Masking: '{sample_phone}' -> '{redactor.mask_phone(sample_phone)}'")

    # Test Geo Jurisdiction
    sample_addr = "01 Hession Street, Seanna Ville, Co. Wexford [D14 PKX2]"
    county, routing = redactor.extract_geo_jurisdiction(sample_addr, "D14 PKX2")
    print(f"Geo Coarsening: County='{county}', RoutingKey='{routing}'")
    print("=" * 80)
