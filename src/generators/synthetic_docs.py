"""Synthetic financial claim document generator for regulatory and operational ingestion.

Generates realistic Irish insurance claim dossiers (commercial property, cyber interruption,
and operational loss) formatted as enterprise PDF documents with ReportLab. Incorporates
strict schema adherence for clean records and deliberate regulatory/data quality anomalies
(negative financial amounts, missing mandatory policy numbers, delayed notification breaches).
"""

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

from faker import Faker
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Ensure project root is accessible when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings

logger = logging.getLogger("synthetic_docs_generator")


@dataclass(frozen=True)
class ClaimLineItem:
    """Itemized loss category and requested compensation line."""

    category: str
    description: str
    amount: float
    deductible: float
    net_claimed: float


@dataclass
class SyntheticClaimRecord:
    """Representation of an insurance claim dossier before PDF serialization."""

    claim_id: str
    policy_number: Optional[str]
    claimant_name: str
    claimant_email: str
    claimant_phone: str
    claimant_ppsn: str
    claimant_address: str
    eircode: str
    incident_type: str
    incident_date: str
    filing_date: str
    days_to_notify: int
    currency: str
    line_items: List[ClaimLineItem]
    total_loss_amount: float
    total_deductible: float
    total_net_claimed: float
    is_anomaly: bool
    anomaly_types: List[str]
    dora_classification: str
    incident_description: str


class SyntheticClaimGenerator:
    """Generator for creating realistic financial and operational claim documents."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize generator with fixed seed for deterministic reproducibility.

        Args:
            seed: Integer random seed for Faker and random generators.
        """
        self.seed = seed
        self.faker = Faker("en_IE")
        Faker.seed(seed)
        random.seed(seed)

        self.incident_types: List[str] = [
            "Commercial Property Storm & Water Ingress",
            "ICT Operational System Outage & Interruption",
            "Ransomware / Malicious Cyber Intrusion",
            "Commercial Fleet Logistics Impact",
            "Executive & Professional Liability Claim",
            "Supply Chain Fulfillment Breakdown",
        ]

        self.dora_tiers: List[str] = [
            "Major ICT Incident (DORA Article 18)",
            "Significant Operational Disruption",
            "Standard Operational Claim (Non-ICT)",
        ]

        self.loss_categories: List[str] = [
            "Emergency Remediation & Containment",
            "Hardware Replacement & System Recovery",
            "Third-Party Forensic Investigation",
            "Business Interruption & Lost Operational Revenue",
            "Legal & Regulatory Consultation Fees",
        ]

    def _generate_irish_ppsn(self) -> str:
        """Generate valid-formatted Irish Personal Public Service Number (7 digits + 1 or 2 letters)."""
        digits = f"{random.randint(1000000, 9999999)}"
        letters = random.choice(["W", "T", "X", "FA", "WA", "R", "L"])
        return f"{digits}{letters}"

    def _generate_eircode(self) -> str:
        """Generate realistic Irish routing key and unique identifier (Eircode)."""
        routing_keys = ["D02", "D04", "D14", "T12", "H91", "V94", "X91", "A96", "C15"]
        routing = random.choice(routing_keys)
        unique_part = "".join(random.choices("0123456789ABCDEFHKNPRTWXYZ", k=4))
        return f"{routing} {unique_part}"

    def create_claim_record(self, index: int, anomaly_profile: Optional[str] = None) -> SyntheticClaimRecord:
        """Synthesize a complete claim data record with optional deliberate anomalies.

        Args:
            index: Numerical sequence identifier (1-indexed).
            anomaly_profile: Specific anomaly pattern to inject or None for clean record.

        Returns:
            SyntheticClaimRecord: Fully populated claim data model.
        """
        claim_id = f"CLM-IE-2026-{index:04d}"
        policy_number: Optional[str] = f"POL-IE-{random.randint(100000, 999999)}"
        claimant_name = self.faker.name()
        claimant_email = self.faker.company_email()
        claimant_phone = f"+353 {random.choice(['83', '85', '87', '89'])} {random.randint(100, 999)} {random.randint(1000, 9999)}"
        claimant_ppsn = self._generate_irish_ppsn()
        claimant_address = f"{self.faker.street_address()}, {self.faker.city()}, Co. {self.faker.county()}"
        eircode = self._generate_eircode()
        incident_type = random.choice(self.incident_types)
        dora_classification = random.choice(self.dora_tiers)

        # Baseline chronological logic: filing date is in early 2026, incident occurred recently
        base_filing_date = date(2026, 2, 1) + timedelta(days=random.randint(1, 25))
        filing_date_str = base_filing_date.isoformat()

        # Clean baseline notification delta: 3 to 21 days
        days_to_notify = random.randint(3, 21)
        incident_date = base_filing_date - timedelta(days=days_to_notify)
        incident_date_str = incident_date.isoformat()

        currency = "EUR"
        is_anomaly = anomaly_profile is not None
        anomaly_types: List[str] = []

        # Generate realistic multi-item loss table
        num_items = random.randint(2, 4)
        line_items: List[ClaimLineItem] = []
        for i in range(num_items):
            cat = self.loss_categories[i % len(self.loss_categories)]
            desc = f"{cat} incurred during mitigation of {incident_type}."
            raw_amt = round(random.uniform(4000.0, 35000.0), 2)
            deductible = round(raw_amt * 0.10, 2)
            net_amt = round(raw_amt - deductible, 2)
            line_items.append(
                ClaimLineItem(
                    category=cat,
                    description=desc,
                    amount=raw_amt,
                    deductible=deductible,
                    net_claimed=net_amt,
                )
            )

        # Inject requested anomalies
        if anomaly_profile == "NEGATIVE_AMOUNT":
            anomaly_types.append("NEGATIVE_CLAIM_AMOUNT")
            # Invert line items and final sums to negative values
            neg_items: List[ClaimLineItem] = []
            for item in line_items:
                neg_items.append(
                    ClaimLineItem(
                        category=item.category,
                        description=f"[ANOMALOUS CHARGEBACK] {item.description}",
                        amount=round(-abs(item.amount), 2),
                        deductible=round(abs(item.deductible), 2),
                        net_claimed=round(-abs(item.net_claimed), 2),
                    )
                )
            line_items = neg_items

        elif anomaly_profile == "MISSING_POLICY_NUMBER":
            anomaly_types.append("MISSING_MANDATORY_POLICY_NUMBER")
            policy_number = None

        elif anomaly_profile == "DELAYED_FILING":
            anomaly_types.append("REGULATORY_NOTIFICATION_DEADLINE_BREACH")
            # Incident occurred 240 to 360 days before notification (violates 30/90-day DORA & policy SLAs)
            delayed_days = random.randint(240, 360)
            days_to_notify = delayed_days
            incident_date = base_filing_date - timedelta(days=delayed_days)
            incident_date_str = incident_date.isoformat()

        # Compute totals
        total_loss = round(sum(item.amount for item in line_items), 2)
        total_deductible = round(sum(item.deductible for item in line_items), 2)
        total_net = round(sum(item.net_claimed for item in line_items), 2)

        incident_description = (
            f"Official loss notification submitted by {claimant_name}. On {incident_date_str}, "
            f"the insured organization suffered a critical operational event classified as '{incident_type}'. "
            f"Immediate technical escalation and mitigation procedures were initiated. "
            f"DORA regulatory oversight tier: {dora_classification}. Claim dossier compiled for "
            f"adjudication and regulatory statutory audit."
        )

        return SyntheticClaimRecord(
            claim_id=claim_id,
            policy_number=policy_number,
            claimant_name=claimant_name,
            claimant_email=claimant_email,
            claimant_phone=claimant_phone,
            claimant_ppsn=claimant_ppsn,
            claimant_address=claimant_address,
            eircode=eircode,
            incident_type=incident_type,
            incident_date=incident_date_str,
            filing_date=filing_date_str,
            days_to_notify=days_to_notify,
            currency=currency,
            line_items=line_items,
            total_loss_amount=total_loss,
            total_deductible=total_deductible,
            total_net_claimed=total_net,
            is_anomaly=is_anomaly,
            anomaly_types=anomaly_types,
            dora_classification=dora_classification,
            incident_description=incident_description,
        )

    def render_pdf(self, record: SyntheticClaimRecord, output_path: Path) -> Path:
        """Render a synthetic claim record into an enterprise PDF document via ReportLab.

        Args:
            record: Data model containing claim details.
            output_path: File system target path for the generated PDF.

        Returns:
            Path: Verified path to the generated PDF.

        Raises:
            IOError: If PDF rendering or filesystem write encounters failure.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()

        # Custom typography styling
        title_style = ParagraphStyle(
            name="ClaimHeaderTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#0d233a"),
        )
        subtitle_style = ParagraphStyle(
            name="ClaimHeaderSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#4b6074"),
        )
        section_style = ParagraphStyle(
            name="ClaimSectionHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#0d233a"),
        )
        cell_bold = ParagraphStyle(
            name="ClaimCellBold",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#1e293b"),
        )
        cell_regular = ParagraphStyle(
            name="ClaimCellRegular",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#334155"),
        )
        cell_amount = ParagraphStyle(
            name="ClaimCellAmount",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            alignment=2,  # Right-aligned
            textColor=colors.HexColor("#0f172a"),
        )
        cell_anomaly_amount = ParagraphStyle(
            name="ClaimCellAnomalyAmount",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            alignment=2,
            textColor=colors.HexColor("#b91c1c"),  # Crimson warning tone
        )
        desc_style = ParagraphStyle(
            name="ClaimDescription",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#334155"),
        )
        disclaimer_style = ParagraphStyle(
            name="ClaimDisclaimer",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#64748b"),
        )

        elements: List[Any] = []

        # 1. Header Banner
        header_table_data = [
            [
                Paragraph("<b>EMERALD SHIELD ASSURANCE DESIGNATED ACTIVITY COMPANY</b>", title_style),
                Paragraph(f"<b>STATUS:</b> {'FLAGGED ANOMALY' if record.is_anomaly else 'CLEAN INGESTION'}", cell_bold),
            ],
            [
                Paragraph(
                    "Regulated by the Central Bank of Ireland (Ref: CBI-INS-84920) | Registered Office: Grand Canal Dock, Dublin 2, D02 X285",
                    subtitle_style,
                ),
                Paragraph(f"<b>DOSSIER ID:</b> {record.claim_id}", cell_regular),
            ],
        ]
        header_table = Table(header_table_data, colWidths=[380, 143])
        header_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                ]
            )
        )
        elements.append(header_table)
        elements.append(Spacer(1, 4))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0d233a"), spaceAfter=8))

        # 2. Key Metadata & Policyholder Block
        policy_display = record.policy_number if record.policy_number is not None else "[MISSING / NOT PROVIDED]"
        meta_table_data = [
            [
                Paragraph("<b>Claim Reference:</b>", cell_bold),
                Paragraph(record.claim_id, cell_regular),
                Paragraph("<b>Policy Number:</b>", cell_bold),
                Paragraph(
                    f"<font color='{'#b91c1c' if record.policy_number is None else '#0f172a'}'>{policy_display}</font>",
                    cell_bold if record.policy_number is None else cell_regular,
                ),
            ],
            [
                Paragraph("<b>Filing Date:</b>", cell_bold),
                Paragraph(record.filing_date, cell_regular),
                Paragraph("<b>Incident Date:</b>", cell_bold),
                Paragraph(record.incident_date, cell_regular),
            ],
            [
                Paragraph("<b>Notice Latency:</b>", cell_bold),
                Paragraph(
                    f"{record.days_to_notify} days "
                    + (f"(<font color='#b91c1c'>BREACH &gt; 90d</font>)" if record.days_to_notify > 90 else "(Compliant SLA)"),
                    cell_regular,
                ),
                Paragraph("<b>DORA Tier:</b>", cell_bold),
                Paragraph(record.dora_classification, cell_regular),
            ],
            [
                Paragraph("<b>Claimant Name:</b>", cell_bold),
                Paragraph(record.claimant_name, cell_regular),
                Paragraph("<b>Claimant PPSN:</b>", cell_bold),
                Paragraph(record.claimant_ppsn, cell_regular),
            ],
            [
                Paragraph("<b>Contact Email:</b>", cell_bold),
                Paragraph(record.claimant_email, cell_regular),
                Paragraph("<b>Contact Phone:</b>", cell_bold),
                Paragraph(record.claimant_phone, cell_regular),
            ],
            [
                Paragraph("<b>Loss Location:</b>", cell_bold),
                Paragraph(f"{record.claimant_address} [{record.eircode}]", cell_regular),
                Paragraph("<b>Incident Type:</b>", cell_bold),
                Paragraph(record.incident_type, cell_regular),
            ],
        ]
        meta_table = Table(meta_table_data, colWidths=[95, 165, 95, 168])
        meta_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        elements.append(meta_table)
        elements.append(Spacer(1, 8))

        # 3. Incident Description Narrative
        elements.append(Paragraph("<b>INCIDENT SYNOPSIS & REGULATORY CLASSIFICATION</b>", section_style))
        elements.append(Spacer(1, 3))
        elements.append(Paragraph(record.incident_description, desc_style))
        elements.append(Spacer(1, 8))

        # 4. Itemized Loss & Financial Ledger Table
        elements.append(Paragraph("<b>ITEMIZED FINANCIAL LOSS ASSESSMENT (EUR)</b>", section_style))
        elements.append(Spacer(1, 3))

        ledger_data = [
            [
                Paragraph("<b>Category</b>", cell_bold),
                Paragraph("<b>Description</b>", cell_bold),
                Paragraph("<b>Gross Loss (€)</b>", cell_amount),
                Paragraph("<b>Deductible (€)</b>", cell_amount),
                Paragraph("<b>Net Claim (€)</b>", cell_amount),
            ]
        ]

        for item in record.line_items:
            amt_style = cell_anomaly_amount if item.amount < 0 else cell_amount
            net_style = cell_anomaly_amount if item.net_claimed < 0 else cell_amount
            ledger_data.append(
                [
                    Paragraph(item.category, cell_regular),
                    Paragraph(item.description, cell_regular),
                    Paragraph(f"{item.amount:,.2f}", amt_style),
                    Paragraph(f"{item.deductible:,.2f}", cell_amount),
                    Paragraph(f"{item.net_claimed:,.2f}", net_style),
                ]
            )

        # Totals Row
        tot_net_style = cell_anomaly_amount if record.total_net_claimed < 0 else cell_amount
        ledger_data.append(
            [
                Paragraph("<b>TOTALS</b>", cell_bold),
                Paragraph("<b>Aggregated Claim Liquidation Value</b>", cell_bold),
                Paragraph(f"<b>{record.total_loss_amount:,.2f}</b>", cell_amount),
                Paragraph(f"<b>{record.total_deductible:,.2f}</b>", cell_amount),
                Paragraph(f"<b>{record.total_net_claimed:,.2f}</b>", tot_net_style),
            ]
        )

        ledger_table = Table(ledger_data, colWidths=[120, 183, 70, 70, 80])
        ledger_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f1f5f9")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        elements.append(ledger_table)
        elements.append(Spacer(1, 8))

        # 5. Statutory Regulatory Attestation & Signatures
        audit_note = (
            "<b>STATUTORY DECLARATION:</b> The undersigned claimant/certified claims adjuster hereby certifies under "
            "penalty of perjury that the loss statements, dates, and amounts described herein reflect genuine financial and "
            "operational exposure. Submitted under European Union Digital Operational Resilience Act (DORA) and Central Bank "
            "of Ireland Insurance Regulations 2026."
        )
        elements.append(Paragraph(audit_note, disclaimer_style))
        elements.append(Spacer(1, 8))

        sig_data = [
            [
                Paragraph("<b>Claimant / Authorized Representative:</b>", cell_bold),
                Paragraph("<b>Loss Adjuster / Compliance Officer:</b>", cell_bold),
            ],
            [
                Paragraph(f"Signature: <i>{record.claimant_name}</i>", cell_regular),
                Paragraph("Signature: <i>C. McCarthy, Senior Claims Examiner</i>", cell_regular),
            ],
            [
                Paragraph(f"Date Attested: {record.filing_date}", cell_regular),
                Paragraph(f"Audit Verification Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", cell_regular),
            ],
        ]
        sig_table = Table(sig_data, colWidths=[260, 263])
        sig_table.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        elements.append(KeepTogether([sig_table]))

        try:
            doc.build(elements)
            logger.debug("Successfully generated claim PDF: %s", output_path)
            return output_path
        except Exception as exc:
            logger.error("Failed building PDF document '%s': %s", output_path, exc)
            raise OSError(f"ReportLab PDF rendering error for {output_path}") from exc


def generate_synthetic_claims(
    total_count: int = 20,
    anomalous_count: int = 6,
    output_dir: Optional[Path] = None,
    seed: int = 42,
) -> Tuple[List[Path], Path]:
    """Orchestrate the generation of clean and anomalous synthetic claim PDFs.

    Creates exactly 20 PDFs (default: 14 clean, 6 anomalous) containing specific
    injected data quality anomalies: negative amounts, missing policy numbers,
    and delayed notification filings. Outputs a companion manifest JSON file.

    Args:
        total_count: Total number of claim dossiers to synthesize (default: 20).
        anomalous_count: Number of dossiers with injected anomalies (default: 6).
        output_dir: Destination folder (defaults to settings.storage.raw_inbound_dir).
        seed: Random seed for deterministic generation.

    Returns:
        Tuple[List[Path], Path]: List of generated PDF paths and path to manifest.json.

    Raises:
        ValueError: If anomalous_count exceeds total_count or parameters are invalid.
    """
    if anomalous_count > total_count:
        raise ValueError(f"Anomalous count ({anomalous_count}) cannot exceed total count ({total_count}).")

    settings = get_settings()
    target_dir: Path = output_dir or settings.storage.raw_inbound_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    generator = SyntheticClaimGenerator(seed=seed)

    # 6 deliberate anomaly slots distributed across the 20 files
    # User specified: negative amounts, missing policy numbers, and delayed filing dates
    anomaly_map: Dict[int, str] = {
        3: "NEGATIVE_AMOUNT",
        7: "MISSING_POLICY_NUMBER",
        10: "DELAYED_FILING",
        13: "NEGATIVE_AMOUNT",
        16: "MISSING_POLICY_NUMBER",
        19: "DELAYED_FILING",
    }

    generated_files: List[Path] = []
    manifest_records: List[Dict[str, Any]] = []

    clean_count = 0
    anomaly_count = 0

    for i in range(1, total_count + 1):
        anomaly_profile = anomaly_map.get(i)
        claim_record = generator.create_claim_record(index=i, anomaly_profile=anomaly_profile)

        file_label = "anom" if claim_record.is_anomaly else "clean"
        file_name = f"claim_{i:02d}_{claim_record.claim_id}_{file_label}.pdf"
        file_path = target_dir / file_name

        generator.render_pdf(claim_record, file_path)
        generated_files.append(file_path)

        manifest_entry = asdict(claim_record)
        manifest_entry["pdf_filename"] = file_name
        manifest_entry["pdf_path"] = str(file_path.resolve())
        manifest_records.append(manifest_entry)

        if claim_record.is_anomaly:
            anomaly_count += 1
        else:
            clean_count += 1

    # Write companion ground-truth manifest for validation and testing
    manifest_path = target_dir / "manifest_ground_truth.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_count": total_count,
                "clean_count": clean_count,
                "anomaly_count": anomaly_count,
                "claims": manifest_records,
            },
            f,
            indent=2,
        )

    logger.info(
        "Successfully synthesized %d claims (%d clean, %d anomalous) at '%s'",
        total_count,
        clean_count,
        anomaly_count,
        target_dir,
    )
    return generated_files, manifest_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    print("=" * 78)
    print("Synthetic Financial Claims Ingestion Generator (ReportLab + Faker en_IE)")
    print("=" * 78)

    files, manifest = generate_synthetic_claims(total_count=20, anomalous_count=6)
    print(f"\nGenerated {len(files)} claim documents in: {manifest.parent}")
    print(f"Ground Truth Manifest: {manifest.name}\n")

    # Read and print summary breakdown
    with open(manifest, "r", encoding="utf-8") as mf:
        data = json.load(mf)

    print(f"{'Idx':<4} | {'Claim ID':<18} | {'Type':<8} | {'Amount (€)':<12} | {'Policy No':<15} | {'Notice (Days)':<14} | {'Anomalies'}")
    print("-" * 105)
    for idx, c in enumerate(data["claims"], start=1):
        status = "ANOMALY" if c["is_anomaly"] else "CLEAN"
        anoms = ", ".join(c["anomaly_types"]) if c["anomaly_types"] else "None"
        policy = c["policy_number"] or "[MISSING]"
        amt_str = f"€{c['total_net_claimed']:,.2f}"
        print(f"{idx:<4} | {c['claim_id']:<18} | {status:<8} | {amt_str:<12} | {policy:<15} | {c['days_to_notify']:<14} | {anoms}")

    print("-" * 105)
    print(f"Summary: Total: {data['total_count']}, Clean: {data['clean_count']}, Injected Anomalies: {data['anomaly_count']}")
