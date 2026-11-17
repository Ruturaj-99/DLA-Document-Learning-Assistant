"""Verified multi-document benchmark dataset for Stage 5G.

The benchmark answers the same four questions as the synthetic evaluation,
but on five real PDFs instead of one generated handbook:

    1. Does retrieval find the page that actually holds the answer?
    2. Does the score gate accept answerable questions?
    3. Does the pipeline honestly reject questions whose answer is absent?
    4. Do citations still round-trip to the exact page text?

Every answerable question carries:
    gold_pages     - 1-based page number(s) copied from the real PDF.
    gold_snippets  - short text copied VERBATIM from that page's extracted
                     text (whitespace is normalized by the metrics).
    expected_answer- a human-verified answer for reporting / future live
                     model grading.

tests/test_multidoc_questions.py re-extracts each PDF and fails if a snippet
no longer appears on its gold page, so the dataset cannot silently drift.

The PDFs are user documents and are NOT committed. Set DLA_BENCHMARK_DIR to
the folder that holds them; the default is the current user's Downloads
folder. Documents whose file is missing are skipped by the runner and their
questions are listed as skipped, never scored as failures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkDocument:
    """One PDF in the benchmark."""

    key: str
    file_name: str
    label: str
    document_id: str


@dataclass(frozen=True)
class MultiDocQuestion:
    """One verified question tied to one benchmark document."""

    id: str
    document_key: str
    question: str
    category: str  # "fact" | "paraphrase" | "toc" | "unanswerable"
    gold_pages: tuple[int, ...] = ()
    gold_snippets: tuple[str, ...] = ()
    expected_answer: str | None = None


DOCUMENTS: dict[str, BenchmarkDocument] = {
    "pdf-testing": BenchmarkDocument(
        key="pdf-testing",
        file_name="PDF TESTING FOR LLM.pdf",
        label="PDF TESTING FOR LLM.pdf",
        document_id="pdf-testing",
    ),
    "anp-gateway": BenchmarkDocument(
        key="anp-gateway",
        file_name="ANP IoT Data Gateway - Datasheet (R0).pdf",
        label="ANP IoT Data Gateway datasheet",
        document_id="anp-gateway",
    ),
    "tsw114": BenchmarkDocument(
        key="tsw114",
        file_name="Datasheet_TSW114_1.10.pdf",
        label="Teltonika TSW114 datasheet",
        document_id="tsw114",
    ),
    "odams": BenchmarkDocument(
        key="odams",
        file_name="ODAMS_User_Manual.pdf",
        label="ODAMS user manual",
        document_id="odams",
    ),
    "sw245": BenchmarkDocument(
        key="sw245",
        file_name="SW-245.pdf",
        label="SW-245 4G LTE router datasheet",
        document_id="sw245",
    ),
}


def _q(
    number: int,
    document_key: str,
    question: str,
    category: str,
    gold: tuple[tuple[int, str], ...] = (),
    answer: str | None = None,
) -> MultiDocQuestion:
    """Small factory: (page, snippet) pairs -> parallel gold tuples."""
    return MultiDocQuestion(
        id=f"{document_key}-{number:02d}",
        document_key=document_key,
        question=question,
        category=category,
        gold_pages=tuple(page for page, _ in gold),
        gold_snippets=tuple(snippet for _, snippet in gold),
        expected_answer=answer,
    )


QUESTIONS: list[MultiDocQuestion] = [
    # --- PDF TESTING FOR LLM.pdf (40-page BHEL local-gauge datasheet pack) --
    _q(
        1,
        "pdf-testing",
        "What is the minimum liquid specific gravity specified for the float and dial level gauge?",
        "fact",
        ((7, "liquid sg"),),
        "0.8 or higher",
    ),
    _q(
        2,
        "pdf-testing",
        "What is the accuracy of the analog indicator on the float and dial gauge?",
        "fact",
        ((7, "+/- 50 mm"),),
        "+/- 50 mm",
    ),
    _q(
        3,
        "pdf-testing",
        "What are the optional transmitter's output signal, supply voltage and maximum load impedance?",
        "fact",
        (
            (7, "supply                        : 24 vdc"),
            (7, "output                        : 4 - 20 ma (2wire) x 400 ohms load"),
        ),
        "4-20 mA (2-wire), 24 VDC, 400 ohms load",
    ),
    _q(
        4,
        "pdf-testing",
        "What is the specified float size of the float and dial gauge?",
        "fact",
        ((7, "73 x 240h cylindrical"),),
        "Ø73 x 240 mm cylindrical",
    ),
    _q(
        5,
        "pdf-testing",
        "What accuracy and process connection are specified for the Bourdon pressure gauge?",
        "fact",
        (
            (27, "1%of span"),
            (27, "1/2 inch npt (m)"),
        ),
        "+/- 1% of span; 1/2 inch NPT (M)",
    ),
    _q(
        6,
        "pdf-testing",
        "What is listed in the table of contents?",
        "toc",
        ((4, "datasheet of float and dial type level gauge"),),
        "Datasheet of float and dial type level gauge",
    ),
    _q(
        7,
        "pdf-testing",
        "What is the price of the float and dial level gauge?",
        "unanswerable",
    ),
    # --- ANP IoT Data Gateway datasheet (2 pages) --------------------------
    _q(
        1,
        "anp-gateway",
        "What power supply does the ANP IoT data gateway require?",
        "fact",
        ((1, "dc 24v @ 1a"),),
        "DC 24 V @ 1 A",
    ),
    _q(
        2,
        "anp-gateway",
        "Which CPU and how much RAM does the ANP gateway have?",
        "fact",
        ((1, "arm cortex-a7 1.2 ghz cpu, 128 mb ram, 512 flash"),),
        "ARM Cortex-A7 1.2 GHz, 128 MB RAM, 512 MB flash",
    ),
    _q(
        3,
        "anp-gateway",
        "Which serial baud rates does the ANP gateway support?",
        "fact",
        ((1, "300 to 115200 bps"),),
        "300 to 115200 bps",
    ),
    _q(
        4,
        "anp-gateway",
        "Which IoT protocol is supported for cloud integration?",
        "fact",
        ((1, "mqtt protocol - mqtt broker, mqtt publisher"),),
        "MQTT (broker and publisher)",
    ),
    _q(
        5,
        "anp-gateway",
        "What are the enclosure dimensions of the ANP gateway?",
        "fact",
        ((2, "size: 80 x 40 x 75 mm"),),
        "80 x 40 x 75 mm (W x H x D)",
    ),
    _q(
        6,
        "anp-gateway",
        "What is the maximum Wi-Fi range of the ANP gateway?",
        "unanswerable",
    ),
    # --- Teltonika TSW114 datasheet (8 pages) ------------------------------
    _q(
        1,
        "tsw114",
        "How many Ethernet ports and at what speeds does the TSW114 operate?",
        "fact",
        ((2, "5 x eth ports, 10/100/1000 mbps"),),
        "5 Ethernet ports at 10/100/1000 Mbps",
    ),
    _q(
        2,
        "tsw114",
        "What is the TSW114 input voltage range?",
        "fact",
        ((2, "9 - 30 vdc"),),
        "9 - 30 VDC",
    ),
    _q(
        3,
        "tsw114",
        "What is the TSW114 power consumption at idle and at maximum?",
        "fact",
        ((2, "idle: 0.3 w / max: 2.3 w"),),
        "0.3 W idle, 2.3 W maximum",
    ),
    _q(
        4,
        "tsw114",
        "What are the TSW114 dimensions?",
        "fact",
        ((2, "113.1 x 27.4 x 80.5 mm"),),
        "113.1 x 27.4 x 80.5 mm",
    ),
    _q(
        5,
        "tsw114",
        "How much does the TSW114 weigh?",
        "fact",
        ((2, "174 g"),),
        "174 g",
    ),
    _q(
        6,
        "tsw114",
        "What jumbo frame size does the TSW114 support?",
        "fact",
        ((2, "9216 bytes"),),
        "9216 bytes",
    ),
    _q(
        7,
        "tsw114",
        "Which standard order code is listed for the TSW114 without a PSU?",
        "fact",
        ((6, "tsw114000000 / standard package"),),
        "TSW114000000",
    ),
    _q(
        8,
        "tsw114",
        "What is the price of the TSW114?",
        "unanswerable",
    ),
    # --- ODAMS user manual (21 pages, includes a TOC) ----------------------
    _q(
        1,
        "odams",
        "How long does CPCB generally take to process an industry registration application?",
        "fact",
        ((3, "01 working day"),),
        "About 1 working day",
    ),
    _q(
        2,
        "odams",
        "How long does CPCB generally take to process station details?",
        "fact",
        ((3, "02 - 03 working days"),),
        "2 - 3 working days",
    ),
    _q(
        3,
        "odams",
        "What should an industry do if it has not received the API configuration email?",
        "fact",
        ((3, "check the junk folder"),),
        "Check the email account's junk folder, then contact the helpdesk",
    ),
    _q(
        4,
        "odams",
        "Which email address handles API or programming queries?",
        "fact",
        ((21, "cems2.cpcb@gov.in"),),
        "cems2.cpcb@gov.in",
    ),
    _q(
        5,
        "odams",
        "Where can an industry generate the Industry Key for API access?",
        "fact",
        ((18, "industry key generation"),),
        "In the Key Generation module, under Industry Key Generation",
    ),
    _q(
        6,
        "odams",
        "Where does the industry download the datalogger activation executable?",
        "fact",
        ((16, "exe generation page"),),
        "From the Exe Generation page in the left-hand menu",
    ),
    _q(
        7,
        "odams",
        "Which sections are listed in the table of contents?",
        "toc",
        ((2, "section b: miscellaneous information"),),
        "Section B: Miscellaneous Information is listed, among others",
    ),
    _q(
        8,
        "odams",
        "What is the toll-free telephone number of the helpdesk?",
        "unanswerable",
    ),
    # --- SW-245 4G LTE router datasheet (3 pages) --------------------------
    _q(
        1,
        "sw245",
        "What is the accepted power input range of the SW-245 router?",
        "fact",
        ((1, "5~36v"),),
        "DC 5 - 36 V",
    ),
    _q(
        2,
        "sw245",
        "What are the SW-245 LTE downlink and uplink rates?",
        "fact",
        ((2, "downstream rate 150 mbps"),),
        "150 Mbps downlink, 50 Mbps uplink",
    ),
    _q(
        3,
        "sw245",
        "Which Wi-Fi standard does the SW-245 support?",
        "fact",
        ((2, "ieee802.11b/g/n"),),
        "IEEE 802.11b/g/n",
    ),
    _q(
        4,
        "sw245",
        "What are the SW-245 dimensions?",
        "fact",
        ((3, "93x89x24mm"),),
        "93 x 89 x 24 mm",
    ),
    _q(
        5,
        "sw245",
        "What is the SW-245 weight?",
        "fact",
        ((3, "250g"),),
        "250 g",
    ),
    _q(
        6,
        "sw245",
        "How many SIM cards does the SW-245 support?",
        "fact",
        ((1, "dual-sim"),),
        "Dual SIM",
    ),
    _q(
        7,
        "sw245",
        "What is the maximum number of VPN tunnels supported by the SW-245?",
        "unanswerable",
    ),
    # --- Paraphrases: same verified facts, deliberately different wording --
    _q(
        8,
        "pdf-testing",
        "How dense must the liquid be for the float and dial gauge to work?",
        "paraphrase",
        ((7, "liquid sg"),),
        "At least 0.8 specific gravity",
    ),
    _q(
        7,
        "anp-gateway",
        "How much memory does the ANP gateway provide?",
        "paraphrase",
        ((1, "128 mb ram"),),
        "128 MB RAM",
    ),
    _q(
        9,
        "tsw114",
        "What voltage range can power the TSW114?",
        "paraphrase",
        ((2, "9 - 30 vdc"),),
        "9 - 30 VDC",
    ),
    _q(
        9,
        "odams",
        "How soon does CPCB usually approve a new industry registration?",
        "paraphrase",
        ((3, "01 working day"),),
        "About 1 working day",
    ),
    _q(
        8,
        "sw245",
        "How fast is the SW-245's LTE downlink?",
        "paraphrase",
        ((2, "downstream rate 150 mbps"),),
        "150 Mbps",
    ),
]


def benchmark_dir() -> Path:
    """Folder containing the benchmark PDFs (override with DLA_BENCHMARK_DIR)."""
    configured = os.environ.get("DLA_BENCHMARK_DIR")
    return Path(configured) if configured else Path.home() / "Downloads"


def document_path(document: BenchmarkDocument) -> Path:
    """Absolute path of one benchmark PDF."""
    return benchmark_dir() / document.file_name


def questions_by_document() -> dict[str, list[MultiDocQuestion]]:
    """Group the dataset by document key, preserving dataset order."""
    grouped: dict[str, list[MultiDocQuestion]] = {key: [] for key in DOCUMENTS}
    for question in QUESTIONS:
        grouped[question.document_key].append(question)
    return grouped


def count_by_category() -> dict[str, int]:
    """How many questions per category (dataset sanity check)."""
    counts: dict[str, int] = {}
    for question in QUESTIONS:
        counts[question.category] = counts.get(question.category, 0) + 1
    return counts
