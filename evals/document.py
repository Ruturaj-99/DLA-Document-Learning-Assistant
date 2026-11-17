"""The synthetic evaluation document for the RAG evaluation framework.

A 10-page "AX-40 field service handbook", built entirely in code the same
way tests/test_pdf_loader_ocr.py builds its fake scan pages. Every page
covers one section and is packed with concrete, uniquely identifiable
facts (part numbers, temperatures, durations, torques) so that questions
have exactly one verifiable gold passage.

The text below is the single source of truth: questions.py quotes
snippets from it verbatim, so treat it as frozen once the dataset is
written (edit questions, not this text).
"""

import pymupdf

# 1-based page number -> full page text.
PAGE_TEXTS: dict[int, str] = {
    1: (
        "The AX-40 field service handbook covers inspection, calibration and "
        "repair of the AX-40 series circulation pumps. Read this handbook "
        "completely before touching any part of the machine. Always disconnect "
        "mains power and wait five minutes before opening the electrical "
        "cabinet. The pump housing can reach eighty degrees Celsius during "
        "normal operation, so wear heat resistant gloves at all times. Never "
        "operate the pump with the front cover removed. Report any damaged "
        "power cable to the site supervisor immediately; a damaged cable must "
        "be replaced, never repaired with tape. Emergency stop buttons are "
        "located on the control panel and on the rear housing. Pressing an "
        "emergency stop cuts power to the motor within half a second."
    ),
    2: (
        "The auxiliary heater block must be descaled every three months using "
        "HL-9 descaling fluid. Fill the reservoir with five hundred "
        "millilitres of HL-9, run the heater at sixty degrees Celsius for "
        "forty minutes, then drain and rinse twice with demineralised water. "
        "The heater element is part number HX-2207 and is rated at one "
        "thousand two hundred watts. A white crust on the element means hard "
        "water is being used; switch the site to the soft water inlet before "
        "replacing the element. After any heater service, run the leak test: "
        "pressurise the loop to three bar and watch the gauge for ten "
        "minutes. A pressure drop greater than zero point two bar means a "
        "joint is loose. The heater insulation blanket should be replaced if "
        "it shows cracks wider than two millimetres."
    ),
    3: (
        "Calibrate the flow sensor after every pump impeller change. Start "
        "the pump at setting four and compare the panel readout with the "
        "reference flow meter. If the readout differs from the reference by "
        "more than four percent, enter calibration mode with the span key "
        "and adjust until the difference is under two percent. The impeller "
        "is part number IM-1183 and must be torqued to twelve newton metres. "
        "Always replace the impeller O-ring, part number OR-8841, when the "
        "impeller is removed. Never reuse a deformed O-ring. After "
        "calibration, log the reference value, the panel value and the date "
        "in the calibration register. Calibration entries older than twelve "
        "months invalidate the service certificate."
    ),
    4: (
        "Replace the inlet filter cartridge every six months, or sooner if "
        "the differential pressure alarm sounds. The cartridge is part "
        "number FL-5560 and has a nominal rating of twenty microns. Before "
        "removing the cartridge, close the inlet valve and relieve pressure "
        "with the bleed screw. A cartridge that is dark brown or swollen "
        "must be replaced regardless of the schedule. Never wash and reuse "
        "a disposable cartridge; washing collapses the fibres and lets "
        "particles through. After fitting a new cartridge, run the pump for "
        "five minutes and check the bleed screw for drips. Record the "
        "cartridge lot number in the filter register with the change date."
    ),
    5: (
        "The AX-40 ships with two temperature probes, type K thermocouples "
        "rated from minus forty to plus three hundred degrees Celsius. Probe "
        "accuracy is plus or minus one point five degrees across the working "
        "range. The pressure transducer, part number PT-7712, measures from "
        "zero to ten bar with a one percent full scale error. Both probes "
        "plug into the sensor board through six pin connectors; never force "
        "a connector that resists seating. Sensor cables must be routed at "
        "least ten centimetres away from motor power cables to avoid "
        "electrical noise. A noisy sensor shows as a readout that jumps by "
        "more than three units per second. Replace any probe cable with "
        "damaged insulation; cable part number is CB-3310."
    ),
    6: (
        "Error code E-01 means the flow sensor returned no signal; check the "
        "sensor connector first. E-03 indicates overheating: the heater "
        "exceeded ninety five degrees and the safety thermostat tripped. "
        "Let the heater cool below sixty degrees before resetting the "
        "thermostat with the blue button. E-07 signals a low pressure "
        "condition, usually caused by a leaking joint or a clogged inlet "
        "filter. E-12 is a communication fault between the control panel "
        "and the motor drive; inspect the ribbon cable on the drive board. "
        "E-21 marks a calibration memory checksum error; recalibrate the "
        "flow sensor and, if the error returns, replace the memory board, "
        "part number MB-4402. Any error code that appears three times in "
        "one shift must be written in the fault log."
    ),
    7: (
        "Store spare impellers and O-rings in their sealed bags at room "
        "temperature, away from direct sunlight; shelf life is five years "
        "from the bag date. The AX-40 itself must be drained before "
        "transport in temperatures below zero degrees Celsius, because "
        "trapped water cracks the housing. Use only the original transport "
        "brackets, part number TB-9901, when moving the unit with a "
        "forklift. The control panel must be covered with the rain shroud "
        "whenever the unit is stored outdoors, even under a roof. Batteries "
        "inside the panel clock last about two years; the clock battery is "
        "part number BA-1130. When returning a unit to service after "
        "storage, run the leak test from the heater chapter before "
        "reconnecting power."
    ),
    8: (
        "Every service visit needs a completed calibration register entry, "
        "even when nothing was adjusted. An entry contains the unit serial "
        "number, the date, the technician name, the reference meter value "
        "and the panel value. Entries must be written in ink; pencil "
        "entries are not accepted by the auditor. Corrections are made with "
        "a single line through the wrong value plus initials, never with "
        "correction fluid. The register is kept in the blue folder inside "
        "the control panel door. Photographs of the register count as "
        "backups but never replace the paper original. Keep completed "
        "registers for at least seven years. The site manager signs each "
        "register page at the end of the month."
    ),
    9: (
        "Order consumables with the exact part number; similar looking "
        "parts are not interchangeable. HL-9 descaling fluid comes in one "
        "litre bottles, article AR-5010. Demineralised water for rinsing is "
        "article AR-5020 in five litre canisters. The grease for impeller "
        "shafts is silicon free grease, article AR-5035; never use lithium "
        "grease because it attacks the O-ring. Spare fuses for the drive "
        "board are T two amp ceramic, article AR-5045, and only ceramic "
        "fuses may be fitted. Replacement panel labels are article AR-5050 "
        "and must match the language code printed on the panel rear. "
        "Deliveries take five working days; emergency parts can be "
        "couriered in twenty four hours at extra cost."
    ),
    10: (
        "The AX-40 carries a twenty four month warranty from the "
        "commissioning date. The warranty covers parts and labour but "
        "excludes damage from descaling neglect or unauthorised opening of "
        "the drive housing. Keep the commissioning certificate; without it "
        "warranty claims are refused. Warranty service must be performed by "
        "a technician with a valid AX-40 certificate. For warranty cases "
        "contact the service desk on weekdays between eight and sixteen, "
        "reference the serial number and the fault log. Out of warranty "
        "repairs are quoted in advance and need a written approval before "
        "work starts. The regional spare parts warehouse is in Rotterdam; "
        "orders placed before fourteen hours ship the same day."
    ),
}

_DOCUMENT_ID = "eval-handbook"
_FONT_SIZE = 11.0
_MARGIN = 60.0


def document_id() -> str:
    """Stable document id used for every chunk built from this document."""
    return _DOCUMENT_ID


def build_document() -> bytes:
    """Render PAGE_TEXTS as a real 10-page PDF (native text, no images).

    Returns PDF bytes ready for src.pdf_loader.extract_pdf_pages. The
    result is deterministic: same text, same layout, same bytes.
    """
    document = pymupdf.open()
    try:
        for page_number in sorted(PAGE_TEXTS):
            page = document.new_page(width=595, height=842)  # A4 points
            text_rect = pymupdf.Rect(
                _MARGIN, _MARGIN, 595 - _MARGIN, 842 - _MARGIN
            )
            leftover = page.insert_textbox(
                text_rect,
                PAGE_TEXTS[page_number],
                fontname="helv",
                fontsize=_FONT_SIZE,
                align=pymupdf.TEXT_ALIGN_LEFT,
            )
            if leftover < 0:
                raise RuntimeError(
                    f"Evaluation page {page_number} text does not fit "
                    f"(leftover {leftover})."
                )
        return document.tobytes()
    finally:
        document.close()
