"""The 100-question evaluation dataset.

Every question quotes its evidence VERBATIM from evals/document.py:

    gold_pages     - 1-based page(s) where the answer lives.
    gold_snippets  - one exact snippet per gold page; a retrieved chunk
                     counts as a "hit" when it comes from a gold page AND
                     contains that page's snippet. Snippets are kept
                     shorter than the 150-character chunk overlap, so a
                     snippet always survives inside at least one chunk
                     even when it straddles a chunk boundary.
    expected_answer- what a perfect LLM would reply (used by the scripted
                     oracle in run_evaluation.py; never graded as text).

Categories:
    fact          - single fact, question wording close to the document.
    paraphrase    - same fact, question deliberately reworded.
    cross         - needs evidence from TWO passages (both must surface).
    unanswerable  - the fact is NOT in the document; tests the score gate
                    and the honest "not found" behaviour.

Treat the dataset as frozen: editing the document text invalidates the
verbatim snippets (evals/document.py).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuestion:
    """One evaluation question with its gold annotations."""

    id: str
    question: str
    category: str  # "fact" | "paraphrase" | "cross" | "unanswerable"
    gold_pages: tuple[int, ...]  # 1-based page numbers
    gold_snippets: tuple[str, ...]  # one verbatim snippet per gold page
    expected_answer: str | None = None  # oracle answer (retrieval is graded)


def _q(
    number: int,
    question: str,
    category: str,
    gold: tuple[tuple[int, str], ...],
    answer: str | None = None,
) -> EvalQuestion:
    """Small factory: (page, snippet) pairs -> parallel gold tuples."""
    return EvalQuestion(
        id=f"q{number:03d}",
        question=question,
        category=category,
        gold_pages=tuple(page for page, _ in gold),
        gold_snippets=tuple(snippet for _, snippet in gold),
        expected_answer=answer,
    )


QUESTIONS: list[EvalQuestion] = [
    # --- Page 1: safety ----------------------------------------------------
    _q(1, "At what temperature can the pump housing get during normal operation?", "fact", ((1, "reach eighty degrees Celsius"),), "eighty degrees Celsius"),
    _q(2, "How long should you wait after disconnecting mains power before opening the electrical cabinet?", "fact", ((1, "wait five minutes before opening"),), "five minutes"),
    _q(3, "How quickly does the emergency stop cut power to the motor?", "fact", ((1, "within half a second"),), "within half a second"),
    _q(4, "What must happen to a damaged power cable?", "fact", ((1, "damaged cable must be replaced, never repaired with tape"),), "it must be replaced, never repaired with tape"),
    # --- Page 2: heater ----------------------------------------------------
    _q(5, "How often must the auxiliary heater block be descaled?", "fact", ((2, "descaled every three months"),), "every three months"),
    _q(6, "How much HL-9 descaling fluid goes into the reservoir?", "fact", ((2, "five hundred millilitres of HL-9"),), "five hundred millilitres"),
    _q(7, "At what temperature and for how long is the heater run during descaling?", "fact", ((2, "sixty degrees Celsius for forty minutes"),), "sixty degrees Celsius for forty minutes"),
    _q(8, "What is the part number of the heater element?", "fact", ((2, "part number HX-2207"),), "HX-2207"),
    _q(9, "What does a white crust on the heater element mean?", "fact", ((2, "white crust on the element means hard water"),), "hard water is being used; switch to the soft water inlet"),
    _q(10, "How long is the gauge watched during the leak test?", "fact", ((2, "watch the gauge for ten minutes"),), "ten minutes"),
    _q(11, "What pressure drop during the leak test means a loose joint?", "fact", ((2, "greater than zero point two bar"),), "more than 0.2 bar"),
    # --- Page 3: pump calibration ------------------------------------------
    _q(12, "When must the flow sensor be calibrated?", "fact", ((3, "after every pump impeller change"),), "after every pump impeller change"),
    _q(13, "What readout deviation triggers calibration mode?", "fact", ((3, "more than four percent"),), "more than four percent versus the reference meter"),
    _q(14, "How tightly must the impeller be torqued?", "fact", ((3, "torqued to twelve newton metres"),), "twelve newton metres"),
    _q(15, "Which part number is the impeller O-ring?", "fact", ((3, "part number OR-8841"),), "OR-8841"),
    _q(16, "When do calibration entries invalidate the service certificate?", "fact", ((3, "older than twelve months"),), "entries older than twelve months"),
    # --- Page 4: filter schedule -------------------------------------------
    _q(17, "How often is the inlet filter cartridge replaced?", "fact", ((4, "cartridge every six months"),), "every six months, or sooner on the differential pressure alarm"),
    _q(18, "What is the part number of the filter cartridge?", "fact", ((4, "part number FL-5560"),), "FL-5560 (twenty microns)"),
    _q(19, "What must be done before removing the filter cartridge?", "fact", ((4, "close the inlet valve and relieve pressure with the bleed screw"),), "close the inlet valve and relieve pressure with the bleed screw"),
    _q(20, "What cartridge appearance forces replacement regardless of the schedule?", "fact", ((4, "dark brown or swollen"),), "dark brown or swollen"),
    _q(21, "Why must a disposable cartridge never be washed and reused?", "fact", ((4, "washing collapses the fibres"),), "washing collapses the fibres and lets particles through"),
    # --- Page 5: sensor specifications --------------------------------------
    _q(22, "What type of temperature probes does the AX-40 ship with?", "fact", ((5, "type K thermocouples"),), "type K thermocouples"),
    _q(23, "What temperature range are the temperature probes rated for?", "fact", ((5, "minus forty to plus three hundred degrees Celsius"),), "-40 to +300 degrees Celsius"),
    _q(24, "How accurate are the temperature probes?", "fact", ((5, "plus or minus one point five degrees"),), "plus or minus 1.5 degrees"),
    _q(25, "Which part number is the pressure transducer?", "fact", ((5, "part number PT-7712"),), "PT-7712"),
    _q(26, "How far should sensor cables be routed from motor power cables?", "fact", ((5, "at least ten centimetres"),), "at least ten centimetres"),
    _q(27, "What does a noisy sensor readout look like?", "fact", ((5, "jumps by more than three units per second"),), "the readout jumps by more than three units per second"),
    _q(28, "Which part number is the probe cable?", "fact", ((5, "cable part number is CB-3310"),), "CB-3310"),
    # --- Page 6: error codes -------------------------------------------------
    _q(29, "What does error code E-01 mean?", "fact", ((6, "E-01 means the flow sensor returned no signal"),), "flow sensor returned no signal; check the connector"),
    _q(30, "What triggers error code E-03?", "fact", ((6, "the heater exceeded ninety five degrees"),), "overheating: the heater exceeded 95 degrees"),
    _q(31, "Which button resets the safety thermostat?", "fact", ((6, "resetting the thermostat with the blue button"),), "the blue button"),
    _q(32, "What usually causes error code E-07?", "fact", ((6, "a leaking joint or a clogged inlet filter"),), "a leaking joint or a clogged inlet filter"),
    _q(33, "What does error code E-12 indicate?", "fact", ((6, "communication fault between the control panel and the motor drive"),), "communication fault between panel and motor drive"),
    _q(34, "What is done when error E-21 returns after recalibration?", "fact", ((6, "replace the memory board"),), "replace the memory board, part number MB-4402"),
    # --- Page 7: storage and transport --------------------------------------
    _q(35, "What is the shelf life of spare impellers and O-rings?", "fact", ((7, "shelf life is five years"),), "five years from the bag date"),
    _q(36, "Why must the AX-40 be drained before transport in frost?", "fact", ((7, "trapped water cracks the housing"),), "trapped water cracks the housing"),
    _q(37, "Which part number are the transport brackets?", "fact", ((7, "part number TB-9901"),), "TB-9901"),
    _q(38, "How long does the panel clock battery last?", "fact", ((7, "last about two years"),), "about two years"),
    _q(39, "Which test must be run when returning a stored unit to service?", "fact", ((7, "run the leak test from the heater chapter"),), "the leak test from the heater chapter"),
    # --- Page 8: calibration log requirements -------------------------------
    _q(40, "What does every service visit need?", "fact", ((8, "needs a completed calibration register entry"),), "a completed calibration register entry, even when nothing was adjusted"),
    _q(41, "What information does a calibration register entry contain?", "fact", ((8, "the unit serial number, the date, the technician name, the reference meter value and the panel value"),), "serial number, date, technician name, reference meter value, panel value"),
    _q(42, "Who does not accept pencil register entries?", "fact", ((8, "pencil entries are not accepted by the auditor"),), "the auditor"),
    _q(43, "How are corrections made in the calibration register?", "fact", ((8, "a single line through the wrong value plus initials"),), "a single line through the wrong value plus initials"),
    _q(44, "How long must completed registers be kept?", "fact", ((8, "for at least seven years"),), "at least seven years"),
    _q(45, "Who signs each register page at the end of the month?", "fact", ((8, "The site manager signs each register page"),), "the site manager"),
    # --- Page 9: consumables --------------------------------------------------
    _q(46, "In what bottle size does HL-9 descaling fluid come?", "fact", ((9, "comes in one litre bottles"),), "one litre bottles, article AR-5010"),
    _q(47, "Which grease is used for impeller shafts?", "fact", ((9, "silicon free grease"),), "silicon free grease, article AR-5035"),
    _q(48, "Why is lithium grease forbidden?", "fact", ((9, "never use lithium grease because it attacks the O-ring"),), "it attacks the O-ring"),
    _q(49, "Which fuses may be fitted to the drive board?", "fact", ((9, "only ceramic fuses may be fitted"),), "T two amp ceramic fuses, article AR-5045"),
    _q(50, "How long do standard deliveries take?", "fact", ((9, "Deliveries take five working days"),), "five working days"),
    # --- Page 10: warranty ----------------------------------------------------
    _q(51, "How long is the AX-40 warranty?", "fact", ((10, "twenty four month warranty"),), "twenty four months from the commissioning date"),
    _q(52, "Which document is required for a warranty claim?", "fact", ((10, "Keep the commissioning certificate"),), "the commissioning certificate"),
    _q(53, "Who may perform warranty service?", "fact", ((10, "technician with a valid AX-40 certificate"),), "a technician with a valid AX-40 certificate"),
    _q(54, "Where is the regional spare parts warehouse?", "fact", ((10, "warehouse is in Rotterdam"),), "Rotterdam"),
    _q(55, "What is the cut-off for same-day shipping?", "fact", ((10, "orders placed before fourteen hours ship the same day"),), "orders placed before fourteen hours"),
    # --- Paraphrases: reworded questions --------------------------------------
    _q(56, "How hot can the housing get while the pump is running?", "paraphrase", ((1, "reach eighty degrees Celsius"),), "eighty degrees Celsius"),
    _q(57, "What should the difference between panel and reference readouts be after calibration?", "paraphrase", ((3, "under two percent"),), "under two percent"),
    _q(58, "What should you check first when the flow sensor gives no signal?", "paraphrase", ((6, "check the sensor connector first"),), "the sensor connector"),
    _q(59, "How fast can emergency parts arrive?", "paraphrase", ((9, "couriered in twenty four hours"),), "within twenty four hours, at extra cost"),
    _q(60, "Can I re-fit the old O-ring after changing the impeller?", "paraphrase", ((3, "Never reuse a deformed O-ring"),), "no - always fit a new OR-8841; never reuse a deformed O-ring"),
    _q(61, "What protects the control panel when the unit is stored outside?", "paraphrase", ((7, "covered with the rain shroud"),), "the rain shroud"),
    _q(62, "How cool must the heater be before the thermostat is reset?", "paraphrase", ((6, "cool below sixty degrees"),), "below sixty degrees"),
    _q(63, "What fraction of full scale can the pressure transducer be off by?", "paraphrase", ((5, "one percent full scale error"),), "one percent"),
    _q(64, "What must exist before an out-of-warranty repair starts?", "paraphrase", ((10, "need a written approval before work starts"),), "a written approval"),
    _q(65, "What does the warranty say about opening the drive housing yourself?", "paraphrase", ((10, "unauthorised opening of the drive housing"),), "damage from it is excluded from the warranty"),
    # --- Cross-passage: two gold snippets -------------------------------------
    _q(66, "Which descaling fluid goes with the heater element rated at one thousand two hundred watts?", "cross", ((2, "one thousand two hundred watts"), (9, "HL-9 descaling fluid comes in one litre bottles")), "HL-9"),
    _q(67, "Error E-07 points to low pressure: which filter cartridge is the usual suspect?", "cross", ((6, "E-07 signals a low pressure condition"), (4, "part number FL-5560")), "the inlet filter cartridge FL-5560 / a leaking joint"),
    _q(68, "If error E-01 appears, what kind of connector do the probes plug into?", "cross", ((6, "E-01 means the flow sensor returned no signal"), (5, "six pin connectors")), "check the sensor connector; probes use six pin connectors"),
    _q(69, "Which O-ring goes with the impeller, and which grease must never be used near it?", "cross", ((3, "part number OR-8841"), (9, "never use lithium grease")), "OR-8841; lithium grease attacks the O-ring"),
    _q(70, "At what pressure is the heater leak test run, and what drop means a loose joint?", "cross", ((2, "pressurise the loop to three bar"), (2, "greater than zero point two bar")), "three bar; a drop greater than 0.2 bar"),
    _q(71, "After an E-21 checksum error returns, which board is replaced and which fuse type is allowed on the drive board?", "cross", ((6, "part number MB-4402"), (9, "T two amp ceramic")), "memory board MB-4402; T two amp ceramic fuses"),
    _q(72, "What pressure does the leak test use, and what is the maximum pressure the transducer measures?", "cross", ((2, "pressurise the loop to three bar"), (5, "measures from zero to ten bar")), "three bar; the PT-7712 measures zero to ten bar"),
    _q(73, "How long are calibration entries valid, and how long must registers be kept?", "cross", ((3, "older than twelve months"), (8, "for at least seven years")), "entries valid twelve months; registers kept seven years"),
    _q(74, "Which grease attacks the O-ring, and which water rinses the heater after descaling?", "cross", ((9, "never use lithium grease"), (2, "demineralised water")), "lithium grease; demineralised water"),
    _q(75, "What is the panel readout compared with during calibration, and how accurate are the temperature probes?", "cross", ((3, "reference flow meter"), (5, "plus or minus one point five degrees")), "the reference flow meter; probes plus or minus 1.5 degrees"),
    # --- Unanswerable: the fact is NOT in the document -------------------------
    _q(76, "What is the battery life of the handheld remote control?", "unanswerable", ()),
    _q(77, "How do I connect the AX-40 to the building's Wi-Fi network?", "unanswerable", ()),
    _q(78, "What is the torque spec for the cabinet screws?", "unanswerable", ()),
    _q(79, "Which European directive covers the AX-40 noise emissions?", "unanswerable", ()),
    _q(80, "How much does the AX-40 weigh?", "unanswerable", ()),
    _q(81, "What coolant type is used in the chiller module?", "unanswerable", ()),
    _q(82, "How often should the drive belt be replaced?", "unanswerable", ()),
    _q(83, "What is the maximum flow rate of the AX-40 in litres per minute?", "unanswerable", ()),
    _q(84, "Which app is used for remote monitoring?", "unanswerable", ()),
    _q(85, "What is the price of a replacement impeller?", "unanswerable", ()),
    _q(86, "How do I change the menu language on the control panel?", "unanswerable", ()),
    _q(87, "What oil is recommended for the gearbox?", "unanswerable", ()),
    _q(88, "How many service technicians are needed to lift the unit by hand?", "unanswerable", ()),
    _q(89, "What is the serial number format of the AX-40?", "unanswerable", ()),
    _q(90, "Does the warranty cover lightning damage?", "unanswerable", ()),
    _q(91, "Where is the main factory located?", "unanswerable", ()),
    _q(92, "What is the voltage of the panel clock battery?", "unanswerable", ()),
    _q(93, "How often are the emergency stop buttons tested?", "unanswerable", ()),
    _q(94, "What training is required before ordering spare parts?", "unanswerable", ()),
    _q(95, "Which shipping company delivers the emergency parts?", "unanswerable", ()),
    _q(96, "What is the sound pressure level of the pump at one metre?", "unanswerable", ()),
    _q(97, "Can the AX-40 be used with seawater?", "unanswerable", ()),
    _q(98, "How many AX-40 units fit on one pallet?", "unanswerable", ()),
    _q(99, "What is the email address of the service desk?", "unanswerable", ()),
    _q(100, "Which torque wrench is recommended for the impeller bolt?", "unanswerable", ()),
]


def count_by_category() -> dict[str, int]:
    """How many questions per category (sanity check for the dataset)."""
    counts: dict[str, int] = {}
    for question in QUESTIONS:
        counts[question.category] = counts.get(question.category, 0) + 1
    return counts
