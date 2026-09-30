"""Synthetic source data for the demo: a fictional HDSS population linked to clinic records.

Everything here is generated from a fixed seed; no real person is represented. The shape
mirrors what an African HDSS + clinic-EMR linkage typically delivers as CSV exports:

  individuals.csv         registry: id, sex coded 1/2/9, dd/mm/yyyy birth date, names, phone
  residency_episodes.csv  HDSS residence episodes with entry/exit event codes
  clinic_visits.csv       outpatient / inpatient / emergency visits
  diagnoses.csv           ICD-10 codes (some written without the dot) + free-text label
  prescriptions.csv       local drug names, quantity and duration
  lab_results.csv         local test codes, numeric or POS/NEG results, unit strings
  deaths.csv              date of death and verbal-autopsy cause

Deliberate data problems (so quality checks and the repair loop have something to find):
  * outpatient visits have no discharge date;
  * a few lab results belong to people missing from the registry;
  * a few people have sex code 9 (unknown);
  * a handful of rare drug names appear fewer than 5 times (small-cell suppression).
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

VILLAGES = ["Mwangaza", "Tumaini", "Upendo", "Baraka", "Neema", "Amani"]
FACILITIES = ["Mwangaza Health Centre", "Tumaini Dispensary", "County Referral Hospital"]
FIRST_NAMES = ["Achieng", "Wanjiru", "Otieno", "Kamau", "Njeri", "Mutua", "Akinyi", "Kiprop", "Wafula", "Chebet"]
LAST_NAMES = ["Odhiambo", "Mwangi", "Kariuki", "Onyango", "Cheruiyot", "Wekesa", "Nduta", "Barasa"]

# (icd10 code as recorded, label, weight, probability code is written without the dot)
DIAGNOSES = [
    ("B54", "Malaria, unspecified", 30, 0.0),
    ("J18.9", "Pneumonia, unspecified", 14, 0.3),
    ("A09", "Diarrhoea and gastroenteritis", 16, 0.0),
    ("I10", "Essential hypertension", 10, 0.0),
    ("E11.9", "Type 2 diabetes without complications", 6, 0.3),
    ("B20", "HIV disease", 8, 0.0),
    ("A15.0", "Pulmonary tuberculosis", 4, 0.3),
    ("D64.9", "Anaemia, unspecified", 8, 0.3),
    ("J06.9", "Upper respiratory tract infection", 18, 0.3),
]

# drug label -> (weight, days, qty)
DRUGS = {
    "AL 20/120 tabs": (26, 3, 24),
    "Paracetamol 500mg": (30, 3, 18),
    "Amoxicillin 250mg caps": (18, 5, 15),
    "ORS sachets": (12, 3, 6),
    "Zinc 20mg": (10, 10, 10),
    "Metformin 500mg": (6, 30, 60),
    "Amlodipine 5mg": (6, 30, 30),
    "TLD (TDF/3TC/DTG)": (6, 30, 30),
    "Cotrimoxazole 960mg": (5, 30, 30),
}
RARE_DRUGS = ["Quinine inj 600mg", "Ceftriaxone 1g inj"]

# test code -> (unit, generator)
LABS = ["HB", "MRDT", "RBS", "WT", "TEMP", "SBP", "DBP"]

DIAG_DRUGS = {
    "B54": ["AL 20/120 tabs", "Paracetamol 500mg"],
    "J18.9": ["Amoxicillin 250mg caps", "Paracetamol 500mg"],
    "A09": ["ORS sachets", "Zinc 20mg"],
    "I10": ["Amlodipine 5mg"],
    "E11.9": ["Metformin 500mg"],
    "B20": ["TLD (TDF/3TC/DTG)", "Cotrimoxazole 960mg"],
    "A15.0": ["Paracetamol 500mg"],
    "D64.9": ["Paracetamol 500mg"],
    "J06.9": ["Paracetamol 500mg"],
}

VA_CAUSES = ["Malaria", "HIV/AIDS related death", "Pneumonia", "Road traffic accident", "Diarrhoeal diseases"]

START = date(2015, 1, 1)
END = date(2023, 12, 31)


def _rand_date(rng: random.Random, lo: date, hi: date) -> date:
    if hi <= lo:
        return lo
    return lo + timedelta(days=rng.randrange((hi - lo).days + 1))


def _write(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])


def generate(out_dir: str | Path, n_people: int = 1500, seed: int = 42) -> dict[str, int]:
    """Write the demo CSV files into ``out_dir``; returns row counts per file."""
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    individuals, residency, visits, dx_rows, rx_rows, labs, deaths = [], [], [], [], [], [], []
    visit_seq = 0

    for i in range(1, n_people + 1):
        pid = f"KH{i:06d}"
        sex = rng.choices(["1", "2", "9"], weights=[48, 50, 2])[0]
        born_before_start = rng.random() < 0.8
        dob = (
            _rand_date(rng, date(1940, 1, 1), START - timedelta(days=1))
            if born_before_start
            else _rand_date(rng, START, date(2022, 12, 31))
        )
        individuals.append(
            [
                pid,
                sex,
                dob.strftime("%d/%m/%Y"),
                rng.choice(VILLAGES),
                f"HH{rng.randrange(1, n_people // 3):05d}",
                rng.choice(FIRST_NAMES),
                rng.choice(LAST_NAMES),
                f"07{rng.randrange(10_000_000, 99_999_999)}",
            ]
        )

        # Residency: entered by birth or enumeration/in-migration, left by death, out-migration,
        # or still resident at the end of follow-up.
        entry = dob if not born_before_start else _rand_date(rng, START, date(2019, 12, 31))
        entry_event = "BTH" if not born_before_start else rng.choice(["ENU", "ENU", "IMG"])
        u = rng.random()
        dod: date | None = None
        if u < 0.06:
            exit_date = _rand_date(rng, entry + timedelta(days=30), END)
            exit_event, dod = "DTH", exit_date
        elif u < 0.2:
            exit_date, exit_event = _rand_date(rng, entry + timedelta(days=30), END), "OMG"
        else:
            exit_date, exit_event = END, "OBE"
        residency.append([pid, entry.isoformat(), entry_event, exit_date.isoformat(), exit_event])
        if dod:
            deaths.append([pid, dod.isoformat(), rng.choice(VA_CAUSES)])

        # Clinic visits while resident.
        for _ in range(rng.choices([0, 1, 2, 3, 4, 6], weights=[15, 25, 25, 15, 12, 8])[0]):
            visit_seq += 1
            vid = f"V{visit_seq:07d}"
            vdate = _rand_date(rng, entry, exit_date)
            vtype = rng.choices(["OPD", "IPD", "EMERG"], weights=[80, 12, 8])[0]
            discharge = vdate + timedelta(days=rng.randrange(1, 8)) if vtype == "IPD" else None
            visits.append([vid, pid, vdate.isoformat(), discharge.isoformat() if discharge else None, vtype, rng.choice(FACILITIES)])

            codes = rng.choices([d[0] for d in DIAGNOSES], weights=[d[2] for d in DIAGNOSES], k=rng.choice([1, 1, 2]))
            for code in dict.fromkeys(codes):
                _, label, _, nodot = next(d for d in DIAGNOSES if d[0] == code)
                recorded = code.replace(".", "") if rng.random() < nodot else code
                dx_rows.append([vid, pid, recorded, label, vdate.isoformat()])
                for drug in DIAG_DRUGS[code]:
                    _, days, qty = DRUGS[drug]
                    rx_rows.append([vid, pid, drug, qty, days, vdate.isoformat()])

            for test in rng.sample(LABS, k=rng.randrange(1, 4)):
                labs.append([vid, pid, *_lab_result(rng, test, codes), vdate.isoformat()])

    # Rare drug names: below the small-cell threshold, so never shown to the LLM.
    for drug in RARE_DRUGS:
        for _ in range(rng.randrange(1, 4)):
            v = rng.choice(visits)
            rx_rows.append([v[0], v[1], drug, 1, 1, v[2]])

    # Orphan lab results: people who are not in the registry.
    for k in range(12):
        labs.append([None, f"KX{k:06d}", "HB", f"{rng.uniform(8, 15):.1f}", "g/dL", _rand_date(rng, START, END).isoformat()])

    _write(out / "individuals.csv", ["individual_id", "sex", "dob", "village", "hh_id", "first_name", "last_name", "phone"], individuals)
    _write(out / "residency_episodes.csv", ["individual_id", "start_date", "start_event", "end_date", "end_event"], residency)
    _write(out / "clinic_visits.csv", ["visit_id", "individual_id", "visit_date", "discharge_date", "visit_type", "facility"], visits)
    _write(out / "diagnoses.csv", ["visit_id", "individual_id", "icd10", "dx_text", "dx_date"], dx_rows)
    _write(out / "prescriptions.csv", ["visit_id", "individual_id", "drug", "qty", "days", "date_prescribed"], rx_rows)
    _write(out / "lab_results.csv", ["visit_id", "individual_id", "test", "result", "unit", "test_date"], labs)
    _write(out / "deaths.csv", ["individual_id", "dod", "va_cause"], deaths)

    return {
        "individuals": len(individuals),
        "residency_episodes": len(residency),
        "clinic_visits": len(visits),
        "diagnoses": len(dx_rows),
        "prescriptions": len(rx_rows),
        "lab_results": len(labs),
        "deaths": len(deaths),
    }


def _lab_result(rng: random.Random, test: str, codes: list[str]) -> tuple[str, str]:
    if test == "HB":
        return f"{rng.uniform(6.5 if 'D64.9' in codes else 10, 16.5):.1f}", "g/dL"
    if test == "MRDT":
        return ("POS" if "B54" in codes or rng.random() < 0.1 else "NEG"), ""
    if test == "RBS":
        return f"{rng.uniform(9 if 'E11.9' in codes else 3.9, 18 if 'E11.9' in codes else 7.8):.1f}", "mmol/L"
    if test == "WT":
        return f"{rng.uniform(8, 95):.1f}", "kg"
    if test == "TEMP":
        return f"{rng.uniform(37.8 if 'B54' in codes else 36.2, 39.9 if 'B54' in codes else 37.4):.1f}", "C"
    if test == "SBP":
        return str(rng.randrange(140 if "I10" in codes else 100, 185 if "I10" in codes else 135)), "mmHg"
    if test == "DBP":
        return str(rng.randrange(90 if "I10" in codes else 60, 110 if "I10" in codes else 88)), "mmHg"
    raise ValueError(test)
