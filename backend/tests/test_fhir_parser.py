from __future__ import annotations

import copy
import json
from datetime import date

import pytest

from models.exceptions import FHIRParseError
from models.patient import LabFlag, VitalType
from services.fhir_parser import FHIRParser, VITAL_LOINC_CODES


def parse_bundle(bundle: dict[str, object]):
    """Parse a bundle for parser tests."""
    return FHIRParser(bundle).extract()


def test_extract_demographics_complete(patient_context):
    assert patient_context.patient_id == "patient-001"
    assert patient_context.mrn == "MRN-001"
    assert patient_context.first_name == "Ada"
    assert patient_context.last_name == "Lovelace"
    assert patient_context.race == "White"
    assert patient_context.ethnicity == "Not Hispanic or Latino"
    assert patient_context.address_state == "MA"


def test_extract_only_active_conditions(patient_context):
    names = patient_context.get_condition_names()
    assert names == ["Type 2 diabetes mellitus", "Essential hypertension", "Asthma"]
    assert "Acute appendicitis" not in names


def test_extract_only_active_medications(patient_context):
    names = patient_context.get_medication_names()
    assert names == ["Metformin", "Lisinopril", "Atorvastatin"]
    assert "Warfarin" not in names
    assert "Simvastatin" not in names


def test_extract_allergies_with_severity(patient_context):
    severities = {allergy.substance: allergy.severity.value for allergy in patient_context.allergies}
    assert severities["Penicillin"] == "severe"
    assert severities["Peanut"] == "moderate"
    assert severities["Latex"] == "mild"


def test_labs_separated_from_vitals(patient_context):
    vital_codes = set(VITAL_LOINC_CODES)
    assert patient_context.recent_labs
    assert all(lab.loinc_code not in vital_codes for lab in patient_context.recent_labs)


def test_vitals_separated_from_labs(patient_context):
    vital_types = {vital.vital_type for vital in patient_context.recent_vitals}
    assert VitalType.heart_rate in vital_types
    assert VitalType.blood_pressure in vital_types
    assert all(vital.vital_type != "Potassium" for vital in patient_context.recent_vitals)


def test_bp_panel_parsed_as_composite_vital(patient_context):
    blood_pressure = next(
        vital for vital in patient_context.recent_vitals if vital.vital_type == VitalType.blood_pressure
    )
    assert blood_pressure.systolic == 148
    assert blood_pressure.diastolic == 92
    assert blood_pressure.unit == "mmHg"


def test_critical_lab_flag_detected(patient_context):
    potassium = patient_context.get_lab_by_loinc("2823-3")
    assert potassium is not None
    assert potassium.flag == LabFlag.HH
    assert patient_context.has_critical_labs is True


def test_lab_sorted_by_date_descending(patient_context):
    recorded = [lab.recorded_at for lab in patient_context.recent_labs]
    assert recorded == sorted(recorded, reverse=True)


def test_labs_capped_at_twenty(fhir_bundle):
    bundle = copy.deepcopy(fhir_bundle)
    lab_template = next(
        entry for entry in bundle["entry"] if entry["resource"].get("id") == "lab-wbc"
    )
    for index in range(25):
        entry = copy.deepcopy(lab_template)
        resource = entry["resource"]
        resource["id"] = f"extra-lab-{index}"
        resource["code"]["coding"][0]["code"] = f"9{index:03d}-1"
        resource["effectiveDateTime"] = f"2026-06-{(index % 28) + 1:02d}T08:00:00Z"
        bundle["entry"].append(entry)

    ctx = parse_bundle(bundle)
    assert len(ctx.recent_labs) == 20


def test_missing_optional_fields_handled():
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "minimal-patient",
                    "birthDate": "2000-01-01",
                    "gender": "unknown",
                }
            }
        ],
    }
    ctx = parse_bundle(bundle)
    assert ctx.patient_id == "minimal-patient"
    assert ctx.first_name == "Unknown"
    assert ctx.active_conditions == []


def test_empty_bundle_raises_fhir_parse_error():
    with pytest.raises(FHIRParseError):
        parse_bundle({"resourceType": "Bundle", "type": "collection", "entry": []})


def test_invalid_json_raises_fhir_parse_error():
    with pytest.raises(FHIRParseError):
        FHIRParser.from_bytes(b"{not-json")


def test_non_object_json_raises_fhir_parse_error():
    with pytest.raises(FHIRParseError):
        FHIRParser.from_bytes(b"[]")


def test_invalid_bundle_shape_raises_fhir_parse_error():
    with pytest.raises(FHIRParseError):
        FHIRParser.from_bytes(b'{"resourceType":"Patient","id":"not-a-bundle"}')


def test_from_file_parses_valid_bundle(tmp_path, fhir_bundle):
    fixture = tmp_path / "sample_patient.json"
    fixture.write_text(json.dumps(fhir_bundle), encoding="utf-8")
    assert FHIRParser.from_file(fixture).extract().patient_id == "patient-001"


def test_clinical_summary_contains_patient_name(patient_context):
    assert "Ada Lovelace" in patient_context.to_clinical_summary()


def test_clinical_summary_contains_all_conditions(patient_context):
    summary = patient_context.to_clinical_summary()
    assert "Type 2 diabetes mellitus" in summary
    assert "Essential hypertension" in summary
    assert "Asthma" in summary
    assert "E11.9" in summary
    assert "I10" in summary


def test_clinical_summary_flags_critical_labs(patient_context):
    summary = patient_context.to_clinical_summary()
    assert "Potassium" in summary
    assert "HH" in summary
    assert "CRITICAL" in summary


def test_age_calculated_correctly(patient_context):
    today = date.today()
    expected = today.year - 1975 - (1 if (today.month, today.day) < (12, 10) else 0)
    assert patient_context.age == expected


def test_has_condition_by_icd10_prefix(patient_context):
    assert patient_context.has_condition_by_icd10("E11")
    assert patient_context.has_condition_by_icd10("i")
    assert not patient_context.has_condition_by_icd10("K35")


def test_from_bytes_parses_valid_bundle(fhir_bundle):
    parser = FHIRParser.from_bytes(json.dumps(fhir_bundle).encode("utf-8"))
    assert parser.extract().patient_id == "patient-001"
