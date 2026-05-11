from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from pathlib import Path
from time import perf_counter

try:
    from fhir.resources.R4B.bundle import Bundle
except ImportError:
    from fhir.resources.bundle import Bundle
from models.exceptions import FHIRParseError
from models.patient import (
    Allergy,
    AllergySeverity,
    Condition,
    ConditionClinicalStatus,
    LabFlag,
    LabResult,
    Medication,
    MedicationStatus,
    PatientContext,
    Vital,
    VitalType,
)
from services.logging_config import get_logger

ACTIVE_CONDITION_CODES = {"active", "recurrence", "relapse"}
MAX_LABS = 20
MAX_VITALS = 15

VITAL_LOINC_CODES: dict[str, tuple[VitalType, str]] = {
    "8867-4": (VitalType.heart_rate, "bpm"),
    "8480-6": (VitalType.systolic_bp, "mmHg"),
    "8462-4": (VitalType.diastolic_bp, "mmHg"),
    "8310-5": (VitalType.temperature, "degC"),
    "9279-1": (VitalType.respiratory_rate, "/min"),
    "2708-6": (VitalType.oxygen_saturation, "%"),
    "39156-5": (VitalType.bmi, "kg/m2"),
    "29463-7": (VitalType.weight, "kg"),
    "8302-2": (VitalType.height, "cm"),
    "55284-4": (VitalType.blood_pressure, "mmHg"),
}


class FHIRParser:
    """Parser for HL7 FHIR R4 bundles produced by Synthea."""

    def __init__(self, bundle: dict[str, object]) -> None:
        """Index all bundle entries by resourceType for O(1) lookup.

        Args:
            bundle: Decoded FHIR Bundle JSON.

        Returns:
            None.

        Raises:
            None.
        """
        self._resources: dict[str, list[dict[str, object]]] = {}
        self._logger = get_logger(__name__)
        entries = bundle.get("entry", [])
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                resource = entry.get("resource", {})
                if not isinstance(resource, dict):
                    continue
                rtype = resource.get("resourceType")
                if isinstance(rtype, str):
                    self._resources.setdefault(rtype, []).append(resource)

    @classmethod
    def from_file(cls, path: Path) -> FHIRParser:
        """Parse a FHIR bundle from a JSON file.

        Args:
            path: Filesystem path to a FHIR bundle.

        Returns:
            FHIRParser instance.

        Raises:
            FHIRParseError: If the file content is invalid JSON or not a Bundle.
        """
        try:
            return cls.from_bytes(path.read_bytes())
        except OSError as exc:
            raise FHIRParseError("FHIR bundle file could not be read", str(exc)) from exc

    @classmethod
    def from_bytes(cls, data: bytes) -> FHIRParser:
        """Parse a FHIR bundle from raw bytes.

        Args:
            data: Raw JSON bytes from an uploaded FHIR bundle.

        Returns:
            FHIRParser instance.

        Raises:
            FHIRParseError: If the payload is invalid JSON or not a FHIR Bundle.
        """
        try:
            bundle = json.loads(data)
        except json.JSONDecodeError as exc:
            raise FHIRParseError("FHIR bundle is not valid JSON", str(exc)) from exc

        if not isinstance(bundle, dict):
            raise FHIRParseError("FHIR bundle root must be a JSON object")

        try:
            if hasattr(Bundle, "model_validate"):
                Bundle.model_validate(bundle)
            else:
                Bundle.parse_obj(bundle)
        except Exception as exc:
            raise FHIRParseError("Payload is not a valid FHIR R4 Bundle", str(exc)) from exc

        return cls(bundle)

    def extract(self) -> PatientContext:
        """Extract a typed patient context from the bundle.

        Args:
            None.

        Returns:
            Parsed patient context.

        Raises:
            FHIRParseError: If the bundle does not include a Patient resource.
        """
        start = perf_counter()
        patient = self._extract_patient()
        demographics = self._extract_demographics(patient)
        labs, vitals = self._extract_observations()
        ctx = PatientContext(
            **demographics,
            active_conditions=self._extract_conditions(),
            active_medications=self._extract_medications(),
            allergies=self._extract_allergies(),
            recent_labs=labs,
            recent_vitals=vitals,
        )
        self._logger.debug(
            "fhir_bundle_extracted",
            patient_id=ctx.patient_id,
            duration_ms=int((perf_counter() - start) * 1000),
        )
        return ctx

    def _extract_patient(self) -> dict[str, object]:
        patients = self._resources.get("Patient", [])
        if not patients:
            raise FHIRParseError("FHIR bundle does not contain a Patient resource")
        return patients[0]

    def _extract_demographics(self, patient: dict[str, object]) -> dict[str, object]:
        try:
            names = self._as_list(patient.get("name"))
            official_name = self._first_matching(names, "use", "official") or (names[0] if names else {})
            given = self._as_list(official_name.get("given")) if isinstance(official_name, dict) else []
            family = official_name.get("family") if isinstance(official_name, dict) else None

            addresses = self._as_list(patient.get("address"))
            address = addresses[0] if addresses and isinstance(addresses[0], dict) else {}

            identifiers = self._as_list(patient.get("identifier"))
            patient_id = self._string(patient.get("id")) or self._extract_identifier_value(identifiers)
            if not patient_id:
                raise FHIRParseError("Patient resource does not include an id or identifier")

            birth_date = self._parse_date(self._string(patient.get("birthDate"))) or date(1900, 1, 1)

            return {
                "patient_id": patient_id,
                "mrn": self._extract_mrn(identifiers),
                "first_name": self._string(given[0]) if given else "Unknown",
                "last_name": self._string(family) or "Unknown",
                "birth_date": birth_date,
                "gender": self._string(patient.get("gender")) or "unknown",
                "race": self._extract_us_core_extension(patient, "race"),
                "ethnicity": self._extract_us_core_extension(patient, "ethnicity"),
                "address_city": self._string(address.get("city")) if isinstance(address, dict) else None,
                "address_state": self._string(address.get("state")) if isinstance(address, dict) else None,
            }
        except FHIRParseError:
            raise
        except Exception as exc:
            self._logger.debug("fhir_demographics_partial_failure", error=str(exc))
            raise FHIRParseError("Patient demographics could not be extracted", str(exc)) from exc

    def _extract_conditions(self) -> list[Condition]:
        conditions: list[Condition] = []
        for resource in self._resources.get("Condition", []):
            try:
                status = self._clinical_status(resource)
                if status not in ACTIVE_CONDITION_CODES:
                    continue
                codings = self._codings(resource.get("code"))
                name = self._display_text(resource.get("code")) or "Unknown condition"
                conditions.append(
                    Condition(
                        name=name,
                        icd10_code=self._get_coding_by_system(codings, "icd-10"),
                        snomed_code=self._get_coding_by_system(codings, "snomed"),
                        onset_date=self._parse_date(self._string(resource.get("onsetDateTime"))),
                        abatement_date=self._parse_date(self._string(resource.get("abatementDateTime"))),
                        status=ConditionClinicalStatus(status),
                        body_site=self._display_text(self._first_dict(self._as_list(resource.get("bodySite")))),
                    )
                )
            except Exception as exc:
                self._logger.debug(
                    "fhir_condition_skipped",
                    resource_id=self._string(resource.get("id")),
                    error=str(exc),
                )
        return conditions

    def _extract_medications(self) -> list[Medication]:
        medications: list[Medication] = []
        for resource in self._resources.get("MedicationRequest", []):
            try:
                status = self._string(resource.get("status")) or ""
                if status != MedicationStatus.active.value:
                    continue
                medication_concept = self._dict(resource.get("medicationCodeableConcept"))
                codings = self._codings(medication_concept)
                dosage = self._first_dict(self._as_list(resource.get("dosageInstruction")))
                dose, unit = self._extract_dose(dosage)
                medications.append(
                    Medication(
                        name=self._display_text(medication_concept) or "Unknown medication",
                        rxnorm_code=self._get_coding_by_system(codings, "rxnorm"),
                        dosage_value=dose,
                        dosage_unit=unit,
                        route=self._display_text(self._dict(dosage.get("route"))) if dosage else None,
                        frequency=self._extract_frequency(dosage),
                        status=MedicationStatus.active,
                        prescribed_date=self._parse_date(self._string(resource.get("authoredOn"))),
                    )
                )
            except Exception as exc:
                self._logger.debug(
                    "fhir_medication_skipped",
                    resource_id=self._string(resource.get("id")),
                    error=str(exc),
                )
        return medications

    def _extract_allergies(self) -> list[Allergy]:
        allergies: list[Allergy] = []
        for resource in self._resources.get("AllergyIntolerance", []):
            try:
                concept = self._dict(resource.get("code"))
                codings = self._codings(concept)
                reaction = self._first_dict(self._as_list(resource.get("reaction")))
                severity_raw = self._string(reaction.get("severity")) if reaction else None
                severity = self._parse_allergy_severity(severity_raw)
                manifestation = None
                if reaction:
                    manifestation = self._display_text(
                        self._first_dict(self._as_list(reaction.get("manifestation")))
                    )
                categories = self._as_list(resource.get("category"))
                allergies.append(
                    Allergy(
                        substance=self._display_text(concept) or "Unknown substance",
                        rxnorm_code=self._get_coding_by_system(codings, "rxnorm"),
                        snomed_code=self._get_coding_by_system(codings, "snomed"),
                        allergy_type=self._string(categories[0]) if categories else None,
                        reaction=manifestation,
                        severity=severity,
                        onset_date=self._parse_date(self._string(resource.get("onsetDateTime"))),
                    )
                )
            except Exception as exc:
                self._logger.debug(
                    "fhir_allergy_skipped",
                    resource_id=self._string(resource.get("id")),
                    error=str(exc),
                )
        return allergies

    def _extract_observations(self) -> tuple[list[LabResult], list[Vital]]:
        labs: list[LabResult] = []
        vitals: list[Vital] = []

        for resource in self._resources.get("Observation", []):
            try:
                loinc_code = self._get_loinc_code(resource)
                recorded_at = (
                    self._parse_datetime(self._string(resource.get("effectiveDateTime")))
                    or self._parse_datetime(self._string(resource.get("issued")))
                    or datetime.now(timezone.utc)
                )
                if loinc_code in VITAL_LOINC_CODES:
                    if loinc_code == "55284-4":
                        vitals.extend(self._parse_bp_panel(resource, recorded_at))
                    else:
                        value = self._quantity_value(resource)
                        if value is None:
                            continue
                        vital_type, default_unit = VITAL_LOINC_CODES[loinc_code]
                        vitals.append(
                            Vital(
                                vital_type=vital_type,
                                value=value,
                                unit=self._quantity_unit(resource) or default_unit,
                                recorded_at=recorded_at,
                            )
                        )
                    continue

                value = self._quantity_value(resource)
                if value is None:
                    continue
                reference = self._first_dict(self._as_list(resource.get("referenceRange")))
                interpretation = self._first_dict(self._as_list(resource.get("interpretation")))
                labs.append(
                    LabResult(
                        name=self._display_text(resource.get("code")) or "Unknown lab",
                        value=value,
                        unit=self._quantity_unit(resource) or "",
                        loinc_code=loinc_code,
                        reference_range_low=self._reference_value(reference, "low"),
                        reference_range_high=self._reference_value(reference, "high"),
                        reference_range_text=self._string(reference.get("text")) if reference else None,
                        flag=self._map_lab_flag(self._interpretation_code(interpretation)),
                        recorded_at=recorded_at,
                    )
                )
            except Exception as exc:
                self._logger.debug(
                    "fhir_observation_skipped",
                    resource_id=self._string(resource.get("id")),
                    error=str(exc),
                )

        labs.sort(key=lambda lab: lab.recorded_at, reverse=True)
        vitals.sort(key=lambda vital: vital.recorded_at, reverse=True)
        return labs[:MAX_LABS], vitals[:MAX_VITALS]

    def _parse_bp_panel(self, obs: dict[str, object], recorded_at: datetime) -> list[Vital]:
        try:
            systolic: float | None = None
            diastolic: float | None = None
            unit = VITAL_LOINC_CODES["55284-4"][1]
            for component in self._as_list(obs.get("component")):
                if not isinstance(component, dict):
                    continue
                code = self._get_loinc_code(component)
                value = self._quantity_value(component)
                if value is None:
                    continue
                unit = self._quantity_unit(component) or unit
                if code == "8480-6":
                    systolic = value
                elif code == "8462-4":
                    diastolic = value
            if systolic is None and diastolic is None:
                return []
            return [
                Vital(
                    vital_type=VitalType.blood_pressure,
                    value=systolic if systolic is not None else diastolic or 0,
                    unit=unit,
                    systolic=systolic,
                    diastolic=diastolic,
                    recorded_at=recorded_at,
                )
            ]
        except Exception as exc:
            self._logger.debug(
                "fhir_bp_panel_skipped",
                resource_id=self._string(obs.get("id")),
                error=str(exc),
            )
            return []

    def _get_loinc_code(self, resource: dict[str, object]) -> str | None:
        codings = self._codings(resource.get("code"))
        return self._get_coding_by_system(codings, "loinc")

    def _get_coding_by_system(self, codings: list[dict[str, object]], system_substr: str) -> str | None:
        needle = system_substr.lower()
        for coding in codings:
            system = self._string(coding.get("system")) or ""
            code = self._string(coding.get("code"))
            if needle in system.lower() and code:
                return code
        return None

    def _parse_date(self, raw: str | None) -> date | None:
        if not raw:
            return None
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            self._logger.debug("fhir_date_parse_failed")
            return None

    def _parse_datetime(self, raw: str | None) -> datetime | None:
        if not raw:
            return None
        try:
            if len(raw) == 10:
                return datetime.combine(date.fromisoformat(raw), time.min, tzinfo=timezone.utc)
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            self._logger.debug("fhir_datetime_parse_failed")
            return None

    def _map_lab_flag(self, interpretation_code: str | None) -> LabFlag | None:
        if interpretation_code is None:
            return None
        normalized = interpretation_code.upper()
        if normalized in LabFlag.__members__:
            return LabFlag[normalized]
        return None

    def _clinical_status(self, resource: dict[str, object]) -> str:
        status = self._dict(resource.get("clinicalStatus"))
        codings = self._codings(status)
        code = self._string(codings[0].get("code")) if codings else None
        return code or ConditionClinicalStatus.active.value

    def _codings(self, codeable_concept: object) -> list[dict[str, object]]:
        concept = self._dict(codeable_concept)
        return [coding for coding in self._as_list(concept.get("coding")) if isinstance(coding, dict)]

    def _display_text(self, codeable_concept: object) -> str | None:
        concept = self._dict(codeable_concept)
        text_value = self._string(concept.get("text"))
        if text_value:
            return text_value
        for coding in self._codings(concept):
            display = self._string(coding.get("display"))
            if display:
                return display
        return None

    def _quantity_value(self, resource: dict[str, object]) -> float | None:
        quantity = self._dict(resource.get("valueQuantity"))
        raw = quantity.get("value")
        if isinstance(raw, int | float):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw)
            except ValueError:
                return None
        return None

    def _quantity_unit(self, resource: dict[str, object]) -> str | None:
        quantity = self._dict(resource.get("valueQuantity"))
        return self._string(quantity.get("unit")) or self._string(quantity.get("code"))

    def _reference_value(self, reference: dict[str, object] | None, key: str) -> float | None:
        if not reference:
            return None
        bound = self._dict(reference.get(key))
        raw = bound.get("value")
        if isinstance(raw, int | float):
            return float(raw)
        return None

    def _interpretation_code(self, interpretation: dict[str, object] | None) -> str | None:
        if not interpretation:
            return None
        codings = self._codings(interpretation)
        return self._string(codings[0].get("code")) if codings else self._string(interpretation.get("text"))

    def _extract_dose(self, dosage: dict[str, object] | None) -> tuple[float | None, str | None]:
        if not dosage:
            return None, None
        dose_and_rate = self._first_dict(self._as_list(dosage.get("doseAndRate")))
        if not dose_and_rate:
            return None, None
        dose_quantity = self._dict(dose_and_rate.get("doseQuantity"))
        raw_value = dose_quantity.get("value")
        value = float(raw_value) if isinstance(raw_value, int | float) else None
        unit = self._string(dose_quantity.get("unit")) or self._string(dose_quantity.get("code"))
        return value, unit

    def _extract_frequency(self, dosage: dict[str, object] | None) -> str | None:
        if not dosage:
            return None
        text_value = self._string(dosage.get("text"))
        if text_value:
            return text_value
        timing = self._dict(dosage.get("timing"))
        repeat = self._dict(timing.get("repeat"))
        frequency = repeat.get("frequency")
        period = repeat.get("period")
        period_unit = self._string(repeat.get("periodUnit"))
        if isinstance(frequency, int | float) and isinstance(period, int | float) and period_unit:
            return f"{frequency:g} per {period:g} {period_unit}"
        return None

    def _parse_allergy_severity(self, severity: str | None) -> AllergySeverity | None:
        if not severity:
            return None
        normalized = severity.lower()
        if normalized in {item.value for item in AllergySeverity}:
            return AllergySeverity(normalized)
        return None

    def _extract_us_core_extension(self, patient: dict[str, object], kind: str) -> str | None:
        for extension in self._as_list(patient.get("extension")):
            if not isinstance(extension, dict):
                continue
            url = self._string(extension.get("url")) or ""
            if kind not in url.lower():
                continue
            for nested in self._as_list(extension.get("extension")):
                if not isinstance(nested, dict):
                    continue
                value_coding = self._dict(nested.get("valueCoding"))
                display = self._string(value_coding.get("display"))
                if display:
                    return display
        return None

    def _extract_identifier_value(self, identifiers: list[object]) -> str | None:
        for identifier in identifiers:
            if isinstance(identifier, dict):
                value = self._string(identifier.get("value"))
                if value:
                    return value
        return None

    def _extract_mrn(self, identifiers: list[object]) -> str | None:
        for identifier in identifiers:
            if not isinstance(identifier, dict):
                continue
            codings = self._codings(self._dict(identifier.get("type")))
            for coding in codings:
                if self._string(coding.get("code")) == "MR":
                    return self._string(identifier.get("value"))
        return self._extract_identifier_value(identifiers)

    @staticmethod
    def _as_list(value: object) -> list[object]:
        if isinstance(value, list):
            return value
        return []

    @staticmethod
    def _dict(value: object) -> dict[str, object]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _first_dict(values: list[object]) -> dict[str, object] | None:
        for value in values:
            if isinstance(value, dict):
                return value
        return None

    @staticmethod
    def _first_matching(values: list[object], key: str, expected: str) -> dict[str, object] | None:
        for value in values:
            if isinstance(value, dict) and value.get(key) == expected:
                return value
        return None

    @staticmethod
    def _string(value: object) -> str | None:
        return value if isinstance(value, str) and value else None
