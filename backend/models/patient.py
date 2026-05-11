from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class MedicationStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    entered_in_error = "entered-in-error"
    stopped = "stopped"


class ConditionClinicalStatus(str, Enum):
    active = "active"
    recurrence = "recurrence"
    relapse = "relapse"
    inactive = "inactive"
    remission = "remission"
    resolved = "resolved"


class AllergySeverity(str, Enum):
    mild = "mild"
    moderate = "moderate"
    severe = "severe"


class LabFlag(str, Enum):
    H = "H"
    L = "L"
    HH = "HH"
    LL = "LL"
    A = "A"


class VitalType(str, Enum):
    heart_rate = "heart_rate"
    systolic_bp = "systolic_bp"
    diastolic_bp = "diastolic_bp"
    temperature = "temperature"
    respiratory_rate = "respiratory_rate"
    oxygen_saturation = "oxygen_saturation"
    bmi = "bmi"
    weight = "weight"
    height = "height"
    blood_pressure = "blood_pressure"


class ImmutableClinicalModel(BaseModel):
    """Base model for immutable clinical domain data."""

    model_config = ConfigDict(frozen=True)


class Coding(ImmutableClinicalModel):
    system: str
    code: str
    display: str | None = None


class Medication(ImmutableClinicalModel):
    name: str
    rxnorm_code: str | None = None
    drugbank_id: str | None = None
    dosage_value: float | None = None
    dosage_unit: str | None = None
    route: str | None = None
    frequency: str | None = None
    status: MedicationStatus = MedicationStatus.active
    prescribed_date: date | None = None


class Condition(ImmutableClinicalModel):
    name: str
    icd10_code: str | None = None
    snomed_code: str | None = None
    onset_date: date | None = None
    abatement_date: date | None = None
    status: ConditionClinicalStatus = ConditionClinicalStatus.active
    body_site: str | None = None


class Allergy(ImmutableClinicalModel):
    substance: str
    rxnorm_code: str | None = None
    snomed_code: str | None = None
    allergy_type: str | None = None
    reaction: str | None = None
    severity: AllergySeverity | None = None
    onset_date: date | None = None


class LabResult(ImmutableClinicalModel):
    name: str
    value: float
    unit: str
    loinc_code: str | None = None
    reference_range_low: float | None = None
    reference_range_high: float | None = None
    reference_range_text: str | None = None
    flag: LabFlag | None = None
    recorded_at: datetime


class Vital(ImmutableClinicalModel):
    vital_type: VitalType
    value: float
    unit: str
    systolic: float | None = None
    diastolic: float | None = None
    recorded_at: datetime

    @field_validator("systolic", "diastolic")
    @classmethod
    def validate_blood_pressure_component(cls, value: float | None) -> float | None:
        """Validate optional blood pressure components.

        Args:
            value: Optional numeric blood pressure component.

        Returns:
            The validated value.

        Raises:
            ValueError: If the component is negative.
        """
        if value is not None and value < 0:
            raise ValueError("Blood pressure components cannot be negative")
        return value


class PatientContext(ImmutableClinicalModel):
    patient_id: str
    mrn: str | None = None
    first_name: str
    last_name: str
    birth_date: date
    gender: str
    race: str | None = None
    ethnicity: str | None = None
    address_city: str | None = None
    address_state: str | None = None
    parsed_at: datetime = Field(default_factory=datetime.utcnow)
    active_conditions: list[Condition] = Field(default_factory=list)
    active_medications: list[Medication] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    recent_labs: list[LabResult] = Field(default_factory=list)
    recent_vitals: list[Vital] = Field(default_factory=list)

    @computed_field
    @property
    def age(self) -> int:
        """Return the patient's age in completed years.

        Args:
            None.

        Returns:
            Age in years.

        Raises:
            None.
        """
        today = date.today()
        years = today.year - self.birth_date.year
        had_birthday = (today.month, today.day) >= (self.birth_date.month, self.birth_date.day)
        return years if had_birthday else years - 1

    @computed_field
    @property
    def full_name(self) -> str:
        """Return a display-ready patient name.

        Args:
            None.

        Returns:
            Patient first and last name.

        Raises:
            None.
        """
        return f"{self.first_name} {self.last_name}".strip()

    @computed_field
    @property
    def has_critical_labs(self) -> bool:
        """Return whether recent labs include a critical high or low flag.

        Args:
            None.

        Returns:
            True when any lab is HH or LL.

        Raises:
            None.
        """
        return any(lab.flag in (LabFlag.HH, LabFlag.LL) for lab in self.recent_labs)

    def to_clinical_summary(self) -> str:
        """Render a dense clinical summary suitable for LLM grounding.

        Args:
            None.

        Returns:
            SOAP-style plain text summary with demographics, problems,
            medications, allergies, and abnormal labs.

        Raises:
            None.
        """
        lines = [
            (
                f"Patient: {self.full_name}, {self.age}-year-old {self.gender}, "
                f"DOB {self.birth_date.isoformat()}, patient_id {self.patient_id}."
            ),
            "",
            "Problem List:",
        ]

        if self.active_conditions:
            for condition in self.active_conditions:
                code_parts = []
                if condition.icd10_code:
                    code_parts.append(f"ICD-10 {condition.icd10_code}")
                if condition.snomed_code:
                    code_parts.append(f"SNOMED {condition.snomed_code}")
                codes = f" ({'; '.join(code_parts)})" if code_parts else ""
                onset = f", onset {condition.onset_date.isoformat()}" if condition.onset_date else ""
                lines.append(f"- {condition.name}{codes}; status {condition.status.value}{onset}.")
        else:
            lines.append("- No active conditions documented.")

        lines.extend(["", "Medication Reconciliation:"])
        if self.active_medications:
            for medication in self.active_medications:
                dose = ""
                if medication.dosage_value is not None and medication.dosage_unit:
                    dose = f" {medication.dosage_value:g} {medication.dosage_unit}"
                route = f" {medication.route}" if medication.route else ""
                frequency = f" {medication.frequency}" if medication.frequency else ""
                code = f" (RxNorm {medication.rxnorm_code})" if medication.rxnorm_code else ""
                lines.append(
                    f"- {medication.name}{code}:{dose}{route}{frequency}; "
                    f"status {medication.status.value}."
                )
        else:
            lines.append("- No active medications documented.")

        lines.extend(["", "Allergies and Intolerances:"])
        if self.allergies:
            for allergy in self.allergies:
                severity = f", severity {allergy.severity.value}" if allergy.severity else ""
                reaction = f", reaction {allergy.reaction}" if allergy.reaction else ""
                code = f" (SNOMED {allergy.snomed_code})" if allergy.snomed_code else ""
                lines.append(f"- {allergy.substance}{code}{reaction}{severity}.")
        else:
            lines.append("- No allergies documented.")

        lines.extend(["", "Abnormal or Critical Recent Labs:"])
        abnormal_labs = [
            lab
            for lab in self.recent_labs
            if lab.flag is not None
            or (
                lab.reference_range_low is not None
                and lab.reference_range_high is not None
                and not lab.reference_range_low <= lab.value <= lab.reference_range_high
            )
        ]
        if abnormal_labs:
            for lab in abnormal_labs:
                flag = f" [{lab.flag.value}]" if lab.flag else ""
                marker = " CRITICAL" if lab.flag in (LabFlag.HH, LabFlag.LL) else ""
                ref = ""
                if lab.reference_range_text:
                    ref = f", ref {lab.reference_range_text}"
                elif lab.reference_range_low is not None and lab.reference_range_high is not None:
                    ref = f", ref {lab.reference_range_low:g}-{lab.reference_range_high:g}"
                lines.append(
                    f"- {lab.name}: {lab.value:g} {lab.unit}{flag}{marker}"
                    f"{ref}, recorded {lab.recorded_at.date().isoformat()}."
                )
        else:
            lines.append("- No abnormal or critical labs in the recent result set.")

        return "\n".join(lines)

    def get_medication_names(self) -> list[str]:
        """Return active medication names.

        Args:
            None.

        Returns:
            Medication names in source order.

        Raises:
            None.
        """
        return [medication.name for medication in self.active_medications]

    def get_condition_names(self) -> list[str]:
        """Return active condition names.

        Args:
            None.

        Returns:
            Condition names in source order.

        Raises:
            None.
        """
        return [condition.name for condition in self.active_conditions]

    def get_allergy_substances(self) -> list[str]:
        """Return allergy substances.

        Args:
            None.

        Returns:
            Allergy substance names in source order.

        Raises:
            None.
        """
        return [allergy.substance for allergy in self.allergies]

    def has_condition_by_icd10(self, prefix: str) -> bool:
        """Return whether any active condition has an ICD-10 prefix.

        Args:
            prefix: ICD-10 code prefix to match case-insensitively.

        Returns:
            True when a condition code starts with the prefix.

        Raises:
            None.
        """
        normalized = prefix.upper()
        return any(
            condition.icd10_code is not None
            and condition.icd10_code.upper().startswith(normalized)
            for condition in self.active_conditions
        )

    def get_lab_by_loinc(self, loinc_code: str) -> LabResult | None:
        """Return the first recent lab matching a LOINC code.

        Args:
            loinc_code: Exact LOINC code to find.

        Returns:
            Matching lab result, if present.

        Raises:
            None.
        """
        return next((lab for lab in self.recent_labs if lab.loinc_code == loinc_code), None)
