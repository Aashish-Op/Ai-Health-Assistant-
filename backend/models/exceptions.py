from __future__ import annotations


class ClinicalCopilotError(Exception):
    """Base class for expected application errors."""

    def __init__(self, error_code: str, message: str, detail: str | None = None) -> None:
        """Create a domain exception.

        Args:
            error_code: Stable machine-readable code.
            message: Human-readable error message.
            detail: Optional implementation detail safe to return to clients.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.detail = detail


class FHIRParseError(ClinicalCopilotError):
    """Raised when a FHIR bundle cannot be parsed."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        """Create a FHIR parse error.

        Args:
            message: Human-readable error message.
            detail: Optional parse detail safe to return to clients.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__("FHIR_PARSE_ERROR", message, detail)


class PatientNotFoundError(ClinicalCopilotError):
    """Raised when a requested patient does not exist."""

    def __init__(self, patient_id: str) -> None:
        """Create a patient-not-found error.

        Args:
            patient_id: Missing patient identifier.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(
            "PATIENT_NOT_FOUND",
            f"Patient '{patient_id}' was not found",
            patient_id,
        )


class DatabaseError(ClinicalCopilotError):
    """Raised for expected database failures."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        """Create a database error.

        Args:
            message: Human-readable error message.
            detail: Optional database detail safe to return to clients.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__("DATABASE_ERROR", message, detail)


class DuplicatePatientError(ClinicalCopilotError):
    """Raised when a non-idempotent duplicate patient operation is rejected."""

    def __init__(self, patient_id: str) -> None:
        """Create a duplicate-patient error.

        Args:
            patient_id: Duplicate patient identifier.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(
            "DUPLICATE_PATIENT",
            f"Patient '{patient_id}' already exists",
            patient_id,
        )


class InvalidFileTypeError(ClinicalCopilotError):
    """Raised when an uploaded file is not a JSON FHIR bundle."""

    def __init__(self, filename: str | None) -> None:
        """Create an invalid-file-type error.

        Args:
            filename: Original filename if supplied by the client.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(
            "INVALID_FILE_TYPE",
            "FHIR uploads must be JSON files",
            filename,
        )


class FileTooLargeError(ClinicalCopilotError):
    """Raised when an uploaded file exceeds the accepted limit."""

    def __init__(self, max_bytes: int) -> None:
        """Create a file-too-large error.

        Args:
            max_bytes: Maximum accepted file size in bytes.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(
            "FILE_TOO_LARGE",
            "FHIR upload exceeds the maximum accepted file size",
            f"maximum_bytes={max_bytes}",
        )


class RetrievalServiceUnavailableError(ClinicalCopilotError):
    """Raised when vector retrieval is not configured or not reachable."""

    def __init__(self, detail: str | None = None) -> None:
        """Create a retrieval-unavailable error.

        Args:
            detail: Optional detail about the unavailable dependency.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(
            "RETRIEVAL_UNAVAILABLE",
            "Clinical evidence retrieval is unavailable",
            detail,
        )
