from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import httpx

try:
    from rich.console import Console
except ImportError:
    Console = None


@dataclass(frozen=True)
class FileLoadResult:
    """Per-file loading result."""

    file: str
    status: str
    status_code: int | None
    patient_id: str | None
    message: str | None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        None.

    Returns:
        Parsed argparse namespace.

    Raises:
        SystemExit: If arguments are invalid.
    """
    parser = argparse.ArgumentParser(description="Bulk load FHIR JSON bundles through the API.")
    parser.add_argument("--dir", required=True, help="Directory containing FHIR bundle JSON files.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent upload workers.")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading them.")
    parser.add_argument(
        "--api-url",
        default=os.getenv("CLINICAL_COPILOT_API_URL", "http://localhost:8000"),
        help="Base URL for the Clinical Copilot backend.",
    )
    return parser.parse_args()


def emit(message: str) -> None:
    """Print progress with rich when installed.

    Args:
        message: Message to display.

    Returns:
        None.

    Raises:
        None.
    """
    if Console is not None:
        Console().print(message)
    else:
        print(message)


async def upload_file(
    client: httpx.AsyncClient,
    path: Path,
    semaphore: asyncio.Semaphore,
    dry_run: bool,
) -> FileLoadResult:
    """Upload one FHIR file through the API.

    Args:
        client: HTTP client configured with the backend base URL.
        path: JSON file path.
        semaphore: Concurrency limiter.
        dry_run: Whether to skip network calls.

    Returns:
        Per-file load result.

    Raises:
        None.
    """
    async with semaphore:
        if dry_run:
            emit(f"[cyan]dry-run[/cyan] {path.name}")
            return FileLoadResult(str(path), "dry_run", None, None, None)

        try:
            data = await asyncio.to_thread(path.read_bytes)
            files = {"file": (path.name, data, "application/json")}
            response = await client.post("/fhir/patient/load", files=files)
            if response.status_code == 200:
                payload = response.json()
                emit(f"[green]loaded[/green] {path.name} -> {payload.get('patient_id')}")
                return FileLoadResult(
                    str(path),
                    "success",
                    response.status_code,
                    payload.get("patient_id"),
                    None,
                )
            if response.status_code == 409:
                emit(f"[yellow]skipped duplicate[/yellow] {path.name}")
                return FileLoadResult(str(path), "skipped", response.status_code, None, response.text)
            emit(f"[red]failed[/red] {path.name}: HTTP {response.status_code}")
            return FileLoadResult(str(path), "failed", response.status_code, None, response.text)
        except Exception as exc:
            emit(f"[red]failed[/red] {path.name}: {exc}")
            return FileLoadResult(str(path), "failed", None, None, str(exc))


async def run(args: argparse.Namespace) -> dict[str, object]:
    """Run the bulk loader.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Summary report dictionary.

    Raises:
        FileNotFoundError: If the input directory does not exist.
    """
    input_dir = Path(args.dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    files = sorted(input_dir.glob("*.json"))
    semaphore = asyncio.Semaphore(max(args.workers, 1))
    async with httpx.AsyncClient(base_url=args.api_url, timeout=60.0) as client:
        results = await asyncio.gather(
            *(upload_file(client, path, semaphore, args.dry_run) for path in files)
        )

    success_count = sum(1 for result in results if result.status == "success")
    fail_count = sum(1 for result in results if result.status == "failed")
    skip_count = sum(1 for result in results if result.status in {"skipped", "dry_run"})

    report: dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_files": len(files),
        "success_count": success_count,
        "fail_count": fail_count,
        "skip_count": skip_count,
        "results": [asdict(result) for result in results],
    }
    report_path = Path(f"load_report_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    emit(f"Report written to {report_path}")
    return report


def main() -> None:
    """Execute the bulk loader CLI.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If loading fails before a report can be produced.
    """
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
