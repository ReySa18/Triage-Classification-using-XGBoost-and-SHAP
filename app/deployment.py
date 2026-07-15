"""Deployment bootstrap helpers for model artifact availability."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT_DIR / "model" / "artifacts"
RETRAIN_SCRIPT = ROOT_DIR / "scripts" / "retrain.py"
RETRAIN_TIMEOUT_SECONDS = 900

# Keep this manifest aligned with the files loaded by backend.predictor.
REQUIRED_ARTIFACTS = (
    ARTIFACT_DIR / "model_triage_xgb.pkl",
    ARTIFACT_DIR / "scaler_minmax.pkl",
    ARTIFACT_DIR / "shap_explainer.pkl",
    ARTIFACT_DIR / "feature_names.pkl",
    ARTIFACT_DIR / "imputer.pkl",
    ARTIFACT_DIR / "best_thresholds.pkl",
)

_SENSITIVE_VALUE = re.compile(
    r"(?i)(password|passwd|token|api[_-]?key|secret|credential)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_MAX_LOG_CHARACTERS = 8_000


class RetrainingError(RuntimeError):
    """A safe-to-display retraining failure."""

    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(message)
        self.details = details


def invalid_artifacts() -> tuple[Path, ...]:
    """Return required artifacts that are missing, empty, or not regular files."""
    invalid: list[Path] = []
    for path in REQUIRED_ARTIFACTS:
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                invalid.append(path)
        except OSError:
            invalid.append(path)
    return tuple(invalid)


def artifacts_available() -> bool:
    """Return True only when every inference artifact is a non-empty file."""
    return not invalid_artifacts()


def _as_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def sanitize_retraining_output(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    """Combine, redact, and bound subprocess output before displaying it."""
    sections = []
    stdout_text = _as_text(stdout).strip()
    stderr_text = _as_text(stderr).strip()
    if stdout_text:
        sections.append(f"Standard output:\n{stdout_text}")
    if stderr_text:
        sections.append(f"Standard error:\n{stderr_text}")

    safe_output = "\n\n".join(sections)
    safe_output = _SENSITIVE_VALUE.sub(r"\1\2[REDACTED]", safe_output)
    if len(safe_output) > _MAX_LOG_CHARACTERS:
        safe_output = "... output dipotong ...\n" + safe_output[-_MAX_LOG_CHARACTERS:]
    return safe_output


def run_retraining() -> str:
    """Run retraining in the Streamlit interpreter and return sanitized logs."""
    if not RETRAIN_SCRIPT.is_file():
        raise RetrainingError(
            "Script retraining tidak ditemukan.",
            f"Path yang dicari: {RETRAIN_SCRIPT}",
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Memulai retraining: {RETRAIN_SCRIPT}", flush=True)

    try:
        result = subprocess.run(
            [sys.executable, str(RETRAIN_SCRIPT)],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=RETRAIN_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        details = sanitize_retraining_output(error.stdout, error.stderr)
        raise RetrainingError(
            f"Retraining melebihi batas waktu {RETRAIN_TIMEOUT_SECONDS // 60} menit.",
            details,
        ) from error
    except OSError as error:
        raise RetrainingError(
            "Proses retraining tidak dapat dijalankan.",
            str(error),
        ) from error

    safe_output = sanitize_retraining_output(result.stdout, result.stderr)
    if safe_output:
        print(safe_output, flush=True)

    if result.returncode != 0:
        details = safe_output or (
            f"Proses berhenti dengan exit code {result.returncode} tanpa output."
        )
        raise RetrainingError(
            f"Retraining gagal dengan exit code {result.returncode}.",
            details,
        )

    print("Retraining selesai.", flush=True)
    return safe_output
