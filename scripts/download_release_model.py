"""Download and verify the student model from the GitHub Release."""

import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path


REPOSITORY = os.getenv(
    "GITHUB_REPOSITORY",
    "thakuranubhav/cloud_computing_project",
)
RELEASE_TAG = os.getenv("RELEASE_TAG", "final")
MODEL_ASSET = os.getenv(
    "MODEL_ASSET",
    "pythia-70m-command-parser.zip",
)

MODEL_ROOT = Path("models")
MODEL_DIR = MODEL_ROOT / "pythia-70m-command-parser"
CHECKSUM_DIR = MODEL_ROOT / "v1"
ZIP_PATH = Path("release-model.zip")

EXPECTED_CHECKSUM = (
    "1539f339dd6f056a7a934fa45abd1294b7e486f292096f75d0f8bd6d6cd2d3c6"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_model() -> None:
    url = (
        f"https://github.com/{REPOSITORY}/releases/download/"
        f"{RELEASE_TAG}/{MODEL_ASSET}"
    )

    print(f"Downloading model from: {url}")

    urllib.request.urlretrieve(url, ZIP_PATH)


def extract_model() -> None:
    extraction_dir = Path("release-model-extracted")

    if extraction_dir.exists():
        shutil.rmtree(extraction_dir)

    extraction_dir.mkdir(parents=True)

    print("Extracting model archive...")

    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(extraction_dir)

    config_files = list(extraction_dir.rglob("config.json"))

    if not config_files:
        raise RuntimeError(
            "Could not find config.json in downloaded model archive"
        )

    source_dir = config_files[0].parent

    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(source_dir, MODEL_DIR)

    shutil.rmtree(extraction_dir)


def write_checksum() -> None:
    CHECKSUM_DIR.mkdir(parents=True, exist_ok=True)

    checksum_path = CHECKSUM_DIR / "model.sha256"
    checksum_path.write_text(
        EXPECTED_CHECKSUM + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print(f"Release: {RELEASE_TAG}")
    print(f"Model asset: {MODEL_ASSET}")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    download_model()

    actual_checksum = sha256_file(ZIP_PATH)

    print(f"Downloaded SHA-256: {actual_checksum}")

    if actual_checksum.lower() != EXPECTED_CHECKSUM.lower():
        raise RuntimeError(
            "Model archive checksum verification failed: "
            f"expected {EXPECTED_CHECKSUM}, "
            f"got {actual_checksum}"
        )

    print("MODEL ARCHIVE CHECKSUM VERIFIED")

    extract_model()
    write_checksum()

    ZIP_PATH.unlink()

    print(f"Model extracted to: {MODEL_DIR}")
    print(
        "Checksum recorded at: "
        f"{CHECKSUM_DIR / 'model.sha256'}"
    )


if __name__ == "__main__":
    main()