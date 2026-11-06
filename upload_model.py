import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


endpoint = os.getenv("S3_ENDPOINT", "http://127.0.0.1:8333")
bucket = os.getenv("S3_BUCKET", "model-storage")
model_dir = Path(
    os.getenv("MODEL_DIR", "models/pythia-70m-command-parser")
)

access_key = os.getenv("AWS_ACCESS_KEY_ID", "ccadmin")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "ccadmin-secret")

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name="us-east-1",
)


def ensure_bucket() -> None:
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"Bucket already exists: {bucket}")
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code not in {"404", "NoSuchBucket", "400"}:
            raise

        print(f"Creating bucket: {bucket}")
        s3.create_bucket(Bucket=bucket)


def object_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in {"404", "NoSuchKey", "NoSuchBucket"}:
            return False
        raise


def main() -> None:
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory does not exist: {model_dir}"
        )

    files = [path for path in model_dir.rglob("*") if path.is_file()]

    if not files:
        raise RuntimeError(f"No model files found in {model_dir}")

    ensure_bucket()

    uploaded = 0
    skipped = 0

    print(f"Checking {len(files)} model files...")

    for path in files:
        relative_path = path.relative_to(model_dir).as_posix()
        key = f"pythia-70m-command-parser/{relative_path}"

        if object_exists(key):
            print(f"Skipping existing: {key}")
            skipped += 1
            continue

        print(f"Uploading: {key}")
        s3.upload_file(str(path), bucket, key)
        uploaded += 1

    print(
        f"MODEL SYNC COMPLETE: uploaded={uploaded}, skipped={skipped}"
    )


if __name__ == "__main__":
    main()
