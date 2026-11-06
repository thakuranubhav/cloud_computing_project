import json
import os
import re
import socket
from pathlib import Path

import boto3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelLoader:
    def __init__(self):
        self.model = None
        self.tokenizer = None

        self.model_uri = os.getenv(
            "MODEL_URI",
            "s3://model-storage/pythia-70m-command-parser"
        )

        self.model_version = os.getenv("MODEL_VERSION", "v1")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "128"))

        self.s3_endpoint = os.getenv(
            "S3_ENDPOINT",
            "http://seaweedfs-s3:8333"
        )
        self.s3_access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
        self.s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")

        self.local_model_dir = Path("/tmp/model")

    def load(self):
        model_path = self._resolve_model_path()

        print(f"[MODEL] Loading model from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[MODEL] Model loaded successfully.")

    def _resolve_model_path(self) -> str:
        if self.model_uri.startswith("s3://"):
            return self._download_from_s3()

        path = Path(self.model_uri)

        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            raise FileNotFoundError(
                f"Model directory does not exist: {path}"
            )

        if not path.is_dir():
            raise NotADirectoryError(
                f"Model path is not a directory: {path}"
            )

        if not (path / "config.json").exists():
            raise FileNotFoundError(
                f"config.json was not found in model directory: {path}"
            )

        return str(path)

    def _download_from_s3(self) -> str:
        if not self.s3_access_key or not self.s3_secret_key:
            raise RuntimeError(
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required "
                "when MODEL_URI uses S3."
            )

        parts = self.model_uri[5:].split("/", 1)

        if len(parts) != 2:
            raise ValueError(
                "MODEL_URI must have the form "
                "s3://bucket/prefix"
            )

        bucket = parts[0]
        prefix = parts[1].rstrip("/") + "/"

        print(
            f"[S3] Downloading model from "
            f"s3://{bucket}/{prefix}"
        )

        self.local_model_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        s3 = boto3.client(
            "s3",
            endpoint_url=self.s3_endpoint,
            aws_access_key_id=self.s3_access_key,
            aws_secret_access_key=self.s3_secret_key,
            region_name="us-east-1",
        )

        paginator = s3.get_paginator("list_objects_v2")

        downloaded = 0

        for page in paginator.paginate(
            Bucket=bucket,
            Prefix=prefix
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]

                if key.endswith("/"):
                    continue

                relative_path = key[len(prefix):]

                if not relative_path:
                    continue

                destination = self.local_model_dir / relative_path
                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                print(
                    f"[S3] Downloading {relative_path}"
                )

                s3.download_file(
                    bucket,
                    key,
                    str(destination)
                )

                downloaded += 1

        if downloaded == 0:
            raise RuntimeError(
                f"No model files found at "
                f"s3://{bucket}/{prefix}"
            )

        config_file = self.local_model_dir / "config.json"

        if not config_file.exists():
            raise RuntimeError(
                "Model download completed but config.json "
                "was not found."
            )

        print(
            f"[S3] Model download complete. "
            f"Downloaded {downloaded} files."
        )

        return str(self.local_model_dir)

    def generate(self, command: str) -> dict:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded.")

        prompt = (
            "### Command:\n"
            f"{command}\n\n"
            "### Response:\n"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        print(f"[MODEL RAW OUTPUT] {generated_text}")

        return self._extract_json(generated_text)

    @staticmethod
    def _extract_json(text: str) -> dict:
        try:
            result = json.loads(text)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if not match:
            raise ValueError(
                f"Model did not return valid JSON: {text}"
            )

        json_text = match.group(0)

        try:
            result = json.loads(json_text)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON generated by model: {json_text}"
            ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                "Model output is not a JSON object."
            )

        return result

    @property
    def is_ready(self) -> bool:
        return (
            self.model is not None
            and self.tokenizer is not None
        )

    @property
    def served_by(self) -> str:
        return socket.gethostname()


model_loader = ModelLoader()