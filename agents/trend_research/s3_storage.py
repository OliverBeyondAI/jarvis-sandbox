"""
S3 Storage Helper for the Trend Research system.

Supports both real S3 (via boto3) and local filesystem storage for
development and testing. All operations are async-safe.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config


class S3Storage:
    """
    Unified storage interface that writes to S3 or local filesystem.

    In local mode (default for development), files are written to a local
    directory mirroring the S3 key structure. In S3 mode, uses boto3.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self._s3_client: Any = None

    # ------------------------------------------------------------------
    # Public API (all async)
    # ------------------------------------------------------------------

    async def store_json(self, key: str, data: dict[str, Any]) -> str:
        """
        Store a JSON document. Returns the full path/URI where it was saved.

        Args:
            key: Storage key (e.g. "ai_ml/20260505_120000/report.json")
            data: Dictionary to serialize as JSON.
        """
        full_key = self._resolve_key(key)
        payload = json.dumps(data, indent=2, default=str)

        if self.config.storage_local:
            return await self._store_local(full_key, payload)
        return await self._store_s3(full_key, payload, content_type="application/json")

    async def store_text(self, key: str, text: str) -> str:
        """Store a plain text document. Returns the full path/URI."""
        full_key = self._resolve_key(key)

        if self.config.storage_local:
            return await self._store_local(full_key, text)
        return await self._store_s3(full_key, text, content_type="text/plain")

    async def load_json(self, key: str) -> dict[str, Any]:
        """Load a JSON document by key."""
        full_key = self._resolve_key(key)

        if self.config.storage_local:
            return await self._load_local_json(full_key)
        return await self._load_s3_json(full_key)

    async def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys under a prefix."""
        full_prefix = self._resolve_key(prefix)

        if self.config.storage_local:
            return await self._list_local(full_prefix)
        return await self._list_s3(full_prefix)

    async def exists(self, key: str) -> bool:
        """Check if a key exists in storage."""
        full_key = self._resolve_key(key)

        if self.config.storage_local:
            path = Path(self.config.local_storage_dir) / full_key
            return path.exists()
        return await self._exists_s3(full_key)

    # ------------------------------------------------------------------
    # Local filesystem implementation
    # ------------------------------------------------------------------

    async def _store_local(self, key: str, content: str) -> str:
        """Write content to local filesystem."""
        def _write() -> str:
            path = Path(self.config.local_storage_dir) / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"[storage] Saved locally: {path}", file=sys.stderr)
            return str(path.resolve())

        return await asyncio.to_thread(_write)

    async def _load_local_json(self, key: str) -> dict[str, Any]:
        """Read JSON from local filesystem."""
        def _read() -> dict[str, Any]:
            path = Path(self.config.local_storage_dir) / key
            return json.loads(path.read_text(encoding="utf-8"))

        return await asyncio.to_thread(_read)

    async def _list_local(self, prefix: str) -> list[str]:
        """List files under a local directory prefix."""
        def _scan() -> list[str]:
            base = Path(self.config.local_storage_dir)
            prefix_path = base / prefix if prefix else base
            if not prefix_path.exists():
                return []
            return sorted(
                str(p.relative_to(base))
                for p in prefix_path.rglob("*")
                if p.is_file()
            )

        return await asyncio.to_thread(_scan)

    # ------------------------------------------------------------------
    # S3 implementation
    # ------------------------------------------------------------------

    def _get_s3_client(self) -> Any:
        """Lazy-initialize the boto3 S3 client."""
        if self._s3_client is None:
            import boto3

            self._s3_client = boto3.client("s3", region_name=self.config.s3_region)
        return self._s3_client

    async def _store_s3(self, key: str, content: str, content_type: str) -> str:
        """Upload content to S3."""
        def _upload() -> str:
            client = self._get_s3_client()
            client.put_object(
                Bucket=self.config.s3_bucket,
                Key=key,
                Body=content.encode("utf-8"),
                ContentType=content_type,
            )
            uri = f"s3://{self.config.s3_bucket}/{key}"
            print(f"[storage] Uploaded to S3: {uri}", file=sys.stderr)
            return uri

        return await asyncio.to_thread(_upload)

    async def _load_s3_json(self, key: str) -> dict[str, Any]:
        """Download and parse JSON from S3."""
        def _download() -> dict[str, Any]:
            client = self._get_s3_client()
            response = client.get_object(Bucket=self.config.s3_bucket, Key=key)
            body = response["Body"].read().decode("utf-8")
            return json.loads(body)

        return await asyncio.to_thread(_download)

    async def _list_s3(self, prefix: str) -> list[str]:
        """List objects in S3 under a prefix."""
        def _list_objects() -> list[str]:
            client = self._get_s3_client()
            paginator = client.get_paginator("list_objects_v2")
            keys: list[str] = []
            for page in paginator.paginate(Bucket=self.config.s3_bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            return sorted(keys)

        return await asyncio.to_thread(_list_objects)

    async def _exists_s3(self, key: str) -> bool:
        """Check if an object exists in S3."""
        def _head() -> bool:
            client = self._get_s3_client()
            try:
                client.head_object(Bucket=self.config.s3_bucket, Key=key)
                return True
            except client.exceptions.ClientError:
                return False

        return await asyncio.to_thread(_head)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_key(self, key: str) -> str:
        """Prepend the configured S3 prefix to a key."""
        prefix = self.config.s3_prefix.strip("/")
        key = key.lstrip("/")
        if prefix:
            return f"{prefix}/{key}"
        return key
