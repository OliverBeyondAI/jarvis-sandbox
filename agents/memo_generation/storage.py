"""
S3 Storage for the Memo Generation Agent.

Handles storing all pipeline outputs (research, synthesis, memo) in S3
or local filesystem. Produces an ArtifactBundle manifest linking everything.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import MemoConfig
from .models import ArtifactBundle


class MemoStorage:
    """
    Unified storage interface for the memo generation pipeline.

    Stores research reports, synthesis reports, memos, and a bundle
    manifest linking all artifacts together.
    """

    def __init__(self, config: MemoConfig | None = None) -> None:
        self.config = config or MemoConfig.from_env()
        self._s3_client: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def store_all_outputs(
        self,
        research_data: dict[str, Any],
        synthesis_data: dict[str, Any],
        memo_markdown: str,
        memo_html: str,
        memo_title: str = "",
    ) -> ArtifactBundle:
        """
        Store all pipeline outputs and produce an ArtifactBundle manifest.

        Args:
            research_data: Serialized ResearchReport from Agent 1.
            synthesis_data: Serialized SynthesisReport from Agent 2.
            memo_markdown: Rendered memo in Markdown format.
            memo_html: Rendered memo in HTML format.
            memo_title: Title of the memo.

        Returns:
            ArtifactBundle with paths to all stored artifacts.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_prefix = f"pipeline_run/{ts}"

        # Store all artifacts
        research_path = await self._store_json(
            f"{base_prefix}/research_report.json", research_data
        )
        synthesis_path = await self._store_json(
            f"{base_prefix}/synthesis_report.json", synthesis_data
        )
        memo_md_path = await self._store_text(
            f"{base_prefix}/memo.md", memo_markdown
        )
        memo_html_path = await self._store_text(
            f"{base_prefix}/memo.html", memo_html
        )

        # Build bundle manifest
        bundle = ArtifactBundle(
            bundle_id=f"run-{ts}",
            research_report_path=research_path,
            synthesis_report_path=synthesis_path,
            memo_path=memo_md_path,
            memo_html_path=memo_html_path,
            research_title=research_data.get("title", ""),
            synthesis_title=synthesis_data.get("title", ""),
            memo_title=memo_title,
            executive_summary=synthesis_data.get("executive_summary", ""),
            trends_analyzed=len(research_data.get("trends", [])),
            applications_identified=sum(
                len(ts_item.get("applications", []))
                for ts_item in synthesis_data.get("trend_syntheses", [])
            ),
            strategic_themes=len(synthesis_data.get("strategic_themes", [])),
        )

        # Store the manifest
        manifest_path = await self._store_json(
            f"{base_prefix}/manifest.json",
            bundle.model_dump(mode="json"),
        )
        bundle.metadata["manifest_path"] = manifest_path

        print(f"[storage] All artifacts saved under: {base_prefix}/", file=sys.stderr)
        return bundle

    async def store_memo(self, key: str, content: str) -> str:
        """Store a single memo file."""
        return await self._store_text(key, content)

    # ------------------------------------------------------------------
    # Internal: JSON storage
    # ------------------------------------------------------------------

    async def _store_json(self, key: str, data: dict[str, Any]) -> str:
        """Store a JSON document. Returns the storage path/URI."""
        full_key = self._resolve_key(key)
        payload = json.dumps(data, indent=2, default=str)

        if self.config.storage_local:
            return await self._store_local(full_key, payload)
        return await self._store_s3(full_key, payload, content_type="application/json")

    async def _store_text(self, key: str, text: str) -> str:
        """Store a plain text document. Returns the storage path/URI."""
        full_key = self._resolve_key(key)

        if self.config.storage_local:
            return await self._store_local(full_key, text)
        return await self._store_s3(full_key, text, content_type="text/plain")

    # ------------------------------------------------------------------
    # Local filesystem
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

    # ------------------------------------------------------------------
    # S3
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_key(self, key: str) -> str:
        """Prepend the configured prefix to a key."""
        prefix = self.config.s3_prefix.strip("/")
        key = key.lstrip("/")
        if prefix:
            return f"{prefix}/{key}"
        return key
