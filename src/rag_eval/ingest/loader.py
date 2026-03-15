"""Download and iterate over HuggingFace datasets.

Supports flexible field mapping so different legal datasets can be
normalized into a common schema::

    {"id": ..., "text": ..., "metadata": {"title": ..., "source": ..., ...}}

For datasets with only a ``text`` column (e.g. santoshtyss/us-court-cases),
the loader auto-generates IDs and extracts metadata heuristically from the
first lines of the text.

**Streaming mode** (``--streaming``): for very large datasets (e.g. 35 GB
us-court-cases) the loader can stream rows on-the-fly without downloading
the entire dataset first.  This is much faster when you only need a subset
via ``--max-rows``.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Generator

from datasets import load_dataset, DatasetDict, Dataset, IterableDataset

from rag_eval.config import Settings

log = logging.getLogger(__name__)


def _extract_title_from_text(text: str, max_len: int = 120) -> str:
    """Heuristic: use the first non-empty line as a title."""
    for line in text.split("\n"):
        line = line.strip()
        if line and len(line) > 5:
            return line[:max_len]
    return ""


def load_hf_dataset(
    dataset_name: str,
    *,
    split: str = "train",
    text_field: str = "context",
    id_field: str | None = "id",
    title_field: str | None = "title",
    extra_meta_fields: list[str] | None = None,
    subset: str | None = None,
    max_rows: int | None = None,
    streaming: bool = False,
    settings: Settings | None = None,
) -> Generator[dict, None, None]:
    """Yield ``{"id": ..., "text": ..., "metadata": {...}}`` dicts from a HF dataset.

    Parameters
    ----------
    dataset_name:
        HuggingFace dataset identifier, e.g. ``"HFforLegal/case-law"``.
    split:
        Which split to load.  For HFforLegal/case-law, use country codes
        like ``"us"``, ``"gb"``, ``"fr"``.
    text_field:
        Column name that contains the main text to index.
    id_field:
        Column used as a unique document id.  Falls back to content hash.
    title_field:
        Optional column used in metadata.
    extra_meta_fields:
        Additional columns to include in metadata (e.g.
        ``["citation", "docket_number", "state", "issuer"]``).
    subset:
        Dataset configuration/subset name if applicable.
    max_rows:
        Cap the number of rows yielded (useful for dev/test).
    streaming:
        If True, stream rows on-the-fly without downloading the full dataset.
        Highly recommended for large datasets when using ``max_rows``.
    settings:
        Platform settings (used for cache dir).
    """
    settings = settings or Settings()

    use_streaming = streaming or (max_rows is not None and max_rows > 0)
    if use_streaming:
        log.info("Loading dataset %s (split=%s, subset=%s) in STREAMING mode",
                 dataset_name, split, subset)
    else:
        log.info("Loading dataset %s (split=%s, subset=%s)",
                 dataset_name, split, subset)

    kwargs: dict = {
        "path": dataset_name,
        "split": split,
        "streaming": use_streaming,
    }
    if subset:
        kwargs["name"] = subset
    if settings.hf_cache_dir and not use_streaming:
        kwargs["cache_dir"] = settings.hf_cache_dir

    ds = load_dataset(**kwargs)
    if isinstance(ds, DatasetDict):
        ds = ds[split]

    # For non-streaming datasets, validate columns up front
    columns: set[str] | None = None
    if isinstance(ds, Dataset):
        columns = set(ds.column_names)
        if text_field not in columns:
            raise ValueError(
                f"text_field={text_field!r} not found. Available: {sorted(columns)}"
            )

    count = 0
    seen_hashes: set[str] = set()
    source_short = dataset_name.split("/")[-1]

    for idx, row in enumerate(ds):
        # For streaming datasets, discover columns from first row
        if columns is None:
            columns = set(row.keys())
            if text_field not in columns:
                raise ValueError(
                    f"text_field={text_field!r} not found. Available: {sorted(columns)}"
                )

        text = str(row.get(text_field, "")).strip()
        if not text:
            continue

        content_hash = hashlib.sha256(text.encode()).hexdigest()[:20]
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        # ID: prefer the dataset's own id field, fall back to content hash
        if id_field and id_field in columns and row.get(id_field):
            doc_id = f"{source_short}_{row[id_field]}"
        else:
            doc_id = f"{source_short}_{content_hash}"

        # Metadata
        meta: dict = {"source": dataset_name}
        if title_field and title_field in columns:
            meta["title"] = str(row.get(title_field, ""))
        elif title_field:
            meta["title"] = _extract_title_from_text(text)

        for field in (extra_meta_fields or []):
            if field in columns:
                val = row.get(field)
                if val:
                    meta[field] = str(val)

        yield {"id": doc_id, "text": text, "metadata": meta}
        count += 1
        if max_rows and count >= max_rows:
            break

    log.info("Yielded %d unique documents from %s", count, dataset_name)
