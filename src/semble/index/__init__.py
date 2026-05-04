"""Indexing pipeline: chunking, embedding, graph extraction, and search.

Orchestrated by ``create_index_from_path`` and exposed via ``SembleIndex``.
"""

from semble.index.index import SembleIndex

__all__ = ["SembleIndex"]
