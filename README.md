
<h2 align="center">
  <img width="30%" alt="semble logo" src="https://raw.githubusercontent.com/MinishLab/semble/main/assets/images/semble_logo.png"><br/>
  Semble-GAR: Graph-Augmented Retrieval for Code Search<br/>
  <sub>Fork of <a href="https://github.com/MinishLab/semble">MinishLab/semble</a> — adds SQLite + tree-sitter relational code graph</sub>
</h2>

<div align="center">

[![Fork](https://img.shields.io/badge/fork-MinishLab%2Fsemble-blue)](https://github.com/MinishLab/semble)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/ferrolab-rs/semble-gar/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

<br/>

**Hybrid search (semantic + BM25) + relational code graph.**<br/>
**Understand not just *what* matches, but *how* the code connects.**

<br/>

[Quickstart](#quickstart) •
[What's new](#whats-new-vs-upstream) •
[Main Features](#main-features) •
[MCP Server](#mcp-server) •
[How it works](#how-it-works) •
[Benchmarks](#benchmarks)

</div>

> **Note:** this is a fork of [MinishLab/semble](https://github.com/MinishLab/semble) enhanced with a **Graph-Augmented Retrieval** layer. All original hybrid search features are preserved. The graph layer adds relational context (`called_by` / `depends_on`) to every search result and boosts structurally important code via graph centrality — with zero new dependencies.

## What's new vs upstream

| Feature | Original Semble | Semble-GAR |
|---|---|---|
| Search | Hybrid (semantic + BM25 + RRF) | Hybrid + graph centrality boost |
| Code graph | None | SQLite call + import graph (cross-file) |
| Symbols extracted | None | Functions, classes via tree-sitter |
| Imports resolved | None | `from X import Y` → scoped cross-file edges |
| Result context | Text only | JSON with `called_by` / `depends_on` |
| MCP output | Markdown blocks | JSON with relational metadata |
| Fallback | N/A | Silent vector-only on parse errors |
| New dependencies | — | **Zero** (sqlite3 stdlib, tree-sitter from Chonkie) |
| Upstream tests | 116 | 116 (all passing, no regression) |

## Quickstart

```bash
git clone https://github.com/ferrolab-rs/semble-gar.git
cd semble-gar
pip install -e "."        # Base install (CPU-only, zero extra deps)
pip install -e ".[mcp]"   # or with MCP server support
```

```python
from semble import SembleIndex

# Index a local directory
index = SembleIndex.from_path("./my-project")

# Index a remote git repository
index = SembleIndex.from_git("https://github.com/MinishLab/model2vec")

# Search the index with a natural-language or code query
results = index.search("save model to disk", top_k=3)

# Find code similar to a specific result
related = index.find_related(results[0], top_k=3)

# Each result exposes the matched chunk
result = results[0]
result.chunk.file_path   # "model2vec/model.py"
result.chunk.start_line  # 127
result.chunk.end_line    # 150
result.chunk.content     # "def save_pretrained(self, path: PathLike, ..."

# Get relational context: who calls this code, what it depends on
ctx = index.get_context_for_chunk(result.chunk)
ctx.called_by   # ["model2vec/hub.py:10-25", "model2vec/cli.py:40-55"]
ctx.depends_on  # ["model2vec/persistence.py:100-130"]
```

## Main Features

- **Fast**: indexes a repo in ~250 ms and answers queries in ~1.5 ms, all on CPU.
- **Accurate**: NDCG@10 of 0.854 on our [benchmarks](#benchmarks), on par with code-specialized transformer models, at a fraction of the size and cost.
- **Graph-Augmented Retrieval**: extracts the code's call graph and import tree via tree-sitter, stores it in a zero-dependency SQLite graph, and boosts structurally important chunks while enriching results with `called_by`/`depends_on` context.
- **Token-efficient**: returns only the relevant chunks, using ~98% fewer tokens than grep+read.
- **Zero setup**: runs on CPU with no API keys, GPU, or external services required.
- **MCP server**: drop-in tool for Claude Code, Cursor, Codex, OpenCode, and any other MCP-compatible agent.
- **Local and remote**: pass a local path or a git URL.

## MCP Server

Semble can run as an MCP server so agents can search any codebase directly. Repos are cloned and indexed on demand, and indexes are cached for the lifetime of the session.

### Setup

> Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and a local clone.

```bash
git clone https://github.com/ferrolab-rs/semble-gar.git
cd semble-gar
pip install -e ".[mcp]"
```

Then configure your agent to run `semble`:

#### Claude Code
```bash
claude mcp add semble -s user -- semble
```

#### Codex
Add to `~/.codex/config.toml`:
```toml
[mcp_servers.semble]
command = "semble"
```

#### OpenCode
Add to `~/.opencode/config.json`:
```json
{
  "mcp": {
    "semble": {
      "type": "local",
      "command": ["semble"]
    }
  }
}
```

#### Cursor
Add to `~/.cursor/mcp.json` (or `.cursor/mcp.json` in your project):
```json
{
  "mcpServers": {
    "semble": {
      "command": "semble"
    }
  }
}
```

### Tools

| Tool | Description |
|------|-------------|
| `search` | Search a codebase with a natural-language or code query. Pass `repo` as a git URL or local path. Results include relational context (`called_by`, `depends_on`). |
| `find_related` | Given a file path and line number, return chunks semantically similar to the code at that location, with relational metadata. |

Each result is returned as JSON with the following structure:

```json
{
  "file": "model2vec/model.py",
  "line": "127-150",
  "file_lines": 385,
  "code": "def save_pretrained(self, path: PathLike, ...",
  "score": 0.95,
  "source": "hybrid",
  "context": {
    "called_by": ["model2vec/hub.py:10-25"],
    "depends_on": ["model2vec/persistence.py:100-130"]
  }
}
```

### Sub-agent support

Claude Code and Codex CLI lazy-load MCP tool schemas, so sub-agents cannot call `mcp__semble__search` directly. The fix is to invoke semble through the [CLI](#cli) via Bash instead.

**Claude Code**: run this once in your project root:

```bash
semble init
```

This writes [`.claude/agents/semble-search.md`](src/semble/agents/semble-search.md).

**Other tools (Codex, etc.)**: append the following to your `AGENTS.md`:

```markdown
## Code Search

Use `semble search` to find code by describing what it does or naming a symbol/identifier, instead of grep:

​```bash
semble search "authentication flow" ./my-project
semble search "save_pretrained" ./my-project
semble search "save model to disk" ./my-project --top-k 10
​```

Use `semble find-related` to discover code similar to a known location (pass `file_path` and `line` from a prior search result):

​```bash
semble find-related src/auth.py 42 ./my-project
​```

`path` defaults to the current directory when omitted; git URLs are accepted.

## Workflow

1. Start with `semble search` to find relevant chunks.
2. Inspect full files only when the returned chunk is not enough context.
3. Optionally use `semble find-related` with a promising result's `file_path` and `line` to discover related implementations.
4. Use grep only when you need exhaustive literal matches or quick confirmation of an exact string.
```

## CLI

Semble also ships as a standalone CLI for use outside of MCP. This is useful in scripts, sub-agents, or anywhere you want search results without an MCP session.

```bash
# Search a local repo
semble search "authentication flow" ./my-project

# Search for a symbol or identifier
semble search "save_pretrained" ./my-project

# Search a remote repo (cloned on demand)
semble search "save model to disk" https://github.com/MinishLab/model2vec

# Find code similar to a known location (file_path and line from a prior search result)
semble find-related src/auth.py 42 ./my-project
```

`path` defaults to the current directory when omitted; git URLs are accepted.

## How it works

Semble splits each file into code-aware chunks using [Chonkie](https://github.com/chonkie-inc/chonkie), then builds a **code relationship graph** powered by tree-sitter and SQLite. The graph captures function/class definitions, call edges, and import dependencies across files — without adding any heavy dependencies (sqlite3 is stdlib, tree-sitter is a transitive dependency of Chonkie).

At query time, Semble scores every query against the chunks with two complementary retrievers: static [Model2Vec](https://github.com/MinishLab/model2vec) embeddings using the code-specialized [potion-code-16M](https://huggingface.co/minishlab/potion-code-16M) model for semantic similarity, and [BM25](https://github.com/xhluca/bm25s) for lexical matches on identifiers and API names. The two score lists are fused with Reciprocal Rank Fusion (RRF).

After fusing, chunks are boosted by their **graph centrality** (degree in the call+import graph), so well-connected "hub" functions rank higher. Results are then enriched with relational context (`called_by` / `depends_on`) so agents understand not just *what* matches, but *how* the code fits into the codebase structure. Finally, results are reranked with a set of code-aware signals:

<details>
<summary><b>Ranking signals</b></summary>

- **Graph centrality boost.** Chunks with high degree in the code relationship graph (many callers, many dependencies) are boosted — structurally important "hub" functions surface above isolated utilities.
- **Adaptive weighting.** Symbol-like queries (`Foo::bar`, `_private`, `getUserById`) get more lexical weight, while natural-language queries stay balanced between semantic and lexical retrievers.
- **Definition boosts.** A chunk that defines the queried symbol (a `class`, `def`, `func`, etc.) is ranked above chunks that merely reference it.
- **Identifier stems.** Query tokens are stemmed and matched against identifier stems in a chunk, giving an additional weight to chunks that contain them. For example, querying `parse config` boosts chunks containing `parseConfig`, `ConfigParser`, or `config_parser`.
- **File coherence.** When multiple chunks from the same file match the query, the file is boosted so the top result reflects broad file-level relevance rather than a single out-of-context chunk.
- **Noise penalties.** Test files, `compat/`/`legacy/` shims, example code, and `.d.ts` declaration stubs are down-ranked so canonical implementations surface first.

</details>

Because the embedding model is static with no transformer forward pass at query time, all of this runs in milliseconds on CPU.

## Benchmarks

### Upstream quality (original Semble)

These results are from the upstream [MinishLab/semble](https://github.com/MinishLab/semble) benchmark suite: ~1,250 queries over 63 repositories in 19 languages. Semble-GAR inherits the same hybrid search backbone and graph boosts are **additive** — they re-rank within the candidate pool.

![Speed vs quality](https://raw.githubusercontent.com/MinishLab/semble/main/assets/images/speed_vs_ndcg_cold.png)

| Method | NDCG@10 | Index time | Query p50 |
|--------|--------:|-----------:|----------:|
| CodeRankEmbed Hybrid | 0.862 | 57 s | 16 ms |
| **semble (upstream)** | **0.854** | **263 ms** | **1.5 ms** |
| CodeRankEmbed | 0.765 | 57 s | 16 ms |
| ColGREP | 0.693 | 5.8 s | 124 ms |
| BM25 | 0.673 | 263 ms | 0.02 ms |
| grepai | 0.561 | 35 s | 48 ms |
| probe | 0.387 | — | 207 ms |
| ripgrep | 0.126 | — | 12 ms |

Semble achieves 99% of the performance of the 137M-parameter [CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed) Hybrid, while indexing 218x faster and answering queries 11x faster. See [benchmarks](benchmarks/README.md) for per-language results, ablations, and methodology.

### Graph overhead (Semble-GAR)

Measured on 21 Python files (~2 500 lines) — the `src/semble` codebase itself:

| Metric | Original Semble | Semble-GAR | Overhead |
|--------|----------------:|-----------:|---------:|
| Index time (graph only) | — | 77 ms | +3.7 ms/file |
| Relational context query | — | **0.10 ms** | negligible |
| Symbols extracted | 0 | 127 | — |
| Edges (cross-file) | 0 | 222 (127) | — |
| Memory (graph DB) | 0 | in-memory SQLite | ~0 (no disk) |
| New dependencies | — | 0 | sqlite3 (stdlib) |

> **Key takeaway:** the graph layer adds ~4 ms per file during indexing and 0.1 ms per search query. Both are well within the 25% indexing slowdown and 50 ms query latency constraints.

### Graph boost ablation

Simulated on 1 000 queries using real graph centrality values from `src/semble` (21 files, 127 symbols, 222 edges). Each query is assigned synthetic RRF scores; the boost is applied and ranking changes are measured.

| Metric | Value |
|---|---|
| Hub promotions (centrality > 0.5) | **+18 053 positions net** |
| Isolated demotions (centrality = 0) | **0 positions** |
| Top-10 volatility | 4.1 changes/query (~60% stable) |
| Max single-chunk promotion | +16 ranks |

> The boost is **surgical**: it promotes structurally important hub functions without ever penalizing isolated chunks (their score is multiplied by 1.0). A full NDCG@10 ablation against the upstream benchmark suite is pending.

### Token efficiency

Agents using grep+read spend most of their context budget on irrelevant code. Semble returns only the chunks that match, keeping token usage low even at high recall.

![Token efficiency: recall vs. retrieved tokens](https://raw.githubusercontent.com/MinishLab/semble/main/assets/images/token_efficiency.png)

Semble uses **98% fewer tokens** on average, and reaches 94% recall at a budget of only 2k tokens, while grep+read needs a full 100k context window to reach 85%. See [benchmarks](benchmarks/README.md#token-efficiency) for details.

## License

MIT

## Citing

If you use Semble-GAR in your research, please cite the original Semble paper:

```bibtex
@software{minishlab2026semble,
  author       = {{van Dongen}, Thomas and Stephan Tulkens},
  title        = {Semble: Fast and Accurate Code Search for Agents},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19785932},
  url          = {https://github.com/MinishLab/semble},
  license      = {MIT}
}
```
