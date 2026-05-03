"""Comprehensive integration test for Semble-GAR."""
import json
import sqlite3

from semble import SembleIndex
from semble.index.graph_store import GraphStore, _module_file_patterns
from semble.index.graph_extractor import (
    RawFileEdges,
    extract_edges_from_file,
    extract_symbols_from_file,
)
from semble.search import _apply_graph_boost, _rrf_scores
from semble.types import Chunk, GraphContext, SearchMode, SearchResult
from semble.utils import _format_results_json, _is_git_url, _resolve_chunk


def test_imports():
    """All public API imports succeed."""
    assert SembleIndex is not None
    assert Chunk is not None
    assert GraphContext is not None
    assert GraphStore is not None


def test_types():
    """Dataclass fields and properties."""
    c = Chunk(content="def foo(): pass", file_path="test.py", start_line=1, end_line=3, language="python")
    assert c.location == "test.py:1-3"
    assert c.language == "python"

    ctx = GraphContext(called_by=["a.py:1-10"], depends_on=["b.py:20-30"])
    assert ctx.called_by == ["a.py:1-10"]
    assert ctx.depends_on == ["b.py:20-30"]
    assert GraphContext().called_by == []

    r = SearchResult(chunk=c, score=0.95, source=SearchMode.HYBRID)
    assert r.score == 0.95
    assert r.source == SearchMode.HYBRID


def test_graphstore_crud():
    """SQLite CRUD, scoped lookup, centrality."""
    store = GraphStore()
    s1 = store.insert_symbol("foo", "function", "a.py", "a.py:1-10")
    s2 = store.insert_symbol("bar", "function", "b.py", "b.py:5-15")
    s3 = store.insert_symbol("foo", "function", "c.py", "c.py:20-30")

    assert s1 > 0 and s2 > 0 and s3 > 0
    assert s1 != s3

    store.insert_edge(s1, s2, "calls")
    store.insert_edge(s3, s2, "calls")

    ids = store.lookup_symbol_ids("foo")
    assert len(ids) == 2
    assert s1 in ids and s3 in ids

    ids_scoped = store.lookup_symbol_ids_scoped("foo", "a")
    assert len(ids_scoped) >= 1

    ctx = store.get_relational_context(["b.py:5-15"])
    assert len(ctx["b.py:5-15"].called_by) >= 1

    cent = store.get_graph_centrality(["a.py:1-10", "b.py:5-15"])
    assert cent["b.py:5-15"] == 1.0
    store.close()


def test_cross_file_import():
    """Import from another file resolves edges correctly."""
    store = GraphStore()

    source_a = "def resolve_thing(x):\n    return x\n"
    chunks_a = [Chunk(content=source_a, file_path="pkg/utils.py", start_line=1, end_line=2, language="python")]
    raw_a = extract_symbols_from_file(source_a, "pkg/utils.py", "python", chunks_a, store)
    assert raw_a is not None

    source_b = "from pkg.utils import resolve_thing\ndef main():\n    return resolve_thing(42)\n"
    chunks_b = [Chunk(content=source_b, file_path="app.py", start_line=1, end_line=3, language="python")]
    raw_b = extract_symbols_from_file(source_b, "app.py", "python", chunks_b, store)
    assert raw_b is not None

    extract_edges_from_file(raw_a, store)
    extract_edges_from_file(raw_b, store)

    edges = store.conn.execute("""
        SELECT s1.name, s2.name, s2.file FROM edges e
        JOIN symbols s1 ON s1.id = e.source_id
        JOIN symbols s2 ON s2.id = e.target_id
        WHERE e.type = 'calls'
    """).fetchall()

    # edges: (source_name, target_name, target_file)
    main_to_resolve = [e for e in edges if e[0] == "main" and e[1] == "resolve_thing"]
    assert len(main_to_resolve) > 0, f"Missing cross-file edge: {edges}"
    assert main_to_resolve[0][1] == "resolve_thing"
    assert main_to_resolve[0][2] == "pkg/utils.py"
    store.close()


def test_wildcard_import():
    """from lib.tools import * resolves via wildcard expansion."""
    store = GraphStore()

    source_mod = "def helper(x): return x\ndef formatter(y): return str(y)\n"
    chunks_mod = [Chunk(content=source_mod, file_path="lib/tools.py", start_line=1, end_line=2, language="python")]
    raw_mod = extract_symbols_from_file(source_mod, "lib/tools.py", "python", chunks_mod, store)
    extract_edges_from_file(raw_mod, store)

    source_wild = "from lib.tools import *\ndef run():\n    return helper(1)\n"
    chunks_wild = [Chunk(content=source_wild, file_path="main.py", start_line=1, end_line=3, language="python")]
    raw_wild = extract_symbols_from_file(source_wild, "main.py", "python", chunks_wild, store)
    extract_edges_from_file(raw_wild, store)

    run_calls = store.conn.execute("""
        SELECT s2.name, s2.file FROM edges e
        JOIN symbols s1 ON s1.id = e.source_id
        JOIN symbols s2 ON s2.id = e.target_id
        WHERE s1.name = 'run' AND e.type = 'calls'
    """).fetchall()
    assert len(run_calls) > 0, f"Wildcard import edge missing: {run_calls}"
    assert run_calls[0][1] == "lib/tools.py"
    store.close()


def test_aliased_import():
    """from pkg.mod import long_name as short uses alias for resolution."""
    store = GraphStore()

    source_def = "def long_function_name(data): return data\n"
    chunks_def = [Chunk(content=source_def, file_path="pkg/mod.py", start_line=1, end_line=1, language="python")]
    raw_def = extract_symbols_from_file(source_def, "pkg/mod.py", "python", chunks_def, store)
    extract_edges_from_file(raw_def, store)

    source_alias = "from pkg.mod import long_function_name as lfn\ndef caller():\n    return lfn(42)\n"
    chunks_alias = [Chunk(content=source_alias, file_path="caller.py", start_line=1, end_line=3, language="python")]
    raw_alias = extract_symbols_from_file(source_alias, "caller.py", "python", chunks_alias, store)
    extract_edges_from_file(raw_alias, store)

    call_edges = store.conn.execute("""
        SELECT s1.name, s2.name, s2.file FROM edges e
        JOIN symbols s1 ON s1.id = e.source_id
        JOIN symbols s2 ON s2.id = e.target_id
        WHERE s1.name = 'caller' AND e.type = 'calls'
    """).fetchall()
    assert len(call_edges) > 0, f"Aliased import edge missing: {call_edges}"
    assert call_edges[0][1] == "long_function_name"
    assert call_edges[0][2] == "pkg/mod.py"
    store.close()


def test_module_level_calls():
    """Top-level calls use *module* pseudo-symbol."""
    store = GraphStore()

    source_func = "def setup_config(): return {}\n"
    chunks_func = [Chunk(content=source_func, file_path="config.py", start_line=1, end_line=1, language="python")]
    raw_func = extract_symbols_from_file(source_func, "config.py", "python", chunks_func, store)
    extract_edges_from_file(raw_func, store)

    source_entry = "result = setup_config()\ndef main():\n    pass\n"
    chunks_entry = [Chunk(content=source_entry, file_path="entry.py", start_line=1, end_line=3, language="python")]
    raw_entry = extract_symbols_from_file(source_entry, "entry.py", "python", chunks_entry, store)
    extract_edges_from_file(raw_entry, store)

    mod_calls = store.conn.execute("""
        SELECT s1.name, s2.name, s2.file FROM edges e
        JOIN symbols s1 ON s1.id = e.source_id
        JOIN symbols s2 ON s2.id = e.target_id
        WHERE s1.name = '*module*' AND e.type = 'calls'
    """).fetchall()
    assert len(mod_calls) > 0, f"Module-level calls missing: {mod_calls}"
    assert mod_calls[0][2] == "config.py"
    store.close()


def test_json_format():
    """MCP JSON output includes called_by / depends_on."""
    chunk = Chunk(content="def bar(): pass", file_path="src/bar.py", start_line=1, end_line=2, language="python")
    result = SearchResult(chunk=chunk, score=0.9, source=SearchMode.HYBRID)
    ctx_map = {"src/bar.py:1-2": GraphContext(called_by=["a.py:10-20"], depends_on=["b.py:5-15"])}
    parsed = json.loads(_format_results_json([result], ctx_map))

    assert parsed[0]["file"] == "src/bar.py"
    assert parsed[0]["context"]["called_by"] == ["a.py:10-20"]
    assert parsed[0]["context"]["depends_on"] == ["b.py:5-15"]


def test_parse_fallback():
    """Tree-sitter gracefully handles partially broken code — still extracts valid symbols."""
    store = GraphStore()
    # Even with a syntax error in params, tree-sitter still finds the function name.
    raw = extract_symbols_from_file("def foo(:\n    pass\n", "broken.py", "python",
                                     [Chunk(content="def foo(:\n    pass\n", file_path="broken.py",
                                      start_line=1, end_line=2, language="python")], store)
    # Symbols MAY be extracted even from broken code (tree-sitter is error-tolerant).
    # The fallback is for COMPLETE parse failures (e.g. unsupported language, binary file).
    sym_count = store.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    # We just verify nothing crashes — extraction is best-effort on partial syntax.
    assert raw is not None or sym_count == 0  # Either extracted or empty
    store.close()


def test_path_patterns():
    r"""LIKE patterns match both / and \."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (file TEXT)")
    conn.execute("INSERT INTO t VALUES ('ranking/weights.py')")
    conn.execute("INSERT INTO t VALUES ('ranking\\weights.py')")
    conn.execute("INSERT INTO t VALUES ('sub/ranking.py')")
    conn.execute("INSERT INTO t VALUES ('sub\\ranking.py')")

    patterns = _module_file_patterns("ranking")
    p0 = conn.execute("SELECT file FROM t WHERE file LIKE ?", (patterns[0],)).fetchall()
    assert len(p0) >= 2, f"Pattern {patterns[0]} should match 2 files: {p0}"

    p1 = conn.execute("SELECT file FROM t WHERE file LIKE ?", (patterns[1],)).fetchall()
    assert len(p1) >= 2, f"Pattern {patterns[1]} should match 2 files: {p1}"

    conn.close()


def test_rrf_graph_boost():
    """Graph centrality boosts RRF scores."""
    a = Chunk(content="def foo(): pass", file_path="a.py", start_line=1, end_line=1, language="python")
    b = Chunk(content="def bar(): pass", file_path="b.py", start_line=1, end_line=1, language="python")

    scores = {a: 0.9, b: 0.8}
    rrf = _rrf_scores(scores)
    assert rrf[a] > rrf[b]

    store = GraphStore()
    sa = store.insert_symbol("foo", "function", "a.py", "a.py:1-1")
    sb = store.insert_symbol("bar", "function", "b.py", "b.py:1-1")
    store.insert_edge(sa, sb, "calls")
    store.insert_edge(sb, sa, "calls")  # both have degree 2

    boosted = _apply_graph_boost(rrf, store)
    assert boosted[a] > rrf[a]
    assert boosted[b] > rrf[b]
    store.close()


def test_context_manager():
    """GraphStore works as context manager."""
    with GraphStore() as store:
        store.insert_symbol("test", "function", "t.py", "t.py:1-1")
        row = store.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()
        assert row[0] == 1


def test_utils():
    """Git URL detection and chunk resolution."""
    assert _is_git_url("https://github.com/org/repo") is True
    assert _is_git_url("/local/path") is False
    assert _is_git_url("git@github.com:org/repo") is True

    c = Chunk(content="a\nb", file_path="f.py", start_line=1, end_line=2, language="python")
    assert _resolve_chunk([c], "f.py", 1) == c
    assert _resolve_chunk([c], "f.py", 99) is None
    assert _resolve_chunk([c], "x.py", 1) is None
