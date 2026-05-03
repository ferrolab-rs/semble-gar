"""Ablation benchmark: compare Semble-GAR vs baseline hybrid search.

Runs the full upstream benchmark suite twice: once with graph-augmented
ranking (default, _GRAPH_BOOST=0.4) and once with baseline hybrid search
(_GRAPH_BOOST=0.0).  Reports the NDCG@10 delta for every repo and the
global average.

Usage:
    python benchmarks/ablation.py [--repo <name>] [--language <lang>]
"""
import argparse
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
from model2vec import StaticModel

from benchmarks.data import (
    RepoSpec,
    Task,
    add_filter_args,
    grouped_tasks,
    load_filtered_tasks,
    save_results,
)
from benchmarks.metrics import ndcg_at_k, target_rank
from semble import SembleIndex
from semble.index.dense import _DEFAULT_MODEL_NAME
from semble.types import SearchResult

_LATENCY_RUNS = 3  # fewer runs for ablation (we run everything twice)
_DIRECT_TOP_K = 10


@dataclass(frozen=True)
class RepoResult:
    repo: str
    language: str
    chunks: int
    ndcg5: float
    ndcg10: float
    p50_ms: float
    index_ms: float
    graph_boost: float


def _evaluate(
    index: SembleIndex, tasks: list[Task], *, verbose: bool = False
) -> tuple[float, float, list[float], dict[str, float]]:
    ndcg5_sum = 0.0
    ndcg10_sum = 0.0
    latencies: list[float] = []
    category_ndcg10: dict[str, list[float]] = defaultdict(list)

    for task in tasks:
        query_latencies: list[float] = []
        results: list[SearchResult] = []
        for _ in range(_LATENCY_RUNS):
            started = time.perf_counter()
            results = index.search(task.query, top_k=_DIRECT_TOP_K)
            query_latencies.append((time.perf_counter() - started) * 1000)
        latencies.append(float(np.median(query_latencies)))

        relevant_ranks = [rank for t in task.all_relevant if (rank := target_rank(results, t)) is not None]
        n_relevant = len(task.all_relevant)
        q_ndcg5 = ndcg_at_k(relevant_ranks, n_relevant, 5)
        q_ndcg10 = ndcg_at_k(relevant_ranks, n_relevant, _DIRECT_TOP_K)
        ndcg5_sum += q_ndcg5
        ndcg10_sum += q_ndcg10
        category_ndcg10[task.category or "unknown"].append(q_ndcg10)

    total = len(tasks)
    by_category = {cat: sum(vals) / len(vals) for cat, vals in sorted(category_ndcg10.items())}
    return ndcg5_sum / total, ndcg10_sum / total, latencies, by_category


def _run_ablation(
    repo_tasks: dict[str, list[Task]],
    model: StaticModel,
    specs: dict[str, RepoSpec],
    *,
    graph_boost: float,
    verbose: bool = False,
) -> list[RepoResult]:
    """Run benchmark with the given graph boost value."""
    import semble.search
    old_boost = semble.search._GRAPH_BOOST
    semble.search._GRAPH_BOOST = graph_boost

    label = f"GAR boost={graph_boost}" if graph_boost > 0 else "Baseline (no graph)"
    print(f"\n{'=' * 80}", file=sys.stderr)
    print(f"  {label}", file=sys.stderr)
    print(f"{'=' * 80}", file=sys.stderr)
    print(
        f"{'Repo':<12} {'Lang':<8} {'Chunks':>6} {'idx':>8} {'NDCG@5':>8} {'NDCG@10':>8} {'p50':>8}",
        file=sys.stderr,
    )
    print(f"{'-' * 12} {'-' * 8} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}", file=sys.stderr)

    results: list[RepoResult] = []
    try:
        for repo, tasks in sorted(repo_tasks.items()):
            spec = specs[repo]
            started = time.perf_counter()
            index = SembleIndex.from_path(spec.benchmark_dir, model=model)
            index_ms = (time.perf_counter() - started) * 1000
            ndcg5, ndcg10, latencies, _by_category = _evaluate(index, tasks, verbose=verbose)
            p50 = float(np.percentile(latencies, 50))

            results.append(RepoResult(
                repo=repo,
                language=spec.language,
                chunks=len(index.chunks),
                ndcg5=ndcg5,
                ndcg10=ndcg10,
                p50_ms=p50,
                index_ms=index_ms,
                graph_boost=graph_boost,
            ))
            print(
                f"{repo:<12} {spec.language:<8} {len(index.chunks):>6} "
                f"{index_ms:>7.0f}ms {ndcg5:>8.3f} {ndcg10:>8.3f} {p50:>7.2f}ms",
                file=sys.stderr,
            )
    finally:
        semble.search._GRAPH_BOOST = old_boost

    return results


def _print_comparison(gar_results: list[RepoResult], baseline_results: list[RepoResult]) -> None:
    """Print side-by-side comparison and per-repo delta."""
    gar_by_repo = {r.repo: r for r in gar_results}
    baseline_by_repo = {r.repo: r for r in baseline_results}

    common_repos = sorted(set(gar_by_repo) & set(baseline_by_repo))
    if not common_repos:
        print("\nNo repos to compare.", file=sys.stderr)
        return

    print(f"\n{'=' * 100}", file=sys.stderr)
    print("  ABLATION: Semble-GAR vs Baseline (no graph boost)", file=sys.stderr)
    print(f"{'=' * 100}", file=sys.stderr)
    print(
        f"{'Repo':<15} {'Lang':<8} {'Baseline NDCG@10':>16} {'GAR NDCG@10':>14} {'Delta':>8} {'p50 Δ':>8}",
        file=sys.stderr,
    )
    print(f"{'-' * 15} {'-' * 8} {'-' * 16} {'-' * 14} {'-' * 8} {'-' * 8}", file=sys.stderr)

    ndcg_deltas: list[float] = []
    p50_deltas: list[float] = []
    for repo in common_repos:
        g = gar_by_repo[repo]
        b = baseline_by_repo[repo]
        d_ndcg = g.ndcg10 - b.ndcg10
        d_p50 = g.p50_ms - b.p50_ms
        ndcg_deltas.append(d_ndcg)
        p50_deltas.append(d_p50)
        sign = "+" if d_ndcg > 0 else " " if d_ndcg == 0 else ""
        print(
            f"{repo:<15} {g.language:<8} {b.ndcg10:>16.4f} {g.ndcg10:>14.4f} {sign}{d_ndcg:>7.4f} {d_p50:>7.2f}ms",
            file=sys.stderr,
        )

    gar_avg = sum(r.ndcg10 for r in gar_results) / len(gar_results)
    bas_avg = sum(r.ndcg10 for r in baseline_results) / len(baseline_results)
    gar_p50 = sum(r.p50_ms for r in gar_results) / len(gar_results)
    bas_p50 = sum(r.p50_ms for r in baseline_results) / len(baseline_results)
    gar_idx = sum(r.index_ms for r in gar_results) / len(gar_results)
    bas_idx = sum(r.index_ms for r in baseline_results) / len(baseline_results)

    print(f"{'-' * 15} {'-' * 8} {'-' * 16} {'-' * 14} {'-' * 8} {'-' * 8}", file=sys.stderr)
    print(
        f"{'AVERAGE':<15} {'':<8} {bas_avg:>16.4f} {gar_avg:>14.4f} {gar_avg - bas_avg:>+8.4f} {gar_p50 - bas_p50:>7.2f}ms",
        file=sys.stderr,
    )

    # Summary
    wins = sum(1 for d in ndcg_deltas if d > 0.001)
    ties = sum(1 for d in ndcg_deltas if abs(d) <= 0.001)
    losses = sum(1 for d in ndcg_deltas if d < -0.001)
    print(f"\n  Wins: {wins}  Ties: {ties}  Losses: {losses}  (out of {len(common_repos)} repos)", file=sys.stderr)
    if ndcg_deltas:
        print(f"  Mean NDCG delta: {sum(ndcg_deltas)/len(ndcg_deltas):+.4f}", file=sys.stderr)
        print(f"  Max gain: {max(ndcg_deltas):+.4f}  Max loss: {min(ndcg_deltas):+.4f}", file=sys.stderr)
    print(f"\n  Index time: {bas_idx:.0f}ms → {gar_idx:.0f}ms (Δ={gar_idx - bas_idx:+.0f}ms)", file=sys.stderr)
    print(f"  Query p50:  {bas_p50:.2f}ms → {gar_p50:.2f}ms (Δ={gar_p50 - bas_p50:+.2f}ms)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: compare Semble-GAR vs baseline hybrid search."
    )
    add_filter_args(parser, verbose=True)
    parser.add_argument(
        "--boost",
        type=float,
        default=0.4,
        help="Graph boost value for the GAR run (default: 0.4).",
    )
    args = parser.parse_args()

    repo_specs, tasks = load_filtered_tasks(args.repo or None, args.language or None)
    repo_tasks = grouped_tasks(tasks)

    print(f"Loading model...", file=sys.stderr)
    started = time.perf_counter()
    model = StaticModel.from_pretrained(_DEFAULT_MODEL_NAME)
    print(f"Loaded in {(time.perf_counter() - started) * 1000:.0f} ms", file=sys.stderr)
    print(f"Repos: {len(repo_tasks)}  Tasks: {len(tasks)}", file=sys.stderr)

    # Run baseline (no graph boost)
    baseline = _run_ablation(repo_tasks, model, repo_specs, graph_boost=0.0, verbose=args.verbose)

    # Run GAR (graph boost)
    gar = _run_ablation(repo_tasks, model, repo_specs, graph_boost=args.boost, verbose=args.verbose)

    _print_comparison(gar, baseline)


if __name__ == "__main__":
    main()
