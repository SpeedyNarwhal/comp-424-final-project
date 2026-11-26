"""Analyze the search stats captured in the agent's log files."""

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


TESTING_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TESTING_DIR / "results"
AGENT_LOGS_DIR = TESTING_DIR / "agent_logs"
STEP_TIME_LIMIT = 2.0
SEARCH_LOG_PREFIX = "SEARCH_STATS "

def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile() requires at least one value")

    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def write_csv(filepath, rows, fieldnames):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Search-log analysis

def infer_target_agent(result_filepath):
    """Infer the agent being tested from '<agent>_vs_<opponent>_seed<N>'."""
    stem = result_filepath.stem
    if "_vs_" not in stem:
        return None
    return stem.split("_vs_", 1)[0]


def load_search_logs(result_filepath):
    """Load SEARCH_STATS records from the copied logs for one experiment."""
    log_paths = sorted(AGENT_LOGS_DIR.glob(
        f"{result_filepath.stem}_game*_seat*.txt"
    ))

    search_records = []
    malformed_lines = 0

    for log_path in log_paths:
        with log_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line.startswith(SEARCH_LOG_PREFIX):
                    continue

                payload = line[len(SEARCH_LOG_PREFIX):]
                try:
                    record = json.loads(payload)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    print(
                        f"Warning: malformed SEARCH_STATS JSON in "
                        f"{log_path.name}, line {line_number}."
                    )
                    continue

                # Keep the source log so decisions can be traced to a game.
                record["_log_file"] = log_path.name
                search_records.append(record)

    return search_records, len(log_paths), malformed_lines


def numeric_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values

def mean_or_none(values, digits=4):
    return round(statistics.mean(values), digits) if values else None


def median_or_none(values, digits=4):
    return round(statistics.median(values), digits) if values else None


def p95_or_none(values, digits=4):
    return round(percentile(values, 0.95), digits) if values else None


def analyze_search_records(search_records):
    """Flatten per-decision SEARCH_STATS records into decision and iteration data."""
    decision_rows = []
    iteration_rows = []

    for record in search_records:
        iterations = record.get("iterations", [])
        decision_nodes = sum(int(item.get("nodes", 0) or 0) for item in iterations)
        decision_cutoffs = sum(
            int(item.get("alpha_beta_cutoffs", 0) or 0) for item in iterations
        )
        decision_pruned = sum(
            int(item.get("children_pruned", 0) or 0) for item in iterations
        )
        chosen_depth = record.get("chosen_move_depth")
        completed_depth = int(record.get("completed_depth", 0) or 0)

        decision_rows.append({
            "log_file": record.get("_log_file", ""),
            "agent_turn": record.get("agent_turn"),
            "player": record.get("player"),
            "root_ply": record.get("root_ply"),
            "completed_depth": completed_depth,
            "attempted_depth": record.get("attempted_depth"),
            "chosen_move_depth": chosen_depth,
            "chosen_from_partial_depth": bool(
                chosen_depth is not None and chosen_depth > completed_depth
            ),
            "timed_out": bool(record.get("timed_out", False)),
            "elapsed_s": record.get("elapsed_s"),
            "nodes_across_iterations": decision_nodes,
            "cutoffs_across_iterations": decision_cutoffs,
            "children_pruned_across_iterations": decision_pruned,
            "iteration_count": len(iterations),
            "used_random_fallback": bool(record.get("used_random_fallback", False)),
            "chosen_move": record.get("chosen_move"),
        })

        for iteration in iterations:
            depth = iteration.get("depth")
            root_legal = int(iteration.get("root_legal_moves", 0) or 0)
            root_considered = int(iteration.get("root_moves_considered", 0) or 0)
            root_coverage = (
                root_considered / root_legal if root_legal > 0 else None
            )

            iteration_rows.append({
                "log_file": record.get("_log_file", ""),
                "agent_turn": record.get("agent_turn"),
                "player": record.get("player"),
                "depth": depth,
                "completed": bool(iteration.get("completed", False)),
                "timed_out": bool(iteration.get("timed_out", False)),
                "elapsed_s": iteration.get("elapsed_s"),
                "nodes": iteration.get("nodes", 0),
                "expanded_nodes": iteration.get("expanded_nodes", 0),
                "evaluation_leaves": iteration.get("evaluation_leaves", 0),
                "depth_limit_leaves": iteration.get("depth_limit_leaves", 0),
                "terminal_nodes": iteration.get("terminal_nodes", 0),
                "legal_moves_generated": iteration.get("legal_moves_generated", 0),
                "average_branching_factor": iteration.get("average_branching_factor"),
                "max_branching_factor": iteration.get("max_branching_factor", 0),
                "alpha_beta_cutoffs": iteration.get("alpha_beta_cutoffs", 0),
                "children_pruned": iteration.get("children_pruned", 0),
                "pruned_child_fraction": iteration.get("pruned_child_fraction"),
                "average_cutoff_move_index": iteration.get("average_cutoff_move_index"),
                "root_legal_moves": root_legal,
                "root_moves_started": iteration.get("root_moves_started", 0),
                "root_moves_considered": root_considered,
                "root_coverage_fraction": root_coverage,
                "immediate_win_found": bool(iteration.get("immediate_win_found", False)),
                "best_move": iteration.get("best_move"),
                "best_value": iteration.get("best_value"),
            })

    return decision_rows, iteration_rows


# Build statistics and graphs

def build_search_depth_rows(iteration_rows):
    """Aggregate search work and pruning statistics separately for each depth."""
    by_depth = defaultdict(list)
    for row in iteration_rows:
        if row.get("depth") is not None:
            by_depth[int(row["depth"])].append(row)

    depth_rows = []
    for depth in sorted(by_depth):
        rows = by_depth[depth]
        times = numeric_values(rows, "elapsed_s")
        nodes = numeric_values(rows, "nodes")
        expanded = numeric_values(rows, "expanded_nodes")
        leaves = numeric_values(rows, "evaluation_leaves")
        branching = numeric_values(rows, "average_branching_factor")
        max_branching = numeric_values(rows, "max_branching_factor")
        cutoffs = numeric_values(rows, "alpha_beta_cutoffs")
        pruned = numeric_values(rows, "children_pruned")
        pruned_fractions = numeric_values(rows, "pruned_child_fraction")
        cutoff_indices = numeric_values(rows, "average_cutoff_move_index")
        root_coverage = numeric_values(rows, "root_coverage_fraction")

        completed_count = sum(row["completed"] for row in rows)
        timeout_count = sum(row["timed_out"] for row in rows)

        depth_rows.append({
            "depth": depth,
            "iterations": len(rows),
            "completed": completed_count,
            "timed_out": timeout_count,
            "timeout_pct": round(100 * timeout_count / len(rows), 2),
            "mean_iteration_time_s": mean_or_none(times),
            "median_iteration_time_s": median_or_none(times),
            "p95_iteration_time_s": p95_or_none(times),
            "max_iteration_time_s": round(max(times), 4) if times else None,
            "mean_nodes": mean_or_none(nodes, 2),
            "total_nodes": int(sum(nodes)),
            "mean_expanded_nodes": mean_or_none(expanded, 2),
            "mean_evaluation_leaves": mean_or_none(leaves, 2),
            "mean_branching_factor": mean_or_none(branching, 3),
            "mean_max_branching_factor": mean_or_none(max_branching, 2),
            "mean_alpha_beta_cutoffs": mean_or_none(cutoffs, 2),
            "total_alpha_beta_cutoffs": int(sum(cutoffs)),
            "mean_children_pruned": mean_or_none(pruned, 2),
            "total_children_pruned": int(sum(pruned)),
            "mean_pruned_child_fraction": mean_or_none(pruned_fractions, 3),
            "mean_cutoff_move_index": mean_or_none(cutoff_indices, 3),
            "mean_root_coverage_pct": (
                round(100 * statistics.mean(root_coverage), 2)
                if root_coverage else None
            ),
            "immediate_win_iterations": sum(
                row["immediate_win_found"] for row in rows
            ),
        })

    return depth_rows


def build_search_summary(agent_name, experiment_name, search_records, log_file_count,
                         malformed_lines, decision_rows, iteration_rows):
    """Build one compact summary row for an experiment's search behaviour."""
    elapsed_times = numeric_values(decision_rows, "elapsed_s")
    completed_depths = numeric_values(decision_rows, "completed_depth")
    chosen_depths = numeric_values(decision_rows, "chosen_move_depth")
    nodes_per_decision = numeric_values(decision_rows, "nodes_across_iterations")

    timed_out = sum(row["timed_out"] for row in decision_rows)
    partial_choices = sum(row["chosen_from_partial_depth"] for row in decision_rows)
    fallback_count = sum(row["used_random_fallback"] for row in decision_rows)
    immediate_win_count = sum(
        row["immediate_win_found"] for row in iteration_rows
    )
    completed_iterations = sum(row["completed"] for row in iteration_rows)
    timed_out_iterations = sum(row["timed_out"] for row in iteration_rows)

    decisions = len(decision_rows)
    iterations = len(iteration_rows)

    return {
        "experiment": experiment_name,
        "agent": agent_name,
        "agent_log_files": log_file_count,
        "search_decisions": decisions,
        "malformed_search_lines": malformed_lines,
        "timed_out_decisions": timed_out,
        "decision_timeout_pct": round(100 * timed_out / decisions, 2) if decisions else 0.0,
        "partial_depth_choices": partial_choices,
        "partial_depth_choice_pct": round(100 * partial_choices / decisions, 2) if decisions else 0.0,
        "immediate_win_iterations": immediate_win_count,
        "random_fallback_decisions": fallback_count,
        "mean_completed_depth": mean_or_none(completed_depths, 3),
        "mean_chosen_move_depth": mean_or_none(chosen_depths, 3),
        "mean_search_time_s": mean_or_none(elapsed_times),
        "median_search_time_s": median_or_none(elapsed_times),
        "p95_search_time_s": p95_or_none(elapsed_times),
        "max_search_time_s": round(max(elapsed_times), 4) if elapsed_times else None,
        "mean_nodes_per_decision": mean_or_none(nodes_per_decision, 2),
        "total_nodes_across_decisions": int(sum(nodes_per_decision)),
        "iterations": iterations,
        "completed_iterations": completed_iterations,
        "timed_out_iterations": timed_out_iterations,
    }


def save_search_depth_plot(decision_rows, output_dir):
    """Plot the depth that ultimately supplied the selected move."""
    counts = Counter(
        row["chosen_move_depth"] for row in decision_rows
        if row["chosen_move_depth"] is not None
    )
    if not counts:
        return

    depths = sorted(counts)
    labels = [str(depth) for depth in depths]
    values = [counts[depth] for depth in depths]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values)
    ax.set(title="Search depth that supplied the chosen move",
           xlabel="Depth", ylabel="Number of decisions")
    for bar, value in zip(bars, values):
        ax.annotate(str(value),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "search_chosen_move_depth.png", dpi=160)
    plt.close(fig)


def save_search_nodes_plot(depth_rows, output_dir):
    if not depth_rows:
        return

    depths = [row["depth"] for row in depth_rows]
    nodes = [row["mean_nodes"] for row in depth_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, nodes, marker="o")
    ax.set(title="Average nodes searched by iteration depth",
           xlabel="Search depth", ylabel="Mean alpha-beta nodes")
    ax.set_xticks(depths)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "search_nodes_by_depth.png", dpi=160)
    plt.close(fig)


def save_search_time_plot(depth_rows, output_dir):
    if not depth_rows:
        return

    depths = [row["depth"] for row in depth_rows]
    times = [row["mean_iteration_time_s"] for row in depth_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, times, marker="o")
    ax.axhline(STEP_TIME_LIMIT - 0.2, linestyle="--", linewidth=2,
               label="Search budget (2.0s minus 0.2s margin)")
    ax.set(title="Average time per iterative-deepening pass",
           xlabel="Search depth", ylabel="Mean iteration time (seconds)")
    ax.set_xticks(depths)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "search_time_by_depth.png", dpi=160)
    plt.close(fig)


def save_search_pruning_plot(depth_rows, output_dir):
    rows = [row for row in depth_rows
            if row["mean_pruned_child_fraction"] is not None]
    if not rows:
        return

    depths = [row["depth"] for row in rows]
    fractions = [100 * row["mean_pruned_child_fraction"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, fractions, marker="o")
    ax.set(title="Average alpha-beta child-pruning fraction by depth",
           xlabel="Search depth", ylabel="Children pruned (%)", ylim=(0, 100))
    ax.set_xticks(depths)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "search_pruning_by_depth.png", dpi=160)
    plt.close(fig)


def write_search_analysis(experiment_name, result_filepath, output_dir):
    agent_name = infer_target_agent(result_filepath) or "unknown agent"
    search_records, log_file_count, malformed_lines = load_search_logs(result_filepath)

    if not search_records:
        print(
            f"No SEARCH_STATS records found for {agent_name} in {experiment_name}. "
            f"Looked for copied logs in: {AGENT_LOGS_DIR}"
        )
        return None

    decision_rows, iteration_rows = analyze_search_records(search_records)
    depth_rows = build_search_depth_rows(iteration_rows)
    summary = build_search_summary(
        agent_name, experiment_name, search_records, log_file_count, malformed_lines,
        decision_rows, iteration_rows
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "search_summary.csv", [summary], list(summary.keys())
    )

    decision_fields = [
        "log_file", "agent_turn", "player", "root_ply", "completed_depth",
        "attempted_depth", "chosen_move_depth", "chosen_from_partial_depth",
        "timed_out", "elapsed_s", "nodes_across_iterations",
        "cutoffs_across_iterations", "children_pruned_across_iterations",
        "iteration_count", "used_random_fallback", "chosen_move",
    ]
    write_csv(output_dir / "search_decisions.csv", decision_rows, decision_fields)

    iteration_fields = [
        "log_file", "agent_turn", "player", "depth", "completed", "timed_out",
        "elapsed_s", "nodes", "expanded_nodes", "evaluation_leaves",
        "depth_limit_leaves", "terminal_nodes", "legal_moves_generated",
        "average_branching_factor", "max_branching_factor", "alpha_beta_cutoffs",
        "children_pruned", "pruned_child_fraction", "average_cutoff_move_index",
        "root_legal_moves", "root_moves_started", "root_moves_considered",
        "root_coverage_fraction", "immediate_win_found", "best_move", "best_value",
    ]
    write_csv(output_dir / "search_iterations.csv", iteration_rows, iteration_fields)

    depth_fields = list(depth_rows[0].keys()) if depth_rows else ["depth"]
    write_csv(output_dir / "search_depth_breakdown.csv", depth_rows, depth_fields)

    save_search_depth_plot(decision_rows, output_dir)
    save_search_nodes_plot(depth_rows, output_dir)
    save_search_time_plot(depth_rows, output_dir)
    save_search_pruning_plot(depth_rows, output_dir)

    print(f"\nSearch analysis for {agent_name}:")
    print(
        f"  Decisions: {summary['search_decisions']} | "
        f"Timeouts: {summary['timed_out_decisions']} "
        f"({summary['decision_timeout_pct']}%) | "
        f"Partial-depth choices: {summary['partial_depth_choices']} "
        f"({summary['partial_depth_choice_pct']}%)"
    )
    print(
        f"  Mean search time: {summary['mean_search_time_s']}s | "
        f"P95: {summary['p95_search_time_s']}s | "
        f"Mean nodes/decision: {summary['mean_nodes_per_decision']}"
    )
    print(f"  Search CSVs and plots saved under: {output_dir}")

    return summary


# Analyze agent logs for each result file

def analyze_experiments(result_files, output_root):
    summary_rows = []

    for result_filepath in result_files:
        experiment_name = result_filepath.stem
        experiment_output = output_root / experiment_name

        summary = write_search_analysis(
            experiment_name, result_filepath, experiment_output
        )
        if summary is not None:
            summary_rows.append(summary)

    if summary_rows:
        summary_path = output_root / "search_experiment_summary.csv"
        write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
        print(f"Search-log summary saved to: {summary_path}")

    return summary_rows