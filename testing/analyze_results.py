"""Analyze the output jsonl files from the experiments and create tables and plots."""

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import analyze_agent_logs


# Constants

TESTING_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TESTING_DIR / "results"
OUTPUT_DIR = TESTING_DIR / "analysis_output"

STEP_TIME_LIMIT = 2.0
MEMORY_LIMIT_MB = 500.0

SIZE_NAMES = {
    0: "Small",
    1: "Medium",
    2: "Large",
}

FORFEIT_REASONS = {
    "hard_timeout",
    "crash",
    "too_many_strikes",
    "init_failed",
    "init_timeout",
}

PHASES = ("Opening", "Middle", "Endgame")


# Helpers

def load_records(filepath):
    records = []

    with filepath.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {filepath.name}, line {line_number}: {error}"
                ) from error

    return records

def get_agent_seat(record):
    players = record.get("players")
    if not isinstance(players, dict):
        return None

    for seat in ("1", "2"):
        player_info = players.get(seat)
        if isinstance(player_info, dict) and player_info.get("agent") == "student_agent":
            return int(seat)
    return None

def get_outcome(winner, agent_seat):
    if winner == agent_seat:
        return "win", 1.0
    if winner == 0:
        return "draw", 0.5
    return "loss", 0.0

def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight

def empty_group():
    return {
        "games": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0.0,
        "seeds": set(),
    }

def update_group(group, record, outcome, points):
    group["games"] += 1
    outcome_key = {"win": "wins", "draw": "draws", "loss": "losses"}[outcome]
    group[outcome_key] += 1
    group["points"] += points
    if record.get("seed") is not None:
        group["seeds"].add(record["seed"])

def group_match_points(group):
    if group["games"] == 0:
        return 0.0
    return 100 * group["points"] / group["games"]


def extract_memory_mb(record, agent_seat):
    memory_by_seat = record.get("max_rss_mb")
    if not isinstance(memory_by_seat, dict):
        return None

    raw_value = memory_by_seat.get(str(agent_seat))
    if raw_value is None:
        return None

    try:
        memory_mb = float(raw_value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(memory_mb) or memory_mb < 0:
        return None

    return memory_mb


def get_memory_metrics(values):
    if not values:
        return {
            "count": 0,
            "mean_mb": None,
            "median_mb": None,
            "p95_mb": None,
            "min_mb": None,
            "max_mb": None,
            "std_mb": None,
            "mean_budget_pct": None,
            "max_budget_pct": None,
            "max_headroom_mb": None,
            "games_over_budget": 0,
            "games_over_budget_pct": 0.0,
        }

    maximum = max(values)
    return {
        "count": len(values),
        "mean_mb": statistics.mean(values),
        "median_mb": statistics.median(values),
        "p95_mb": percentile(values, 0.95),
        "min_mb": min(values),
        "max_mb": maximum,

        # Population standard deviation is for all measurements in this experiment
        "std_mb": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "mean_budget_pct": 100 * statistics.mean(values) / MEMORY_LIMIT_MB,
        "max_budget_pct": 100 * maximum / MEMORY_LIMIT_MB,
        "max_headroom_mb": MEMORY_LIMIT_MB - maximum,
        "games_over_budget": sum(value > MEMORY_LIMIT_MB for value in values),
        "games_over_budget_pct": 100 * sum(value > MEMORY_LIMIT_MB for value in values) / len(values),
    }


# Extract statistics

def analyze_records(records):
    stats = {
        "games": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0.0,
        "seats": {
            1: empty_group(),
            2: empty_group(),
        },
        "times": [],
        "phase_times": {phase: [] for phase in PHASES},
        "progress_times": defaultdict(list),
        "plies": [],
        "strikes": 0,
        "forfeits": 0,
        "flags": Counter(),
        "reasons": defaultdict(Counter),
        "obstacles": defaultdict(empty_group),
        "removed_sizes": defaultdict(empty_group),
        "move_types": Counter(),
        "piece_sizes": Counter(),
        "move_type_by_size": Counter(),
        "opponents": Counter(),
        "memory_samples": [],
        "memory_by_seat": {1: [], 2: []},
        "memory_games_by_seat": {1: 0, 2: 0},
        "memory_missing_by_seat": {1: 0, 2: 0},
        "memory_records": [],
        "ignored_records": 0,
        "ignored_record_details": [],
    }

    for record_index, record in enumerate(records, start=1):
        agent_seat = get_agent_seat(record)

        if agent_seat is None:
            stats["ignored_records"] += 1
            if len(stats["ignored_record_details"]) < 5:
                players = record.get("players")
                if isinstance(players, dict):
                    player_agents = {
                        seat: (info.get("agent") if isinstance(info, dict) else info)
                        for seat, info in players.items()
                    }
                else:
                    player_agents = players
                stats["ignored_record_details"].append({
                    "record": record_index,
                    "players": player_agents,
                })
            continue

        winner = record.get("winner")

        outcome, points = get_outcome(winner, agent_seat)
        reason = record.get("reason", "unknown")
        opponent_seat = "2" if agent_seat == 1 else "1"
        opponent_info = record.get("players", {}).get(opponent_seat, {})
        opponent_name = (
            opponent_info.get("agent", "unknown")
            if isinstance(opponent_info, dict) else "unknown"
        )

        stats["games"] += 1
        outcome_key = {"win": "wins", "draw": "draws", "loss": "losses"}[outcome]
        stats[outcome_key] += 1
        stats["points"] += points
        stats["opponents"][opponent_name] += 1
        stats["plies"].append(record.get("plies", 0))
        stats["reasons"][reason][outcome] += 1

        update_group(stats["seats"][agent_seat], record, outcome, points)

        stats["strikes"] += record.get("strikes", {}).get(str(agent_seat), 0)
        if outcome == "loss" and reason in FORFEIT_REASONS:
            stats["forfeits"] += 1

        # Get the agent's peak resident memory for this game
        memory_mb = extract_memory_mb(record, agent_seat)
        stats["memory_games_by_seat"][agent_seat] += 1
        if memory_mb is None:
            stats["memory_missing_by_seat"][agent_seat] += 1
        else:
            stats["memory_samples"].append(memory_mb)
            stats["memory_by_seat"][agent_seat].append(memory_mb)

        stats["memory_records"].append({
            "record": record_index,
            "seed": record.get("seed"),
            "agent_seat": agent_seat,
            "opponent": opponent_name,
            "outcome": outcome,
            "reason": reason,
            "plies": record.get("plies"),
            "max_rss_mb": memory_mb,
            "memory_budget_pct": (
                100 * memory_mb / MEMORY_LIMIT_MB if memory_mb is not None else None
            ),
        })

        # Break down results by starting obstacle and removed reserve-piece size
        obstacle = record.get("obstacle")
        if isinstance(obstacle, list) and len(obstacle) == 2:
            update_group(stats["obstacles"][tuple(obstacle)], record, outcome, points)

        removed_size = record.get("removed_size")
        if removed_size in SIZE_NAMES:
            update_group(stats["removed_sizes"][removed_size], record, outcome, points)

        # Only inspect moves made by our agent
        total_plies = max(record.get("plies", 1), 1)

        for move in record.get("moves", []):
            if move.get("player") != agent_seat:
                continue

            for flag in move.get("flags", []):
                stats["flags"][flag] += 1

            if "pass" in move:
                continue

            source = move.get("src")
            move_type = "Placement" if source is None else "Board move"
            piece_size = move.get("size")
            size_name = SIZE_NAMES.get(piece_size, f"Unknown ({piece_size})")

            stats["move_types"][move_type] += 1
            stats["piece_sizes"][size_name] += 1
            stats["move_type_by_size"][(move_type, size_name)] += 1

            agent_time = move.get("agent_time")
            if agent_time is None:
                continue

            agent_time = float(agent_time)
            stats["times"].append(agent_time)

            # Relative game progress: 0.0 at the start, approximately 1.0 at the end
            ply = move.get("ply", 0)
            progress = min(max(ply / max(total_plies - 1, 1), 0.0), 1.0)

            if progress < 1 / 3:
                phase = "Opening"
            elif progress < 2 / 3:
                phase = "Middle"
            else:
                phase = "Endgame"
            stats["phase_times"][phase].append(agent_time)

            # Put the move in a 10%-wide game-progress bucket for the trend plot
            progress_bin = min(int(progress * 10), 9)
            stats["progress_times"][progress_bin].append(agent_time)

    stats["memory_missing_games"] = sum(stats["memory_missing_by_seat"].values())
    stats["memory_metrics"] = get_memory_metrics(stats["memory_samples"])
    stats["memory_metrics_by_seat"] = {
        seat: get_memory_metrics(values)
        for seat, values in stats["memory_by_seat"].items()
    }

    return stats


# CSVs

def write_csv(filepath, rows, fieldnames):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def build_summary_row(filename, stats):
    games = stats["games"]
    times = stats["times"]
    memory = stats["memory_metrics"]

    opponents = ", ".join(sorted(stats["opponents"]))

    return {
        "experiment": filename,
        "opponent(s)": opponents,
        "games": games,
        "wins": stats["wins"],
        "draws": stats["draws"],
        "losses": stats["losses"],
        "match_points_pct": round(group_match_points({
            "games": games, "points": stats["points"]
        }), 2) if games else 0.0,
        "p1_match_points_pct": round(group_match_points(stats["seats"][1]), 2),
        "p2_match_points_pct": round(group_match_points(stats["seats"][2]), 2),
        "mean_step_time_s": round(statistics.mean(times), 4) if times else None,
        "median_step_time_s": round(statistics.median(times), 4) if times else None,
        "p95_step_time_s": round(percentile(times, 0.95), 4) if times else None,
        "max_step_time_s": round(max(times), 4) if times else None,
        "mean_game_plies": round(statistics.mean(stats["plies"]), 2) if stats["plies"] else None,
        "strikes": stats["strikes"],
        "forfeits": stats["forfeits"],
        "memory_budget_mb": MEMORY_LIMIT_MB,
        "memory_games_measured": memory["count"],
        "memory_games_missing": stats["memory_missing_games"],
        "memory_games_over_budget": memory["games_over_budget"],
        "memory_games_over_budget_pct": round(memory["games_over_budget_pct"], 2),
        "mean_memory_mb": round(memory["mean_mb"], 3) if memory["mean_mb"] is not None else None,
        "median_memory_mb": round(memory["median_mb"], 3) if memory["median_mb"] is not None else None,
        "p95_memory_mb": round(memory["p95_mb"], 3) if memory["p95_mb"] is not None else None,
        "max_memory_mb": round(memory["max_mb"], 3) if memory["max_mb"] is not None else None,
        "min_memory_mb": round(memory["min_mb"], 3) if memory["min_mb"] is not None else None,
        "memory_std_mb": round(memory["std_mb"], 3) if memory["std_mb"] is not None else None,
        "max_memory_budget_pct": round(memory["max_budget_pct"], 2) if memory["max_budget_pct"] is not None else None,
        "memory_headroom_mb": round(memory["max_headroom_mb"], 3) if memory["max_headroom_mb"] is not None else None,
    }


def write_memory_tables(output_dir, stats):
    """Write per-game measurements and aggregate memory statistics."""
    memory_game_fields = [
        "record", "seed", "agent_seat", "opponent", "outcome", "reason",
        "plies", "max_rss_mb", "memory_budget_pct",
    ]
    write_csv(
        output_dir / "memory_by_game.csv",
        stats["memory_records"],
        memory_game_fields,
    )

    summary_rows = []
    for scope, values, games, missing in [
        (
            "Overall",
            stats["memory_samples"],
            stats["games"],
            stats["memory_missing_games"],
        ),
        (
            "Student agent as P1",
            stats["memory_by_seat"][1],
            stats["memory_games_by_seat"][1],
            stats["memory_missing_by_seat"][1],
        ),
        (
            "Student agent as P2",
            stats["memory_by_seat"][2],
            stats["memory_games_by_seat"][2],
            stats["memory_missing_by_seat"][2],
        ),
    ]:
        metrics = get_memory_metrics(values)
        summary_rows.append({
            "scope": scope,
            "games": games,
            "games_measured": metrics["count"],
            "games_missing": missing,
            "memory_budget_mb": MEMORY_LIMIT_MB,
            "mean_mb": round(metrics["mean_mb"], 3) if metrics["mean_mb"] is not None else None,
            "median_mb": round(metrics["median_mb"], 3) if metrics["median_mb"] is not None else None,
            "p95_mb": round(metrics["p95_mb"], 3) if metrics["p95_mb"] is not None else None,
            "min_mb": round(metrics["min_mb"], 3) if metrics["min_mb"] is not None else None,
            "max_mb": round(metrics["max_mb"], 3) if metrics["max_mb"] is not None else None,
            "std_mb": round(metrics["std_mb"], 3) if metrics["std_mb"] is not None else None,
            "mean_budget_pct": round(metrics["mean_budget_pct"], 2) if metrics["mean_budget_pct"] is not None else None,
            "max_budget_pct": round(metrics["max_budget_pct"], 2) if metrics["max_budget_pct"] is not None else None,
            "max_headroom_mb": round(metrics["max_headroom_mb"], 3) if metrics["max_headroom_mb"] is not None else None,
            "games_over_budget": metrics["games_over_budget"],
            "games_over_budget_pct": round(metrics["games_over_budget_pct"], 2),
        })

    write_csv(
        output_dir / "memory_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )


def write_breakdown_tables(output_dir, stats):
    obstacle_rows = []
    for (row, col), group in sorted(stats["obstacles"].items()):
        obstacle_rows.append({
            "obstacle_row": row,
            "obstacle_col": col,
            "games": group["games"],
            "seed_configurations": len(group["seeds"]),
            "wins": group["wins"],
            "draws": group["draws"],
            "losses": group["losses"],
            "match_points_pct": round(group_match_points(group), 2),
        })
    write_csv(
        output_dir / "obstacle_breakdown.csv",
        obstacle_rows,
        ["obstacle_row", "obstacle_col", "games", "seed_configurations",
         "wins", "draws", "losses", "match_points_pct"],
    )

    removed_rows = []
    for size in (0, 1, 2):
        group = stats["removed_sizes"].get(size)
        if group:
            removed_rows.append({
                "removed_piece_size": SIZE_NAMES[size],
                "games": group["games"],
                "seed_configurations": len(group["seeds"]),
                "wins": group["wins"],
                "draws": group["draws"],
                "losses": group["losses"],
                "match_points_pct": round(group_match_points(group), 2),
            })
    write_csv(
        output_dir / "removed_size_breakdown.csv",
        removed_rows,
        ["removed_piece_size", "games", "seed_configurations",
         "wins", "draws", "losses", "match_points_pct"],
    )

    phase_rows = []
    for phase in PHASES:
        values = stats["phase_times"][phase]
        if values:
            phase_rows.append({
                "phase": phase,
                "moves": len(values),
                "mean_time_s": round(statistics.mean(values), 4),
                "median_time_s": round(statistics.median(values), 4),
                "p95_time_s": round(percentile(values, 0.95), 4),
                "max_time_s": round(max(values), 4),
            })
    write_csv(
        output_dir / "phase_times.csv",
        phase_rows,
        ["phase", "moves", "mean_time_s", "median_time_s", "p95_time_s", "max_time_s"],
    )

    move_rows = []
    for (move_type, size_name), count in sorted(stats["move_type_by_size"].items()):
        move_rows.append({"move_type": move_type, "piece_size": size_name, "count": count})
    write_csv(output_dir / "move_behavior.csv", move_rows,
              ["move_type", "piece_size", "count"])

    reason_rows = []
    for reason, outcomes in sorted(stats["reasons"].items()):
        reason_rows.append({
            "ending_reason": reason,
            "wins": outcomes["win"],
            "draws": outcomes["draw"],
            "losses": outcomes["loss"],
            "games": sum(outcomes.values()),
        })
    write_csv(output_dir / "ending_reasons.csv", reason_rows,
              ["ending_reason", "wins", "draws", "losses", "games"])

    flag_rows = [
        {"flag": flag, "count": count}
        for flag, count in sorted(stats["flags"].items())
    ]
    write_csv(output_dir / "move_flags.csv", flag_rows, ["flag", "count"])

    write_memory_tables(output_dir, stats)


def print_summary(row):
    print("\n" + "=" * 78)
    print(f"EXPERIMENT: {row['experiment']}")
    print(f"Opponent(s): {row['opponent(s)']}")
    print("=" * 78)
    print(f"Games: {row['games']} | W-D-L: "
          f"{row['wins']}-{row['draws']}-{row['losses']} | "
          f"Match points: {row['match_points_pct']:.1f}%")
    print(f"P1 match points: {row['p1_match_points_pct']:.1f}% | "
          f"P2 match points: {row['p2_match_points_pct']:.1f}%")
    print(f"Step time (s): mean {row['mean_step_time_s']} | "
          f"median {row['median_step_time_s']} | "
          f"95th percentile {row['p95_step_time_s']} | "
          f"max {row['max_step_time_s']}")
    print(f"Mean game length: {row['mean_game_plies']} plies | "
          f"Strikes: {row['strikes']} | Forfeits: {row['forfeits']}")
    print(
        f"Peak RSS memory (MB): mean {row['mean_memory_mb']} | "
        f"median {row['median_memory_mb']} | p95 {row['p95_memory_mb']} | "
        f"max {row['max_memory_mb']} / {MEMORY_LIMIT_MB:.0f} MB "
        f"({row['max_memory_budget_pct']}% of budget) | "
        f"over-budget games {row['memory_games_over_budget']} | "
        f"missing {row['memory_games_missing']}"
    )


# Plots

def save_time_histogram(times, output_dir):
    if not times:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(times, bins=20, edgecolor="black")
    ax.axvline(STEP_TIME_LIMIT, linestyle="--", linewidth=2,
               label=f"{STEP_TIME_LIMIT:.1f}s soft limit")
    ax.set(title="Agent move-time distribution", xlabel="Agent time (seconds)",
           ylabel="Number of moves")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "step_time_histogram.png", dpi=160)
    plt.close(fig)


def save_phase_boxplot(phase_times, output_dir):
    labels = [phase for phase in PHASES if phase_times[phase]]

    if not labels:
        return

    values = [phase_times[phase] for phase in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(values, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels=labels)
    ax.axhline(
        STEP_TIME_LIMIT,
        linestyle="--",
        linewidth=2,
        label=f"{STEP_TIME_LIMIT:.1f}s soft limit"
    )
    ax.set(title="Agent move time by game phase", ylabel="Agent time (seconds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "step_time_by_phase.png", dpi=160)
    plt.close(fig)


def save_progress_plot(progress_times, output_dir):
    bins = sorted(progress_times)
    if not bins:
        return

    x_values = [(index + 0.5) * 10 for index in bins]
    means = [statistics.mean(progress_times[index]) for index in bins]
    counts = [len(progress_times[index]) for index in bins]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_values, means, marker="o")
    ax.axhline(STEP_TIME_LIMIT, linestyle="--", linewidth=2,
               label=f"{STEP_TIME_LIMIT:.1f}s soft limit")
    for x, y, count in zip(x_values, means, counts):
        ax.annotate(f"n={count}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.set(title="Mean agent move time over game progress",
           xlabel="Game progress (%)", ylabel="Mean agent time (seconds)",
           xlim=(0, 100))
    ax.set_xticks(range(0, 101, 10))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "step_time_by_progress.png", dpi=160)
    plt.close(fig)

def save_obstacle_heatmap(obstacles, output_dir):
    if not obstacles:
        return

    grid = np.full((4, 4), np.nan)
    for (row, col), group in obstacles.items():
        grid[row, col] = group_match_points(group)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(grid, vmin=0, vmax=100, cmap="RdYlGn")
    ax.set(title="Match points by obstacle position", xlabel="Obstacle column",
           ylabel="Obstacle row", xticks=range(4), yticks=range(4))

    for row in range(4):
        for col in range(4):
            group = obstacles.get((row, col))
            if group:
                label = (f"{group_match_points(group):.0f}%\n"
                         f"{group['games']} games / {len(group['seeds'])} seeds")
                ax.text(col, row, label, ha="center", va="center", fontsize=8)
            else:
                ax.text(col, row, "No data", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label="Match points (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "obstacle_heatmap.png", dpi=160)
    plt.close(fig)

def save_removed_size_plot(removed_sizes, output_dir):
    sizes = [size for size in (0, 1, 2) if size in removed_sizes]
    if not sizes:
        return

    labels = [SIZE_NAMES[size] for size in sizes]
    values = [group_match_points(removed_sizes[size]) for size in sizes]
    counts = [removed_sizes[size]["games"] for size in sizes]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values)
    ax.set(title="Agent performance by removed reserve-piece size",
           xlabel="Piece size removed", ylabel="Match points (%)", ylim=(0, 100))
    for bar, count, value in zip(bars, counts, values):
        ax.annotate(f"{value:.1f}%\nn={count}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "removed_size_performance.png", dpi=160)
    plt.close(fig)

def save_move_type_plot(move_types, output_dir):
    labels = [label for label in ("Placement", "Board move") if move_types[label]]
    if not labels:
        return

    values = [move_types[label] for label in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values)
    ax.set(title="Agent move types", ylabel="Number of moves")
    for bar, value in zip(bars, values):
        ax.annotate(str(value),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "move_types.png", dpi=160)
    plt.close(fig)

def save_piece_size_plot(piece_sizes, output_dir):
    labels = [name for name in ("Small", "Medium", "Large") if piece_sizes[name]]
    if not labels:
        return

    values = [piece_sizes[name] for name in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values)
    ax.set(title="Agent piece-size usage", ylabel="Number of moves")
    for bar, value in zip(bars, values):
        ax.annotate(str(value),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "piece_size_usage.png", dpi=160)
    plt.close(fig)

def save_ending_reason_plot(reasons, output_dir):
    labels = sorted(reasons)
    if not labels:
        return

    wins = [reasons[label]["win"] for label in labels]
    draws = [reasons[label]["draw"] for label in labels]
    losses = [reasons[label]["loss"] for label in labels]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, wins, width, label="Agent wins")
    ax.bar(x, draws, width, label="Draws")
    ax.bar(x + width, losses, width, label="Agent losses")
    ax.set(title="Game outcomes by ending reason", ylabel="Number of games",
           xticks=x, xticklabels=labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ending_reasons.png", dpi=160)
    plt.close(fig)

def save_memory_distribution(memory_values, output_dir):
    if not memory_values:
        return

    metrics = get_memory_metrics(memory_values)
    bins = min(20, max(5, math.ceil(math.sqrt(len(memory_values)))))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(memory_values, bins=bins, edgecolor="black")
    ax.axvline(metrics["mean_mb"], linestyle="--", linewidth=2,
               label=f"Mean: {metrics['mean_mb']:.2f} MB")
    ax.axvline(metrics["max_mb"], linestyle=":", linewidth=2,
               label=f"Maximum: {metrics['max_mb']:.2f} MB")
    ax.set(
        title="Student agent peak resident memory distribution",
        xlabel="Peak resident memory per game (MB)",
        ylabel="Number of games",
    )
    ax.text(
        0.98, 0.95,
        f"Budget: {MEMORY_LIMIT_MB:.0f} MB\n"
        f"Samples: {metrics['count']}\n"
        f"95th percentile: {metrics['p95_mb']:.2f} MB",
        transform=ax.transAxes, ha="right", va="top",
        bbox={"boxstyle": "round", "alpha": 0.12},
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "memory_usage_distribution.png", dpi=160)
    plt.close(fig)

def save_memory_budget_plot(memory_values, output_dir):
    if not memory_values:
        return

    metrics = get_memory_metrics(memory_values)
    labels = ["Mean", "Median", "95th percentile", "Maximum"]
    values_mb = [
        metrics["mean_mb"], metrics["median_mb"],
        metrics["p95_mb"], metrics["max_mb"],
    ]
    values_pct = [100 * value / MEMORY_LIMIT_MB for value in values_mb]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values_pct)
    ax.axhline(100, linestyle="--", linewidth=2, label="Memory budget (100%)")
    for bar, value_mb, value_pct in zip(bars, values_mb, values_pct):
        ax.annotate(
            f"{value_mb:.2f} MB\n{value_pct:.2f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8,
        )
    upper = max(110, max(values_pct, default=0) * 1.2)
    ax.set(
        title="Student agent memory usage relative to budget",
        ylabel=f"Memory budget used (%) — {MEMORY_LIMIT_MB:.0f} MB budget",
        ylim=(0, upper),
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "memory_budget_usage.png", dpi=160)
    plt.close(fig)

def save_memory_by_game_plot(memory_records, output_dir):
    measured = [record for record in memory_records if record["max_rss_mb"] is not None]
    if not measured:
        return

    game_numbers = [record["record"] for record in measured]
    memory_values = [record["max_rss_mb"] for record in measured]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(game_numbers, memory_values, marker="o", linestyle="-")
    ax.set(
        title="Student agent peak resident memory by game",
        xlabel="Record number in result file",
        ylabel="Peak resident memory (MB)",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "memory_usage_by_game.png", dpi=160)
    plt.close(fig)

def save_memory_by_seat_plot(memory_by_seat, output_dir):
    seats = [seat for seat in (1, 2) if memory_by_seat[seat]]
    if not seats:
        return

    values = [memory_by_seat[seat] for seat in seats]
    labels = [f"Student as P{seat}" for seat in seats]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(values, labels=labels, showfliers=True)
    ax.set(
        title="Student agent peak memory by player seat",
        ylabel="Peak resident memory (MB)",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "memory_usage_by_seat.png", dpi=160)
    plt.close(fig)

def make_plots(stats, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    save_time_histogram(stats["times"], output_dir)
    save_phase_boxplot(stats["phase_times"], output_dir)
    save_progress_plot(stats["progress_times"], output_dir)
    save_obstacle_heatmap(stats["obstacles"], output_dir)
    save_removed_size_plot(stats["removed_sizes"], output_dir)
    save_move_type_plot(stats["move_types"], output_dir)
    save_piece_size_plot(stats["piece_sizes"], output_dir)
    save_ending_reason_plot(stats["reasons"], output_dir)
    save_memory_distribution(stats["memory_samples"], output_dir)
    save_memory_budget_plot(stats["memory_samples"], output_dir)
    save_memory_by_game_plot(stats["memory_records"], output_dir)
    save_memory_by_seat_plot(stats["memory_by_seat"], output_dir)


# Main program

def main():
    result_files = sorted(RESULTS_DIR.glob("*.jsonl")) + sorted(RESULTS_DIR.glob("*.json"))

    if not result_files:
        print(f"No .json or .jsonl result files found in: {RESULTS_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    analyzed_result_files = []

    for filepath in result_files:
        records = load_records(filepath)
        stats = analyze_records(records)

        if stats["ignored_records"]:
            print(
                f"Warning: ignored {stats['ignored_records']} record(s) in "
                f"{filepath.name} because they did not contain student_agent."
            )
            for detail in stats["ignored_record_details"]:
                print(f"  Record {detail['record']}: players={detail['players']}")

        if stats["games"] == 0:
            print(f"Skipping {filepath.name}: no completed games involving student_agent.")
            continue

        if stats["memory_missing_games"]:
            print(
                f"Warning: {stats['memory_missing_games']} game(s) in {filepath.name} "
                "had no valid max_rss_mb measurement for student_agent."
            )

        summary = build_summary_row(filepath.name, stats)
        summary_rows.append(summary)
        print_summary(summary)

        experiment_output = OUTPUT_DIR / filepath.stem
        experiment_output.mkdir(parents=True, exist_ok=True)
        write_breakdown_tables(experiment_output, stats)
        make_plots(stats, experiment_output)

        analyzed_result_files.append(filepath)

    if summary_rows:
        write_csv(
            OUTPUT_DIR / "experiment_summary.csv",
            summary_rows,
            list(summary_rows[0].keys()),
        )

    if analyzed_result_files:
        analyze_agent_logs.analyze_experiments(analyzed_result_files, OUTPUT_DIR)

    if summary_rows:
        print("\nAnalysis complete.")
        print(f"Summary CSV and plots saved under: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()