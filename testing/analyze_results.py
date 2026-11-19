"""Analyze the output jsonl files from the experiments and create tables and plots."""

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# Constants

TESTING_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TESTING_DIR / "results"
OUTPUT_DIR = TESTING_DIR / "analysis_output"

STEP_TIME_LIMIT = 2.0

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
            records.append(json.loads(line))

    return records

def get_agent_seat(record):
    for seat in ("1", "2"):
        if record.get("players", {}).get(seat, {}).get("agent") == "student_agent":
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
    }

    for record in records:
        agent_seat = get_agent_seat(record)
        winner = record.get("winner")

        outcome, points = get_outcome(winner, agent_seat)
        reason = record.get("reason", "unknown")
        opponent_seat = "2" if agent_seat == 1 else "1"
        opponent_name = record.get("players", {}).get(opponent_seat, {}).get("agent", "unknown")

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
    }

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

    # Create the boxplot
    ax.boxplot(values, showfliers=False)

    # Set the labels underneath each box
    ax.set_xticks(range(1, len(labels) + 1), labels=labels)

    ax.axhline(
        STEP_TIME_LIMIT,
        linestyle="--",
        linewidth=2,
        label=f"{STEP_TIME_LIMIT:.1f}s soft limit"
    )

    ax.set(
        title="Agent move time by game phase",
        ylabel="Agent time (seconds)"
    )

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


# Main program

def main():
    result_files = sorted(RESULTS_DIR.glob("*.jsonl")) + sorted(RESULTS_DIR.glob("*.json"))

    if not result_files:
        print(f"No .json or .jsonl result files found in: {RESULTS_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for filepath in result_files:
        records = load_records(filepath)
        stats = analyze_records(records)

        if stats["games"] == 0:
            print(f"Skipping {filepath.name}: no completed games involving student_agent.")
            continue

        summary = build_summary_row(filepath.name, stats)
        summary_rows.append(summary)
        print_summary(summary)

        experiment_output = OUTPUT_DIR / filepath.stem
        experiment_output.mkdir(parents=True, exist_ok=True)
        write_breakdown_tables(experiment_output, stats)
        make_plots(stats, experiment_output)

    if summary_rows:
        write_csv(
            OUTPUT_DIR / "experiment_summary.csv",
            summary_rows,
            list(summary_rows[0].keys()),
        )
        print("\nAnalysis complete.")
        print(f"Summary CSV and plots saved under: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()