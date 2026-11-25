"""Run experiments for comparing agents."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import simulator
import random
import json
import shutil

RESULTS_DIR = PROJECT_ROOT / "testing" / "results"
LOGS_DIR = PROJECT_ROOT / "testing" / "agent_logs"


def count_existing_records(filepath):
    if not filepath.exists():
        return 0

    with filepath.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())

def copy_student_logs(filepath, first_new_record):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0

    with filepath.open("r", encoding="utf-8") as file:
        for record_index, line in enumerate(file):
            if record_index < first_new_record or not line.strip():
                continue

            record = json.loads(line)

            for seat in ("1", "2"):
                agent = record.get("players", {}).get(seat, {}).get("agent")

                if agent != "student_agent":
                    continue

                log_path = record.get("log_paths", {}).get(seat)

                if not log_path:
                    print(f"No log path found for game {record_index + 1}, seat {seat}.")
                    continue

                source = Path(log_path)

                if not source.is_file():
                    print(f"Could not find agent log: {source}")
                    continue

                destination = LOGS_DIR / (
                    f"{filepath.stem}_game{record_index + 1:04d}_seat{seat}.txt"
                )

                shutil.copy2(source, destination)
                copied += 1

    return copied

def run_simulation(player, opponent, num_runs, seed):
    """Run a series of games between the given player and opponent agents."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"{player}_vs_{opponent}_seed{seed}.jsonl"

    # Remember how many records existed before this experiment.
    first_new_record = count_existing_records(output_file)

    simulator.main(["--player_1", player, 
                    "--player_2", opponent, 
                    "--seed", str(seed),
                    "--autoplay",
                    "--autoplay_runs", str(num_runs),
                    "--strict",
                    "--json", str(output_file),])

    # Copy the student agent's logs out of the temporary directory.
    copied = copy_student_logs(output_file, first_new_record)

    print(f"Simulation complete. Results saved to {output_file}.")
    print(f"Copied {copied} student agent logs to {LOGS_DIR}.")

if __name__ == "__main__":

    seed = random.randint(0, 10000)

    #run_simulation("student_agent", "random_agent", num_runs=10, seed=seed)

    run_simulation("student_agent", "greedy_agent", num_runs=50, seed=seed)

    run_simulation("student_agent", "student_agent_2", num_runs=50, seed=seed)