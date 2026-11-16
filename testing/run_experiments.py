"""Run experiments for comparing agents."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import simulator

def run_simulation(player, opponent, num_runs, seed):
    """Run a series of games between the given player and opponent agents."""
    output_file = "testing/results/" + f"{player}_vs_{opponent}_seed{seed}.jsonl"
    simulator.main(["--player_1", player, 
                    "--player_2", opponent, 
                    "--seed", str(seed),
                    "--autoplay",
                    "--autoplay_runs", str(num_runs),
                    "--strict",
                    "--json", output_file,])
    print(f"Simulation complete. Results saved to {output_file}.")

if __name__ == "__main__":

    run_simulation("student_agent", "random_agent", num_runs=10, seed=67)

    run_simulation("student_agent", "greedy_agent", num_runs=10, seed=67)

    run_simulation("student_agent", "student_agent", num_runs=10, seed=67)