"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view)
from constants import *
import numpy as np
import time
import json

MAX_DEPTH = 10
WIN = 10 ** 6   # score heuristic value
MARGIN = 0.2    # TODO: establish a good margin (do this at the end)
PIECE_WEIGHT = (1, 3, 6)  # for small, medium & large pieces in reserve, we can tweak this later

class SearchTimeout(Exception): pass

LINES = ([[(i, j) for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)] + # 4 rows
        [[(j, i) for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)] + # 4 cols
        [[(i, i) for i in range(BOARD_SIZE)],
         [(i, BOARD_SIZE-1-i) for i in range(BOARD_SIZE)]]) # 2 diagonals

# Create counters for one IDS pass
def make_iteration_stats(depth):
    return {
        "depth": depth,
        "completed": False,
        "elapsed_s": 0.0,

        # Search tree size
        "nodes": 0,
        "nodes_by_remaining_depth": {},
        "expanded_nodes": 0,
        "evaluation_leaves": 0,
        "depth_limit_leaves": 0,
        "no_move_leaves": 0,

        # Terminal positions
        "terminal_nodes": 0,
        "terminal_wins": 0,
        "terminal_losses": 0,
        "terminal_draws": 0,

        # Branching factor
        "legal_moves_generated": 0,
        "max_branching_factor": 0,

        # Alpha-beta pruning
        "alpha_beta_cutoffs": 0,
        "children_pruned": 0,
        "cutoff_move_index_sum": 0,
        "cutoff_move_index_max": 0,
        "timed_out": False,

        # Root move information
        "root_moves_started": 0,
        "root_legal_moves": 0,
        "root_moves_considered": 0,
        "root_move_scores": [],
        "best_move": None,
        "best_value": None,
        "immediate_win_found": False,
    }

# Calc stats for when an IDS iteration finishes or times out
def finalize_iteration_stats(stats, iteration_start):
    stats["elapsed_s"] = round(time.perf_counter() - iteration_start, 6)

    if stats["expanded_nodes"] > 0:
        stats["average_branching_factor"] = round(stats["legal_moves_generated"] / stats["expanded_nodes"], 3)
    else:
        stats["average_branching_factor"] = 0.0

    if stats["alpha_beta_cutoffs"] > 0:
        stats["average_cutoff_move_index"] = round(stats["cutoff_move_index_sum"] / stats["alpha_beta_cutoffs"], 3)
    else:
        stats["average_cutoff_move_index"] = None

    if stats["legal_moves_generated"] > 0:
        stats["pruned_child_fraction"] = round(stats["children_pruned"] / stats["legal_moves_generated"], 3)
    else:
        stats["pruned_child_fraction"] = 0.0


@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"
        self.turn_count = 0

        # TODO: Take advantage of this 30 seconds to set up agent

    def step(self, board, reserves, player, opponent):
        """Return the Move you want to play.

        Takes: board, reserves - your own copies of the current position.
               player, opponent - your player number (1 or 2) and the other one.
        Agent.step in agents/agent.py explains each argument and the time limit.
        """
        start = time.perf_counter()
        # ALLOW_MOVES and WIN_LENGTH (from constants) are the rule flags of this game.
        # helpers.get_valid_moves / check_winner already respect them; your own
        # move generator and evaluation function must too.
        # Keep (time.perf_counter() - start) < STEP_TIME_LIMIT - margin.

        root_ply = 2 * self.turn_count + player - 1
        self.turn_count += 1

        deadline = start + STEP_TIME_LIMIT - MARGIN
        def check_time():
            if time.perf_counter() >= deadline: raise SearchTimeout()

        def evaluate(board, reserves, player, opponent, current):
            check_time()
            owners, sizes = top_view(board)

            def biggest_piece(player):
                """Return size of the biggest piece <player> can still place."""
                usable = [size for size in (SMALL, MEDIUM, LARGE)
                          if reserves[player, size] > 0]
                if ALLOW_MOVES: # movable top pieces on board
                    usable += sizes[owners == player].tolist()
                return max(usable, default=-1) # -1 if no valid moves

            biggest_pieces = {player: biggest_piece(player), opponent: biggest_piece(opponent)}

            def line_score(player, opponent, cells):
                """Value of a line for player."""
                value = 0
                count = 0
                for owner, size in cells:
                    if owner == opponent and size >= biggest_pieces[player]:
                        return 0, count # opp piece that cannot be covered blocks the whole line
                    elif owner == player:
                        # a coverable piece is slightly less valuable
                        value += 1 if size >= biggest_pieces[opponent] else 0.75 # can tweak later
                        count += 1
                return 10 ** value, count

            score = 0
            pThreats = 0
            oThreats = 0
            for line in LINES:
                cells = [(owners[row, col], sizes[row, col]) for row, col in line]
                if all(owner != OBSTACLE for owner, _ in cells):
                    playerScore, playerCount = line_score(player, opponent, cells)
                    oppScore, oppCount = line_score(opponent, player, cells)
                    score += playerScore - oppScore

                    if playerScore > 0 and playerCount == WIN_LENGTH - 1: pThreats += 1
                    if oppScore > 0 and oppCount == WIN_LENGTH - 1: oThreats += 1

            THREAT_SCORE = WIN / 10

            if current == player:
                if pThreats: score += THREAT_SCORE
                elif oThreats >= 2: score -= THREAT_SCORE
            else:
                if oThreats: score -= THREAT_SCORE
                elif pThreats >= 2: score += THREAT_SCORE

            return score

        def alphabeta(board, reserves, player, opponent, depth, alpha, beta, maximizing, ply, stats):
            check_time()

            stats["nodes"] += 1

            depth_key = str(depth)
            stats["nodes_by_remaining_depth"][depth_key] = stats["nodes_by_remaining_depth"].get(depth_key, 0) + 1

            current_player = player if maximizing else opponent

            is_over, winner, _ = check_endgame(board, reserves, current_player, ply)
            if is_over:
                stats["terminal_nodes"] += 1

                if winner == player:
                    stats["terminal_wins"] += 1
                    return WIN + depth
                elif winner == opponent:
                    stats["terminal_losses"] += 1
                    return -WIN - depth
                else:
                    stats["terminal_draws"] += 1
                    return 0

            check_time()
            valid_moves = get_valid_moves(board, reserves, current_player)
            # TODO: Order moves to paximize pruning

            check_time()
            if depth == 0 or not valid_moves:
                stats["evaluation_leaves"] += 1

                if depth == 0:
                    stats["depth_limit_leaves"] += 1
                else:
                    stats["no_move_leaves"] += 1

                return evaluate(board, reserves, player, opponent, current_player)

            # Position is being expanded
            stats["expanded_nodes"] += 1
            stats["legal_moves_generated"] += len(valid_moves)
            stats["max_branching_factor"] = max(stats["max_branching_factor"], len(valid_moves))

            if maximizing:
                for move_index, move in enumerate(valid_moves, start=1):
                    check_time()
                    b, r = copy_state(board, reserves)
                    execute_move(b, r, current_player, move)
                    value = alphabeta(b, r, player, opponent, depth-1, alpha, beta, False, ply + 1, stats)
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        stats["alpha_beta_cutoffs"] += 1
                        stats["children_pruned"] += len(valid_moves) - move_index
                        stats["cutoff_move_index_sum"] += move_index
                        stats["cutoff_move_index_max"] = max(stats["cutoff_move_index_max"], move_index)

                        return beta
                return alpha
            else:
                for move_index, move in enumerate(valid_moves, start=1):
                    check_time()
                    b, r = copy_state(board, reserves)
                    execute_move(b, r, current_player, move)
                    beta = min(beta, alphabeta(b, r, player, opponent, depth - 1, alpha, beta, True, ply + 1, stats))
                    if alpha >= beta:
                        stats["alpha_beta_cutoffs"] += 1
                        stats["children_pruned"] += len(valid_moves) - move_index
                        stats["cutoff_move_index_sum"] += move_index
                        stats["cutoff_move_index_max"] = max(stats["cutoff_move_index_max"], move_index)

                        return alpha
                return beta

        def get_best_move(board, reserves, player, opponent, depth, stats, available_scores):
            best_val, best_move = -np.inf, None
            timed_out = False

            root_moves = get_valid_moves(board, reserves, player)

            root_moves.sort(key=lambda m: available_scores.get(repr(m), (None, -np.inf, depth))[1], reverse=True)

            stats["root_legal_moves"] = len(root_moves)

            for move in root_moves:

                try:
                    check_time()

                    stats["root_moves_started"] += 1

                    b, r = copy_state(board, reserves)
                    execute_move(b, r, player, move)
                    winner = check_winner(b, player)
                    if winner == player:
                        val = WIN + depth
                        stats["immediate_win_found"] = True
                    elif winner == opponent:
                        val = -WIN - depth
                    else:
                        val = alphabeta(b, r, player, opponent, depth-1, -np.inf, np.inf, False, root_ply + 1, stats)

                except SearchTimeout:
                    # This root move wasnt fully evaluated
                    timed_out = True
                    break

                stats["root_moves_considered"] += 1

                # Update this move's latest fully evaluated score.
                available_scores[repr(move)] = (move, val, depth)

                stats["root_move_scores"].append({
                    "move": repr(move),
                    "score": float(val),
                })

                # Only record a move after its entire search has completed
                if val > best_val:
                    best_val, best_move = val, move

                if winner == player:
                    break

            stats["timed_out"] = timed_out
            stats["best_move"] = repr(best_move) if best_move else None
            stats["best_value"] = float(best_val) if best_move else None

            return best_val, best_move, not timed_out

        # IDS where we update the val of each move at each depth
        best_val, best_move = -np.inf, None
        best_move_depth = None

        iterations = []
        completed_depth = 0
        attempted_depth = None
        timed_out = False
        available_scores = {}

        for depth in range(2, MAX_DEPTH + 1):
            iteration_start = time.perf_counter()
            stats = make_iteration_stats(depth)
            attempted_depth = depth

            try:
                check_time()
                val, move, iteration_completed = get_best_move(board, reserves, player, opponent, depth, stats, available_scores)

            except SearchTimeout:
                # Timeout happened before get_best_move could start
                val, move = -np.inf, None
                iteration_completed = False
                stats["timed_out"] = True

            if iteration_completed:
                # Depth completed, always take best move
                best_val, best_move = val, move
                best_move_depth = depth
                completed_depth = depth

                stats["completed"] = True
            else:
                # Depth didnt complete, only use result if better than previous depth's best move
                timed_out = True

                if available_scores:
                    best_move, best_val, best_move_depth = max(available_scores.values(), key=lambda x: x[1])

                stats["completed"] = False

            finalize_iteration_stats(stats, iteration_start)
            iterations.append(stats)

            if not iteration_completed or stats["immediate_win_found"]:
                break

        used_random_fallback = best_move is None
        chosen_move = random_move(board, reserves, player) if used_random_fallback else best_move

        search_record = {
            "event": "SEARCH_STATS",
            "player": int(player),
            "opponent": int(opponent),
            "agent_turn": self.turn_count,
            "root_ply": root_ply,

            "completed_depth": completed_depth,
            "attempted_depth": attempted_depth,
            "chosen_move_depth": best_move_depth,
            "timed_out": timed_out,

            "time_budget_s": round(STEP_TIME_LIMIT - MARGIN, 6),
            "elapsed_s": round(time.perf_counter() - start, 6),

            "chosen_move": repr(chosen_move),
            "chosen_value": float(best_val) if best_move is not None else None,
            "used_random_fallback": used_random_fallback,

            "iterations": iterations
        }

        print("SEARCH_STATS " + json.dumps(search_record), flush=True)

        return chosen_move
