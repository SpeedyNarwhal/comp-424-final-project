"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view)
from constants import *
import numpy as np
import time

MAX_DEPTH = 10
WIN = 10 ** 6   # score heuristic value

MARGIN = 0.2    # TODO: establish a good margin (do this at the end)
class SearchTimeout(Exception): pass

LINES = ([[(i, j) for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)] + # 4 rows
        [[(j, i) for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)] + # 4 cols
        [[(i, i) for i in range(BOARD_SIZE)],
         [(i, BOARD_SIZE-1-i) for i in range(BOARD_SIZE)]]) # 2 diagonals

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

        # TODO: currently does not consider piece sizes & reserves
        def evaluate(board, reserves, player, opponent):
            check_time()

            owners, sizes = top_view(board)
            score = 0

            for line in LINES:
                cells = [owners[row, col] for row, col in line]
                if OBSTACLE in cells:
                    continue
                mine, opp = cells.count(player), cells.count(opponent)
                if mine and not opp:
                    score += 10 ** mine
                elif opp and not mine:
                    score -= 10 ** opp
            return score

        def alphabeta(board, reserves, player, opponent, depth, alpha, beta, maximizing, ply=0):
            check_time()

            current_player = player if maximizing else opponent

            is_over, winner, _ = check_endgame(board, reserves, current_player, ply)
            if is_over:
                if winner == player:
                    return WIN + depth
                elif winner == opponent:
                    return -WIN - depth
                else:
                    return 0

            check_time()
            valid_moves = get_valid_moves(board, reserves, current_player)
            # TODO: Cut moves that are probably not getting chosen. 
            # Maybe sort by evaluation function output value and cut after a certain threshold.

            check_time()
            if depth == 0 or not valid_moves:
                return evaluate(board, reserves, player, opponent)

            if maximizing:
                for move in valid_moves:
                    check_time()
                    b, r = copy_state(board, reserves)
                    execute_move(b, r, current_player, move)
                    value = alphabeta(b, r, player, opponent, depth-1, alpha, beta, False, ply + 1)
                    alpha = max(alpha, value)
                    if alpha >= beta: return beta
                return alpha
            else:
                for move in valid_moves:
                    check_time()
                    b, r = copy_state(board, reserves)
                    execute_move(b, r, current_player, move)
                    beta = min(beta, alphabeta(b, r, player, opponent, depth - 1, alpha, beta, True, ply + 1))
                    if alpha >= beta: return alpha
                return beta

        def get_best_move(board, reserves, player, opponent, depth):
            best_val, best_move = -np.inf, None
            for move in get_valid_moves(board, reserves, player):
                # TODO: Cut moves that are probably not getting chosen. 
                # Maybe sort by evaluation function output value and cut after a certain threshold.
                
                check_time()

                b, r = copy_state(board, reserves)
                execute_move(b, r, player, move)
                winner = check_winner(b, player)
                if winner == player:
                    return WIN + depth, move
                elif winner == opponent:
                    val = -WIN - depth
                else:
                    val = alphabeta(b, r, player, opponent, depth-1, -np.inf, np.inf, False, root_ply + 1)
                if val > best_val:
                    best_val, best_move = val, move
            return best_val, best_move

        # TODO: Want the val of a move to be the deepest (a move can look good at depth 1 but bad at depth 3)
        best_val, best_move = -np.inf, None
        try:
            for depth in range(2, MAX_DEPTH + 1):
                check_time()
                val, move = get_best_move(board, reserves, player, opponent, depth)
                if val > best_val: best_val, best_move = val, move
        except SearchTimeout:
            pass

        return best_move if best_move else random_move(board, reserves, player)
