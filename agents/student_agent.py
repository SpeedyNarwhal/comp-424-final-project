"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view)
from constants import *
import numpy as np
import time

DEPTH = 3       # TODO: implement actual incrementing
WIN = 10 ** 6   # score heuristic value

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

        # TODO: currently does not consider piece sizes & reserves
        def evaluate(board, reserves, player, opponent):
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

        def alphabeta(board, reserves, player, opponent, depth, alpha, beta, maximizing):

            current_player = player if maximizing else opponent
            valid_moves = get_valid_moves(board, reserves, current_player)

            if depth == 0 or not valid_moves:
                return evaluate(board, reserves, player, opponent)

            if maximizing:
                for move in valid_moves:
                    b, r = copy_state(board, reserves)
                    execute_move(b, r, current_player, move)
                    value = alphabeta(b, r, player, opponent, depth-1, alpha, beta, False)
                    alpha = max(alpha, value)
                    if alpha >= beta: return beta
                return alpha
            else:
                for move in valid_moves:
                    b, r = copy_state(board, reserves)
                    execute_move(b, r, current_player, move)
                    beta = min(beta, alphabeta(b, r, player, opponent, depth - 1, alpha, beta, True))
                    if alpha >= beta: return alpha
                return beta

        def get_best_move(board, reserves, player, opponent, depth):
            best_val, best_move = -np.inf, None
            for move in get_valid_moves(board, reserves, player):
                b, r = copy_state(board, reserves)
                execute_move(b, r, player, move)
                winner = check_winner(b, player)
                if winner == player:
                    return move
                elif winner == opponent:
                    val = -WIN - depth
                else:
                    val = alphabeta(b, r, player, opponent, depth-1, -np.inf, np.inf, False)
                if val > best_val:
                    best_val, best_move = val, move
            return best_move

        move = get_best_move(board, reserves, player, opponent, DEPTH)
        return move if move else random_move(board, reserves, player)
