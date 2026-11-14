"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view)
from constants import *
import numpy as np
import time

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
        # TODO: replace with your search. Keep (time.perf_counter() - start) < STEP_TIME_LIMIT - margin.

        def evaluate(board, reserves, player, opponent):

            # TODO: implement the evaluation function. Make sure to updade call in alphabeta when completed.

            return 0

        def alphabeta(board, reserves, player, opponent, depth, alpha, beta, maximizing):

            current_player = player if maximizing else opponent
            valid_moves = get_valid_moves(board, reserves, current_player)
            
            if depth == 0 or not valid_moves:
                return evaluate(board, reserves, player, opponent)
            
            if maximizing:
                for move in valid_moves:
                    b, r = copy_state(board, reserves)
                    execute_move(b, r, current_player, move)
                    alpha = max(alpha, alphabeta(b, r, player, opponent, depth - 1, alpha, beta, False))
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
                val = alphabeta(b, r, player, opponent, depth - 1, -np.inf, np.inf, False)
                if val > best_val:
                    best_val, best_move = val, move
            return best_move

        return random_move(board, reserves, player)
