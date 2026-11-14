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

        def alphabeta(board, reserves, player_to_move, depth, alpha, beta, maximizing):
            return 0

        def get_best_move(board, reserves, player, opponent):
            best_val, best_move = -np.inf, None
            for move in get_valid_moves(board, reserves, player):
                execute_move(board, reserves, player, move)
                val = alphabeta(board, reserves, opponent, 10, -np.inf, np.inf, False)
                if val > best_val:
                    best_val, move = val, move
            return move

        return random_move(board, reserves, player)
