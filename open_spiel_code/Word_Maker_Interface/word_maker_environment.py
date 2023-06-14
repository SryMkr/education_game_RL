"""
define word maker game environment
"""
from environment_interface import Environment
from typing import List, Tuple, Dict
from word_maker_state import WordMakerState


class WordMakerGame(Environment):
    def __init__(self, tasks: Dict[str, str], total_game_round: int):
        super().__init__(tasks, total_game_round)

    def new_initial_state(self):
        return WordMakerState(self._tasks_pool, self._total_game_round)


