"""
define word maker game environment
"""
from environment_interface import Environment
from word_maker_state import WordMakerState


class WordMakerGame(Environment):
    def __init__(self, tasks, total_game_round):
        super().__init__(tasks, total_game_round)

    def new_initial_state(self):
        return WordMakerState(self._tasks_pool, self._total_game_round)


