from state_interface import StateInterface
from state_interface import _PLAYER_ACTION


class VocabSpellState(StateInterface):
    def __init__(self, vocab_data):
        super().__init__(vocab_data)

    def apply_action(self, action: str):
        if action == 'session_collect':
            self._session_data = self._session_player.session_collector(self._current_session)
            self._player_action = _PLAYER_ACTION('session_player', 'session_collect')
            self._current_session += 1
        return self._player_action
