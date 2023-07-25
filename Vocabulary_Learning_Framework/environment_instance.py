"""
define game environment
"""
from environment_interface import EnvironmentInterface
from state_instance import VocabSpellState


class VocabSpellGame(EnvironmentInterface):
    def __init__(self,
                 vocabulary_book_path,
                 vocabulary_book_name,
                 chinese_setting,
                 phonetic_setting,
                 POS_setting,
                 english_setting,
                 new_words_number,
                 new_selection_method,
                 task_selection_method):
        super().__init__(vocabulary_book_path,
                         vocabulary_book_name,
                         chinese_setting,
                         phonetic_setting,
                         POS_setting,
                         english_setting,
                         new_words_number,
                         new_selection_method,
                         task_selection_method)

    def new_initial_state(self):
        return VocabSpellState(self.vocab_data, self.new_words_number, self.new_selection_method,
                               self.task_selection_method)

# if __name__ == '__main__':
#     '''testing interface'''
#
#     SessionCollectorPlayer = SessionCollectorPlayer(1, 'piece', vocabulary_book.vocab_data, 20, 'sequential', 0)
#     SessionCollectorPlayer.piece_data()
#
#     print(SessionCollectorPlayer.session_collector)
