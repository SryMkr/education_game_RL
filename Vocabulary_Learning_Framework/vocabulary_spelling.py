import os
from environment_instance import VocabSpellGame


current_path = os.getcwd()  # get the current path
vocabulary_absolute_path = os.path.join(current_path, 'vocabulary_books', 'CET4', 'Vocab.json')  # get the vocab data path


game = VocabSpellGame(vocabulary_absolute_path, 'CET4', True, True, False)  # initialize game environment
state = game.new_initial_state()  # initialize state 此时知道了词库是什么单词，但是还没有标记好格式
while not state.is_terminal:
    print(state.current_player)
    print(state.legal_action)
    print(state.current_session)
    print(state.apply_action(state.legal_action))
    print(state.session_data)


