"""
对于环境来讲，接受一个动作，环境就要发生变化，并且返回一个信息集合，但是信息集合是对需要做决策的人来说的，现在至少可以断定一点，刚才是并不要做决策
问题1：每次加了新的agent，都需要修改环境中的参数，在一开始就确定每个agent的类型，state内部就不需要再调整了
问题2：observation如何从state中得到并返回到环境中，环境是如何接受action变量返回一个state的？
问题3：observation中应该包含什么信息使得agent做决策，如何定义一个state的数据类型
[word,word_length,test difficulty, student_spelling, accuracy, completeness, letter_judgement]
问题4：agent 与 state, policy是怎么结合的？
"""


import os
from environment_instance import VocabSpellGame


current_path = os.getcwd()  # get the current path
vocabulary_absolute_path = os.path.join(current_path, 'vocabulary_books', 'CET4', 'Vocab.json')  # get the vocab data path


game = VocabSpellGame(vocabulary_book_path=vocabulary_absolute_path,
                      vocabulary_book_name='CET4',
                      chinese_setting=True,
                      phonetic_setting=True,
                      POS_setting=True,
                      english_setting=True,
                      new_words_number=10,
                      new_selection_method='random',
                      task_selection_method='random')  # initialize game environment

state = game.new_initial_state()  # initialize state 此时知道了词库是什么单词，但是还没有标记好格式
while not state.is_terminal:
    state.apply_action(state.legal_action)
    print(state.session_data)
    for i in range(10):
        state.apply_action(state.legal_action)
        print(state.task)
    break





