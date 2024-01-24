"""
the interaction of env and agents
"""

import os
from environment_instance import VocabSpellGame
from agents_instance import SessionCollectorPlayer, PresentWordPlayer, StudentPlayer, ExaminerPlayer


current_path = os.getcwd()  # get the current path
vocabulary_absolute_path = os.path.join(current_path, 'vocabulary_books', 'CET4', 'newVocab.json')  # get the vocab data path

env = VocabSpellGame(vocabulary_book_path=vocabulary_absolute_path,
                     vocabulary_book_name='CET4',
                     chinese_setting=False,
                     phonetic_setting=True,
                     POS_setting=False,
                     english_setting=True,
                     new_words_number=10,
                     )  # initialize game environment

# instance agents
agents = [SessionCollectorPlayer(0, 'session_player', 'random'), PresentWordPlayer(1, 'present_player', 'sequential'),
          StudentPlayer(2, 'stu_player', 'excellent'), ExaminerPlayer(3, 'examiner_player')]


time_step = env.reset()  # initialize state 此时知道了词库是什么单词，但是还没有标记好格式
# print(time_step)
while not time_step.last():  # not terminate
    player_id = time_step.observations["current_player"]  # current player
    # print(player_id)
    agent_output = agents[player_id].step(time_step)  # action
    # print(agent_output)
    time_step = env.step(agent_output)  # current TimeStep


# print(time_step.observations['history'])
