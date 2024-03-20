"""
the interaction of env and agents
通过不停的加遗忘，可以collect每一个单词在多种情况下的反馈信息，可以当作一个信息收集器
问题是我准备了一个top 50的单词，如何确定这50个单词是学生急需学习的单词？

# 先实现random的图像，因为最终肯定是要对比的

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
                     history_words_number=50,
                     review_words_number=10,
                     sessions_number=60,
                     )  # initialize game environment

# instance agents
agents = [SessionCollectorPlayer(0, 'session_player', 'random'), PresentWordPlayer(1, 'present_player', 'sequential'),
          StudentPlayer(2, 'stu_player', 'forget'), ExaminerPlayer(3, 'examiner_player')]


time_step = env.reset()  # initialize state
# print(time_step)
while not time_step.last():  # not terminate
    player_id = time_step.observations["current_player"]  # current player
    agent_output = agents[player_id].step(time_step)  # action
    time_step = env.step(agent_output)  # current TimeStep
    # print(time_step.observations["vocab_sessions_num"])
# print(time_step.observations["history"])
# # # 把这个数据保存为json文件
# with open('agent_RL/history_information.pkl', 'wb') as pkl_file:
#     pickle.dump(time_step.observations["history"], pkl_file)
# print(time_step.observations['history'])
