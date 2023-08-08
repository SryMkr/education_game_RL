"""
对于环境来讲，接受一个动作，环境就要发生变化，并且返回一个信息集合，但是信息集合是对需要做决策的人来说的，现在至少可以断定一点，刚才是并不要做决策
问题1：每次加了新的agent，都需要修改环境中的参数，在一开始就确定每个agent的类型，state内部就不需要再调整了
问题3：observation中应该包含什么信息使得agent做决策，如何定义一个state的数据类型
[word,word_length,test difficulty, student_spelling, accuracy, completeness, letter_judgement]
问题4：agent 与 state, policy是怎么结合的？
每一个agent做一个动作，都应该把信息返回到环境中，而下一个agent从环境中获得自己所需要的信息做决策，所以应该在环境中规定好几类observation，
使得不同的agent可以直接调用
"""

import os
from environment_instance import VocabSpellGame
from agents_instance import SessionCollectorPlayer, PresentWordPlayer, StudentPlayer, ExaminerPlayer

current_path = os.getcwd()  # get the current path
vocabulary_absolute_path = os.path.join(current_path, 'vocabulary_books', 'CET4',
                                        'Vocab.json')  # get the vocab data path

env = VocabSpellGame(vocabulary_book_path=vocabulary_absolute_path,
                     vocabulary_book_name='CET4',
                     chinese_setting=True,
                     phonetic_setting=True,
                     POS_setting=True,
                     english_setting=True,
                     new_words_number=10,
                     )  # initialize game environment

# 如何用index指代agents
agents = [SessionCollectorPlayer(0, 'session_player'), PresentWordPlayer(1, 'present_player', 'easy_to_hard'),
          StudentPlayer(2, 'stu_player', 'random'), ExaminerPlayer(3, 'examiner_player')]


time_step = env.reset()  # initialize state 此时知道了词库是什么单词，但是还没有标记好格式
# print(time_step)
while not time_step.last():  # 只要不是最后一步
    player_id = time_step.observations["current_player"]  # 也是个字典，能获得当前的玩家
    print(player_id)
    agent_output = agents[player_id].step(time_step)  # 输入的是state的时间步,问题1，我返回的不是action而是信息！！！！
    print(agent_output)
    time_step = env.step(agent_output)
    print(time_step)
    player_id = time_step.observations["current_player"]  # 也是个字典，能获得当前的玩家
    print(player_id)
    agent_output = agents[player_id].step(time_step)  # 输入的是state的时间步,问题1，我返回的不是action而是信息！！！！
    print(agent_output)
    time_step = env.step(agent_output)
    print(time_step)
    player_id = time_step.observations["current_player"]  # 也是个字典，能获得当前的玩家
    print(player_id)
    agent_output = agents[player_id].step(time_step)  # 输入的是state的时间步,问题1，我返回的不是action而是信息！！！！
    print(agent_output)
    time_step = env.step(agent_output)
    print(time_step)
    player_id = time_step.observations["current_player"]  # 也是个字典，能获得当前的玩家
    print(player_id)
    agent_output = agents[player_id].step(time_step)  # 输入的是state的时间步,问题1，我返回的不是action而是信息！！！！
    print(agent_output)
    time_step = env.step(agent_output)
    print(time_step)
    break

