"""
according to the excellent student memory, it definitely has a stable answer.
supposedly, the student has a initial accuracy for each session, and gradually forget over time.
1: dataframe格式最好复制循环修改，直接修改会有一系列的bug
2：从一个遗忘记忆开始，每个单词对记忆提升的贡献是不一样的，那么提升的程度和单词的长度有没有关系？
3:总的效果是提高的，但是刚开始每一个单词是下降的
4: 开始循环所有的session，都要计算一次遗忘，每个session遗忘的叠加，最后一个总遗忘曲线
"""
from itertools import chain

import string
import numpy as np
import random
import Levenshtein
import os
import torch
from Spelling_Framework.utils.choose_vocab_book import ReadVocabBook
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

CURRENT_PATH = os.getcwd()  # get the current path
VOCAB_PATH: str = os.path.join(CURRENT_PATH, 'vocabulary_books', 'CET4', 'newVocab.json')  # get the vocab data path
corpus_instance = ReadVocabBook(vocab_book_path=VOCAB_PATH,
                                vocab_book_name='CET4',
                                chinese_setting=False,
                                phonetic_setting=True,
                                POS_setting=False,
                                english_setting=True)
original_corpus = corpus_instance.read_vocab_book()

random.seed(42)
random.shuffle(original_corpus)


def add_position(corpus):
    """add position for each corpus"""
    corpus_with_position = []
    for pair in corpus:
        phonemes_position = ''
        letters_position = ''
        pair_position = []
        phonemes_list = pair[0].split(' ')
        for index, phoneme in enumerate(phonemes_list):
            phoneme_index = phoneme + '_' + str(index)
            phonemes_position = phonemes_position + phoneme_index + ' '
        letters_list = pair[1].split(' ')
        for index, letter in enumerate(letters_list):
            letter_index = letter + '_' + str(index)
            letters_position = letters_position + letter_index + ' '
        pair_position.append(phonemes_position.strip())
        pair_position.append(letters_position.strip())
        corpus_with_position.append(pair_position)
    return corpus_with_position


pos_corpus = add_position(original_corpus)  # get the training data [phonemes, word]
# divide into sessions
vocabulary_sessions = []
session_size = 50

for i in range(0, len(pos_corpus), session_size):
    vocabulary_session = pos_corpus[i:i + session_size]
    vocabulary_sessions.append(vocabulary_session)


def forget_process(unique_phonemes, excellent_dataframe, noise_dataframe, excel_ratio, random_ratio):
    """直接改变一个session的记忆"""
    excellent_dataframe_copy = excellent_dataframe.copy()
    for pho in unique_phonemes:
        excellent_dataframe_copy.loc[pho] = excel_ratio * excellent_dataframe.loc[pho] + random_ratio * \
                                            noise_dataframe.loc[pho]  # 计算修改的记忆行
    result_df = excellent_dataframe_copy.div(excellent_dataframe_copy.sum(axis=1), axis=0)  # normalize
    return result_df


def learn(unique_phonemes, forget_dataframe, excellent_dataframe, learning_rate=0.01):
    """ this function aims to enhance memory, the larger the learning rate, the better the retention"""
    forget_dataframe_copy = forget_dataframe.copy()
    for pho in unique_phonemes:
        forget_dataframe_copy.loc[pho] = forget_dataframe.loc[pho] + learning_rate * excellent_dataframe.loc[
            pho]  # 修改记忆
    # 归一化处理
    result_df = forget_dataframe_copy.div(forget_dataframe_copy.sum(axis=1), axis=0)
    return result_df


def generate_answer(memory_df, test_corpus):
    """ generate answer based on the given phonemes,而且我要知道答案的长度，然后根据所有的音标对每一个位置选择最大值"""
    student_answer_pair = []
    random_student_answer_pair = []  # 记录随机拼写的准确度
    test_corpus = [[item.split() for item in sublist] for sublist in test_corpus]
    for phonemes, answer in test_corpus:
        random_spelling = []
        spelling = []
        answer_length = len(answer)
        alphabet = string.ascii_lowercase
        for i in range(answer_length):
            # 将26个字母和位置结合起来，组成列索引
            if i == 0:
                result_columns = [al + '_' + str(i) for al in alphabet]
                possible_results = memory_df.loc[phonemes[0], result_columns]
                letter = possible_results.idxmax()
            else:
                result_columns = [al + '_' + str(i) for al in alphabet]
                possible_results = memory_df.loc[phonemes, result_columns]
                letters_prob = possible_results.sum(axis=0)  # 每一列相加,取概率最大值
                letter = letters_prob.idxmax()
            random_letter = random.choice(string.ascii_lowercase) + '_' + str(i)
            spelling.append(letter)
            random_spelling.append(random_letter)

        random_student_answer_pair.append([random_spelling, answer])
        student_answer_pair.append([spelling, answer])

    return student_answer_pair, random_student_answer_pair


def evaluation(answer_pair):
    accuracy = []
    for stu_answer, correct_answer in answer_pair:
        stu_answer = ''.join([i.split('_')[0] for i in stu_answer])
        correct_answer = ''.join([i.split('_')[0] for i in correct_answer])
        word_accuracy = round(Levenshtein.ratio(correct_answer, stu_answer), 2)
        accuracy.append(word_accuracy)
    avg_accuracy = sum(accuracy) / len(accuracy)
    return avg_accuracy

# excel_start=0.9, excel_end=0.4, noise_start=0.1, noise_end=1, decay_rate=2
def forgetting_parameters(timing_steps, excel_start=0.9, excel_end=0.4, noise_start=0.1, noise_end=1, decay_rate=2):
    """the function aims to simulate the forgetting curve, adjust the parameter can change the slope of forgetting curve"""
    timing_points = np.linspace(0, 1, timing_steps)
    excel_list = (excel_start - excel_end) * np.exp(-decay_rate * timing_points) + excel_end
    noise_list = (noise_end - noise_start) * (1 - np.exp(-decay_rate * timing_points)) + noise_start
    return excel_list, noise_list


if __name__ == '__main__':
    # the excellent memory matrix
    excellent_memory_df = pd.read_excel('agent_RL/excellent_memory.xlsx', index_col=0,
                                        header=0)  # excellent students memory

    # aims to initialize the noise matrix
    dataframe_tensor = torch.tensor(excellent_memory_df.values, dtype=torch.float32)
    noise = torch.randn_like(dataframe_tensor)  # fixed noise
    scaled_noise = (noise - noise.min()) / (noise.max() - noise.min())
    scaled_noise_df = pd.DataFrame(scaled_noise.numpy(), index=excellent_memory_df.index,
                                   columns=excellent_memory_df.columns)

    steps = len(vocabulary_sessions)  # the number of sessions
    # steps = 4  # the number of sessions

    # the number of sessions is 65
    excel_rate, noise_rate = forgetting_parameters(steps)  # get the ratio of excel and noise over time

    # initialize parameters
    excel_acc_list = []
    random_acc_list = []
    learn_acc_list = []
    forget_acc_list = []

    # iterate each session
    for time in range(1, steps + 1):
        # get the session over time
        selected_sessions = vocabulary_sessions[:time]
        flattened_selected_sessions = list(chain.from_iterable(selected_sessions))  # flatten the corpus
        # 1. calculate the accuracy of excellent memory and random memory
        # excellent_answer_pair, random_answer_pair = generate_answer(excellent_memory_df, flattened_selected_sessions)
        # excel_acc = evaluation(excellent_answer_pair)
        # random_acc = evaluation(random_answer_pair)
        # excel_acc_list.append(excel_acc)
        # random_acc_list.append(random_acc)

        # 2. calculate the forgetting curve
        phonemes_list = []
        for task in flattened_selected_sessions:
            for phoneme in task[0].split(' '):
                phonemes_list.append(phoneme)
        phoneme_set = set(phonemes_list)
        unique_phoneme_list = list(phoneme_set)  # get the unique phoneme of selected session
        # add counterpart session noise
        forget_memory_df = forget_process(unique_phoneme_list, excellent_memory_df, scaled_noise_df,
                                          excel_rate[time - 1], noise_rate[time - 1])
        forget_answer_pair, _ = generate_answer(forget_memory_df, flattened_selected_sessions)
        forget_acc = evaluation(forget_answer_pair)
        forget_acc_list.append(forget_acc)

        # 3. calculate the learning curve (the whole memory)
        # learn_memory_df = learn(unique_phoneme_list, forget_memory_df, excellent_memory_df)
        # learn_answer_pair, _ = generate_answer(learn_memory_df, flattened_selected_sessions)
        # learn_acc = evaluation(learn_answer_pair)
        # learn_acc_list.append(learn_acc)
        # print(f'session {time} excel accuracy is {excel_acc}...forget accuracy is {forget_acc}..'
        #       f'learn accuracy is {learn_acc}....the random accuracy is {random_acc}')

        # 4. 探索每一个单词对准确度的影响
        # phoneme_length_list = []
        # per_word_acc_list = []
        #
        # for task in flattened_selected_sessions:  # 得到每一个corpus
        #     unique_phonemes = task[0].split(' ')  # get the phonemes
        #     phoneme_length_list.append(len(unique_phonemes))
        #     # 修改某一个单词的记忆
        #     per_word_memory = learn(unique_phonemes, forget_memory_df, excellent_memory_df)
        #     # 用这个单词测试整个记忆库
        #     per_word_answer_pair, _ = generate_answer(per_word_memory, flattened_selected_sessions)
        #     per_word_acc = evaluation(per_word_answer_pair)  # 单词按照顺序记忆对准确度的影响
        #     per_word_acc_list.append(per_word_acc)
        # plt.figure()
        # plt.scatter(phoneme_length_list, [i - forget_acc for i in per_word_acc_list], label='accuracy of each word')
        # plt.title('the relationship between length of phonemes and accuracy improvement for each word', fontsize=8)
        # plt.xlabel('phoneme_length')
        # plt.ylabel('average accuracy')
        # plt.legend()
        # plt.show()

        # 5：单词顺序对记忆的影响， 学习一个单词更新一次记忆，记忆要保留，探索单词学习的累积效应
        num_of_sequence = 3
        cumulative_word_memory = forget_memory_df.copy()  # 复制这一轮的遗忘记忆
        cumulative_acc_list = []
        for i in range(3):
            cumulative_accuracy = []
            shuffled_corpus = random.sample(flattened_selected_sessions, len(flattened_selected_sessions))
            for task in shuffled_corpus:  # 得到每一个corpus
                unique_phonemes = task[0].split(' ')
                # 修改某一个单词的记忆
                cumulative_word_memory = learn(unique_phonemes, cumulative_word_memory, excellent_memory_df)
                # 用这个单词测试整个记忆库
                cumulative_answer_pair, _ = generate_answer(cumulative_word_memory, shuffled_corpus)
                cumulative_acc = evaluation(cumulative_answer_pair)  # 单词按照顺序记忆对准确度的影响
                cumulative_accuracy.append(round(cumulative_acc, 3))
            cumulative_acc_list.append(cumulative_accuracy)

        fig, ax = plt.subplots()
        # 循环遍历二维列表并绘制折线图
        for sequence in cumulative_acc_list:
            ax.plot([i for i in range(len(flattened_selected_sessions))], sequence)
        # 添加标题和坐标轴标签
        ax.set_xlabel('word numbers')
        ax.set_ylabel('cumulative accuracy')
        ax.set_title('cumulative accuracy with the number of learned words')
        # 显示图形
        plt.show()

    # graph for question 1, 2, 3
    plt.figure()
    plt.plot([i for i in range(1, steps + 1)], excel_acc_list, label='excel accuracy')
    plt.plot([i for i in range(1, steps + 1)], forget_acc_list, label='forget accuracy')
    plt.plot([i for i in range(1, steps + 1)], learn_acc_list, label='learn accuracy')
    plt.plot([i for i in range(1, steps + 1)], random_acc_list, label='random accuracy')
    plt.xlabel('sessions')
    plt.ylabel('average accuracy')
    plt.legend()
    plt.show()





