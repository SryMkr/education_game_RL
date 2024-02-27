"""
1: 曲线至少是下降的，感觉差不多，主要看怎么优化的平滑一点而且快一点
  先快收慢要模拟出来，感觉主图实在0.99
  其实因为每一次都是叠加，所以其实设置一个固定的比例就好，而且主图的保留程度不能太低 0.9998，噪声的程度也不能太大0.01到0.02经过不断的迭代就
  差不多可以给出一条比较好的遗忘曲线。

2： tensor(0.9998) tensor(0.0179) 这个数值配比比较理想
    tensor(0.9998) tensor(0.0218)
3：α[1,0.5] β[0, 0.9]  α[1,0.4] β[0, 0.1]
"""
from typing import Tuple
import string
import numpy as np
import random
import Levenshtein
import os
import torch
from Spelling_Framework.utils.choose_vocab_book import ReadVocabBook
import pandas as pd
from Spelling_Framework.agents.Position_EM import PositionPhoLetStudent
import matplotlib.pyplot as plt

CURRENT_PATH = os.getcwd()  # get the current path
VOCAB_PATH: str = os.path.join(CURRENT_PATH, 'vocabulary_books', 'CET4', 'newVocab.json')  # get the vocab data path
corpus_instance = ReadVocabBook(vocab_book_path=VOCAB_PATH,
                                vocab_book_name='CET4',
                                chinese_setting=False,
                                phonetic_setting=True,
                                POS_setting=False,
                                english_setting=True)
original_corpus = corpus_instance.read_vocab_book()
random.shuffle(original_corpus)

training_corpus = original_corpus[:int(len(original_corpus) * 0.8)]  # get the training data [phonemes, word]
testing_corpus = original_corpus[int(len(original_corpus) * 0.8):]  # get the testing data

corpus_1 = [[item.split() for item in sublist] for sublist in original_corpus]
letters_1 = []
phonemes_1 = []
for sp in corpus_1:
    for fw in sp[0]:  # phoneme
        phonemes_1.append(fw)
    for ew in sp[1]:  # letters
        letters_1.append(ew)
# covert into lower letter, and omit the duplicated word
letters_set = sorted(list(set(letters_1)), key=lambda s: s.lower())  # 26
phonemes_set = sorted(list(set(phonemes_1)), key=lambda s: s.lower())  # 39
# print(letters_set)
# print(phonemes_set)

df_index = []
df_column = []

# 在这直接构造DF，横坐标和纵坐标都弄上9个，然后初始化
for i in range(10):
    l_p = ''
    p_p = ''
    for l_i in letters_set:
        l_p = l_i + '_' + str(i)
        df_column.append(l_p)
    for p_i in phonemes_set:
        p_p = p_i + '_' + str(i)
        df_index.append(p_p)

""" initialize all prob, all possible included"""
init_prob = 1 / len(df_column)
phoneme_letter_prob = pd.DataFrame(init_prob, index=df_index, columns=df_column)


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


pos_training_corpus = add_position(training_corpus)  # get the training data [phonemes, word]
pos_testing_corpus = add_position(testing_corpus)  # get the testing data


def forward_process_dataframe(dataframe, alphas_bar_sqrt, one_minus_alphas_bar_sqrt):
    dataframe_tensor = torch.tensor(dataframe.values, dtype=torch.float32)
    # noise = torch.randn_like(dataframe_tensor) * 0.012  # 直接改变噪声差不多可以延长遗忘的时间
    noise = torch.randn_like(dataframe_tensor)
    scaled_noise = (noise - noise.min()) / (noise.max() - noise.min())
    print(alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
    x_t = alphas_bar_sqrt * dataframe_tensor + one_minus_alphas_bar_sqrt * scaled_noise
    # 将结果张量转换回DataFrame
    result_df = pd.DataFrame(x_t.numpy(), index=dataframe.index, columns=dataframe.columns)
    # 归一化处理
    result_df = result_df.div(result_df.sum(axis=1), axis=0)
    # print(result_df.sum(axis=1))
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    position_phoLet_student = PositionPhoLetStudent(pos_training_corpus, pos_training_corpus, phoneme_letter_prob,
                                                    df_index, df_column)
    position_phoLet_student.train_model()
    # position_phoLet_student.phoneme_letter_df.to_excel(STU_MEMORY_PATH, engine='openpyxl')
    position_phoLet_student.generate_answer()
    position_phoLet_accuracy, position_phoLet_completeness, position_phoLet_perfect = position_phoLet_student.evaluation()
    print(
        f'position phoLet students accuracy is: {position_phoLet_accuracy}, completeness is: {position_phoLet_completeness}, '
        f'perfect is: {position_phoLet_perfect}')

    steps = 30  # means 30 days
    scaled_steps = np.linspace(0, 1, steps)
    student_memory = position_phoLet_student.phoneme_letter_df

    student_memory = student_memory.div(student_memory.sum(axis=1), axis=0)
    student_memory.to_excel('excellent_memory.xlsx', engine='openpyxl')
    alphas_bar_sqrt_ = 0.5 * np.exp(-2.5 * scaled_steps) + 0.4
    one_minus_alphas_bar_sqrt_ = 0.9 * (1 - np.exp(-2 * scaled_steps)) + 0.1
    print(alphas_bar_sqrt_)
    print(one_minus_alphas_bar_sqrt_)
    avg_acc_list = [position_phoLet_accuracy]
    random_avg_acc_list = [0.15]

    for t in range(1, steps + 1):
        forgetting_student_memory = forward_process_dataframe(student_memory, alphas_bar_sqrt_[t-1], one_minus_alphas_bar_sqrt_[t-1])
        spelling_answer_pair, random_answer_pair = generate_answer(forgetting_student_memory, pos_training_corpus)
        avg_acc = evaluation(spelling_answer_pair)
        random_avg_acc = evaluation(random_answer_pair)
        print(f'epoch: {t}: average accuracy is {avg_acc}........the random accuracy is {random_avg_acc}')
        if avg_acc <= 0.4:
            forgetting_student_memory.to_excel('forgetting_memory.xlsx', engine='openpyxl')
            break
        avg_acc_list.append(avg_acc)
        random_avg_acc_list.append(random_avg_acc)
    plt.figure()
    plt.plot([i for i in range(steps + 1)], avg_acc_list, label='forgetting curve')
    plt.plot([i for i in range(steps + 1)], random_avg_acc_list, label='random')
    plt.xlabel('days')
    plt.ylabel('accuracy')
    plt.show()
