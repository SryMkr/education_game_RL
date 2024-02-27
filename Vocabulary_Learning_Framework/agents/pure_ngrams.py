"""
1: 没有利用任何别的信息，导致预测得准确度不高
2：会出现没有样本的问题，怎么处理？如果直接将模型用于测试信息全都是没出现过的情况，太小了无法捕捉有用信息，太多了全是没出现的样本，如何改进的问题
3： 训练数据取多少？n设置为多少？
4：如何利用信息？ （中文，） 贝叶斯网络
5：如果条件是独一无二的，那么答案一定是正确的， 但是完全没有泛化能力，因为后面的条件一顶不同
7：音标如何辅助预测？

任务
# 2023年11月23日，自己实现n gram的计算，并且可以实现基本的拼写功能, 保证代码完全正确
1: 如果概率一样，没有全部考虑
2：测试数据依旧会出现没有样本的情况，所以需要平滑
3：如何考虑音标的情况
4：如何平滑频率表和概率表格
5: 发现的问题是概率改变导致condition改变，从而无法找到样本数据
6：没有出现的条件怎么处理？直接加有点太鲁莽，明天考虑吧
7：probs of spelling given the phonemes
# 如何将音标和ngrams结合做预测
纯粹的n-grams最终的效果非常差，根本无法模拟完美学生
"""

import os
import random
from itertools import chain
from typing import List, Dict, Tuple
from Spelling_Framework.utils.choose_vocab_book import ReadVocabBook
from collections import Counter
from nltk.tokenize import word_tokenize
import Levenshtein
import pandas as pd

# ----------------------------------------------read vocabulary book---------------------------------------------------------------
_CURRENT_PATH: str = os.getcwd()  # get the current path

_vocab_book_abs_path: str = os.path.join(_CURRENT_PATH, 'vocabulary_books', 'CET4', 'Vocab.json')  # get the vocab data path

# initialize vocabulary instance
_vocab_instance = ReadVocabBook(vocab_book_path=_vocab_book_abs_path,
                                vocab_book_name='CET4',
                                chinese_setting=False,
                                phonetic_setting=True,
                                POS_setting=False,
                                english_setting=True)

# read vocab data [[phoneme, english]......] [['j ɔ r', 'y o u r'], ['j ɝ s ɛ l f', 'y o u r s e l f']......]
_VOCAB_DATA: List[List[str]] = _vocab_instance.read_vocab_book()
print(_VOCAB_DATA)
random.shuffle(_VOCAB_DATA)  # shuffle the vocabulary book


# ----------------------------------------------train model-------------------------------------------------------------

# nltk.download('punkt') # down the necessary file for tokenize

def generate_ngrams(tokens, n: int):
    """ generate n-grams with start, end and pad symbol"""
    BOS = ['<bos>']  # begin of sentence
    EOS = ['<eos>']  # end of sentence
    tokens = BOS + tokens + EOS  # concatenate the start symbol, token, end symbol -> new tokens
    n_grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]  # generate grams
    conditions = [tuple(tokens[i:i + n - 1]) for i in range(len(tokens) - n + 1)]  # generate conditions
    return tokens, n_grams, conditions


def construct_dataset(vocab_data, split_ratio, n_length):
    """ construct freq and prob matrix """
    n_grams_tokens_list = []  # store all grams
    conditions_tokens_list = []  # store all conditions
    task_tokens_list = []  # store correct answer
    test_task_tokens_list = []  # store testing data

    training_data = vocab_data[: int(len(vocab_data) * split_ratio)]  # get the training data [phonemes, word]
    testing_data = vocab_data[int(len(vocab_data) * split_ratio):]  # get the testing data

    training_tokens = [[word_tokenize(data[1]), word_tokenize(data[0])] for data in training_data]
    testing_tokens: List[str] = [word_tokenize(data[1]) for data in testing_data]  # get the testing tokens

    for seq_token in testing_tokens:
        test_task_tokens, test_n_grams_tokens, test_conditions_tokens = generate_ngrams(seq_token, n_length)
        test_task_tokens_list.append(test_task_tokens)

    task_phoneme_tokens_list = []
    # separate the phonemes, word
    for seq_token, phoneme_tokens in training_tokens:
        task_tokens, n_grams_tokens, conditions_tokens = generate_ngrams(seq_token, n_length)

        task_phoneme_tokens_list.append([task_tokens, phoneme_tokens])

        task_tokens_list.append(task_tokens)
        n_grams_tokens_list.append(n_grams_tokens)
        conditions_tokens_list.append(conditions_tokens)

    n_grams_list = list(chain(*n_grams_tokens_list))
    conditions_list = list(chain(*conditions_tokens_list))

    conditions_counts = dict(Counter(conditions_list))  # convert tuple into dictionary
    conditions_counts: Dict[Tuple[str], int] = {"_".join(map(str, key)): value for key, value in
                                                conditions_counts.items()}

    # count the n_grams, and conditions
    ngrams_counts = list(Counter(n_grams_list).items())

    # print(phoneme_letter_matrix)
    # create frequency matrix
    condition_letter_freq_df = pd.DataFrame()
    for ngrams_count in ngrams_counts:
        condition_letter_freq_df.loc['_'.join(ngrams_count[0][:-1]), ngrams_count[0][-1]] = ngrams_count[1]
    condition_letter_freq_df = condition_letter_freq_df.fillna(0)  # frequency table

    # create probability table
    condition_letter_prob_df = condition_letter_freq_df.div(conditions_counts, axis=0)
    return task_tokens_list, test_task_tokens_list, task_phoneme_tokens_list, condition_letter_freq_df, condition_letter_prob_df


# generate spelling  给定一个条件，要得到所有可能的拼写及其概率, 找到对应的单词，打印所有的概率
def generate_spelling(model, task_token, condition_length):
    """ generate spelling based on trained model"""
    generated_answer = []  # the generated content via probability matrix
    # 别的不说，肯定是要先拿到音标，任何一个音标都对应了所有字母的可能性的概率，先打印出来
    word_index = 0
    for task in task_token:
        condition = task[:condition_length]  # get the task condition
        # print(phoneme_letter_matrix.iloc[condition])
        # print(condition)
        # bos_token = condition.copy()
        task_length = len(task)
        # control the length of prediction
        for _ in range(task_length - condition_length):
            try:
                condition_window = '_'.join(condition[-condition_length:])
                # 如何给出所有可能的答案？以及如何将音标加进去辅助拼写
                next_letter = model.loc[condition_window].idxmax()  # select the maximum probability
                # print(next_word)
                condition.append(next_letter)
                if next_letter == '<eos>':
                    break
            except KeyError as e:
                print(f"Error: {e}")
                print(f"No samples for '{condition_window}'. Add some samples first.")
        # print(f'{bos_token} Generated Answer Is {" ".join(condition)}, Correct Answer Is {" ".join(task)}')
        generated_answer.append(" ".join(condition))
        word_index += 1

    return generated_answer, task_token


def evaluation_model(student_answer, task_answer):
    """ evaluation according to accuracy and completeness"""
    accuracy_list = []
    completeness_list = []
    perfect_spelling_counts = 0
    for index in range(len(task_answer)):
        stu_spelling = student_answer[index]
        tutor_answer = ' '.join(task_answer[index])
        # calculate Levenshtein accuracy and completeness
        word_accuracy = round(Levenshtein.ratio(tutor_answer, stu_spelling), 2)
        word_completeness = round(1 - Levenshtein.distance(tutor_answer, stu_spelling) / len(tutor_answer), 2)
        accuracy_list.append(word_accuracy)
        completeness_list.append(word_completeness)
        # calculate perfect spelling accuracy
        if stu_spelling == tutor_answer:
            perfect_spelling_counts += 1

    n_avg_accuracy = sum(accuracy_list) / len(accuracy_list)
    n_avg_completeness = sum(completeness_list) / len(completeness_list)
    perfect_spelling_ratio = perfect_spelling_counts / len(accuracy_list)
    return n_avg_accuracy, n_avg_completeness, perfect_spelling_ratio


def draw_n_plots(split_ratio):
    """explore the relationship between n and whole corpus"""
    N_LENGTH = [2, 3]  # the length of 'n', only use 2,3
    n_accuracy = []
    n_completeness = []
    n_perfect_ratio = []
    for n_index in N_LENGTH:
        task_tokens, test_tokens, task_phoneme_list, freq_matrix, prob_matrix = construct_dataset(_VOCAB_DATA, split_ratio, n_index)
        generated_answer, correct_answer = generate_spelling(prob_matrix, task_tokens, n_index - 1)
        n_avg_accuracy, n_avg_completeness, perfect_spelling_ratio = evaluation_model(generated_answer, correct_answer)
        n_accuracy.append(n_avg_accuracy)
        n_completeness.append(n_avg_completeness)
        n_perfect_ratio.append(perfect_spelling_ratio)
    return n_accuracy, n_completeness, n_perfect_ratio


if __name__ == '__main__':
    _split_ratio = [1]  # use the whole corpus
    accuracy_list = []  # store different n accuracy in a specific data size
    completeness_list = []  # store different n completeness in a specific data size
    perfect_ratio_list = []  # store different n perfect in a specific data size
    for ratio in _split_ratio:  # the whole corpus
        avg_accuracy, avg_completeness, n_perfect_ratio = draw_n_plots(ratio)
        print(f'data size:{ratio}, avg_accuracy:{avg_accuracy}, avg_completeness:{avg_completeness}, perfect_ratio:{n_perfect_ratio}')
        accuracy_list.append(avg_accuracy)
        completeness_list.append(avg_completeness)
        perfect_ratio_list.append(n_perfect_ratio)