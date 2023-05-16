"""
以下是为了得到汉语的向量，以及音标的向量，作为神经网络的输入

"""


from gensim.models import Word2Vec
from nltk.corpus import cmudict
import numpy as np


# Load the CMUDict pronunciation dictionary
pronunciation_dict = cmudict.dict()

# Create a list of sentences (corpus)
sentences = [["行动"], ["目标"], ["禁止"], ["干的"], ["有趣"], ["墨水"], ["快乐"], ["网"], ["缝"], ["最高"]]
word_pool = [["act"], ["aim"], ["ban"], ["dry"], ["fun"], ["ink"], ["joy"], ["net"], ["sew"], ["max"]]

cmu_to_ipa = {'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd',
              'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i',
              'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'r',
              'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z',
              'ZH': 'ʒ'
              }

# ---------------------------------------训练汉语的向量表示---------------------------------------------
# Train a Word2Vec model on the corpus 这个是表示汉语的
# model = Word2Vec(sentences, vector_size=10, min_count=1)
# model.save("chinese_trained_vector.bin")
# # Get the word vector for a specific word
# for word in sentences:
#     word_vector = model.wv[word]
#     print(f"Word vector for '{word}': {word_vector}")

# ---------------------------------------训练音标的向量表示---------------------------------------------
# phonetic_sentences = [["p"], ["b"], ["t"], ["d"], ["k"], ["g"], ["ʔ"], ["f"], ["v"], ["θ"], ["ð"], ["s"], ["z"], ["ʃ"],
#                       ["ʒ"], ["h"], ["m"], ["n"], ["ŋ"], ["l"], ["r"], ["j"], ["w"], ["i"], ["ɪ"], ["e"], ["ɛ"], ["æ"],
#                       ["a"], ["ʌ"], ["ɑ"], ["ɔ"], ["o"], ["ʊ"], ["u"], ["ə"], ["ɚ"], ["ɾ"], ["tʃ"], ["dʒ"], ["ɑɪ"],
#                       ["ɔɪ"], ["ɑʊ"], ["ju"]]
#
# model = Word2Vec(phonetic_sentences, vector_size=10, min_count=1)
# model.save("phonetic_trained_vector.bin") # 这里已经保存了模型了,再次运行也可以保存模型
# # Access the vector representation of a phonetic symbol 这个是表示音标的
# for phonetic in phonetic_sentences:
#     phonetic_vector = model.wv[phonetic]
#     print(f"Vector representation of '{phonetic}': {phonetic_vector}")


# ---------------------------------------以下是为了得到单词的音标---------------------------------------------
def remove_stress(pronunciation):
    if pronunciation[-1].isdigit():
        return pronunciation[:-1]
    else:
        return pronunciation


# Define a function to get the phonetic components of a word
def get_phonetic_components(word):
    phonetic_components = []
    if word.lower() in pronunciation_dict:
        phonemes = pronunciation_dict[word.lower()][0]
        for phoneme in phonemes:
            phoneme_without_stress = remove_stress(phoneme)
            phoneme_without_stress = cmu_to_ipa[phoneme_without_stress]
            phonetic_components.append(phoneme_without_stress.split(" ")[0])
    return phonetic_components


# Load the pre-trained word2vec model 得到音标的向量
word = "cat"
phonetic_model = Word2Vec.load("phonetic_trained_vector.bin")
phonetics = get_phonetic_components(word)
phonetic_vector = np.array([phonetic_model.wv.get_vector(phonetic) for phonetic in phonetics])
phonetic_one_D_vector = phonetic_vector.flatten()
print(phonetic_one_D_vector)


# 得到中文的向量
chinese_model = Word2Vec.load("chinese_trained_vector.bin")
chinese = "行动"
chinese_vector = chinese_model.wv.get_vector(chinese)
print(chinese_vector)

# 得到神经网络的输入
NN_input = np.concatenate((phonetic_one_D_vector, chinese_vector))
print(NN_input)


