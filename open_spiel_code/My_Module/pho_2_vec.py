"""
该文件是将音标转化为向量
# 以下的代码是为了下载字典，必须提前下载才能使用
import nltk
import ssl

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Download the CMU Pronouncing Dictionary
nltk.download('cmudict')

IPA_PHONEMES = [["i:"], ["i"], ["e"], ["æ"], ["ʌ"], ["ə:"], ["ə"], ["u:"], ["u"], ["ɔ:"], ["ɔ"], ["a:"], ["ei"],
# ["ai"], ["ɔi"], ["j"], ["əu"], ["au"], ["iə"], ["εə"], ["uə"], ["p"], ["b"], ["t"], ["d"], ["k"], ["g"], ["f"],
# ["ʃ"], ["θ"], ["h"], ["v"], ["z"], ["ʒ"], ["ð"], ["tʃ"], ["tr"], ["ts"], ["dʒ"], ["dr"], ["dz"], ["m"], ["n"],
# ["ŋ"], ["l"], ["r"], ["w"]]
"""

from nltk.corpus import cmudict
from gensim.models import Word2Vec
import numpy as np

# ---------------------------------------训练音标的向量表示---------------------------------------------

# 因为CMU 字典可以表示大部分单词的音标，且个数固定，所以最终采用CMU字典的音标
CMU_DICT_PHONEMES = [['P'], ['B'], ['T'], ['D'], ['K'], ['G'], ['CH'], ['JH'], ['F'], ['V'], ['TH'], ['DH'], ['S'],
                     ['Z'], ['SH'], ['ZH'], ['HH'], ['M'], ['N'], ['NG'], ['L'], ['R'], ['W'], ['Y'], ['AA'], ['AE'],
                     ['AH'], ['AO'], ['AW'], ['AY'], ['EH'], ['ER'], ['EY'], ['IH'], ['IY'], ['OW'], ['OY'], ['UH'],
                     ['UW']]
CMU_TO_IPA = {'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd',
              'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i',
              'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p',
              'R': 'r', 'S': 's', 'SH': 'ʃ',
              'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'}
# model = Word2Vec(CMU_DICT_PHONEMES, vector_size=50, min_count=1)
# model.save("phonetic_trained_vector.bin")  # 这里已经保存了模型了,再次运行也可以保存模型
# Access the vector representation of a phonetic symbol 这个是表示音标的
# 底下代码是打印输出结果
# for phonetic in CMU_DICT_PHONEMES:
#     phonetic_vector = model.wv[phonetic]
#     print(f"Vector representation of '{phonetic}': {phonetic_vector}")


