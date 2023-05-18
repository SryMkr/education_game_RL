from gensim.models import Word2Vec
from nltk.corpus import cmudict
import numpy as np

# Create a list of sentences (corpus)
chinese = [["行动"], ["目标"], ["禁止"], ["有趣"], ["墨水"], ["快乐"], ["网"], ["缝"], ["最高"]]
word_pool = [["act"], ["aim"], ["ban"], ["fun"], ["ink"], ["joy"], ["net"], ["sew"], ["max"]]
chinese_english = {"行动": "act", "目标": "aim", "禁止": "ban", "有趣": "fun", "墨水": "ink", "快乐": "joy",
                   "网": "net", "缝": "sew", "最高": "max"}

CMU_DICT_PHONEMES = [['P'], ['B'], ['T'], ['D'], ['K'], ['G'], ['CH'], ['JH'], ['F'], ['V'], ['TH'], ['DH'], ['S'],
                     ['Z'], ['SH'], ['ZH'], ['HH'], ['M'], ['N'], ['NG'], ['L'], ['R'], ['W'], ['Y'], ['AA'], ['AE'],
                     ['AH'], ['AO'], ['AW'], ['AY'], ['EH'], ['ER'], ['EY'], ['IH'], ['IY'], ['OW'], ['OY'], ['UH'],
                     ['UW']]
CMU_TO_IPA = {'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd',
              'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i',
              'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p',
              'R': 'r', 'S': 's', 'SH': 'ʃ',
              'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'}

# chinese_model = Word2Vec.load("split_CN_EN/cet4_chinese_vector.model")  # 得到中文的模型
# for word in sentences:
#     word_vector = model.wv[word]
#     print(f"Word vector for '{word}': {word_vector}")

# ---------------------------------------以下是为了得到单词的音标---------------------------------------------
# Load the CMUDict pronunciation dictionary
pronunciation_dict = cmudict.dict()


# 底下函数是为了移除单词的重音
def remove_stress(pronunciation):
    if pronunciation[-1].isdigit():
        return pronunciation[:-1]
    else:
        return pronunciation


# Define a function to get the phonetic components of a word
def get_phonetic_components(word):
    cmu_phonetic_components = []
    ipa_phonetic_components = []
    if word.lower() in pronunciation_dict:
        phonemes = pronunciation_dict[word.lower()][0]
        for phoneme in phonemes:
            phoneme_without_stress = remove_stress(phoneme)  # to get the cmu phonemes
            ipa_phoneme = CMU_TO_IPA[phoneme_without_stress]  # to get the ipa phonemes
            cmu_phonetic_components.append(phoneme_without_stress.split(" ")[0])
            ipa_phonetic_components.append(ipa_phoneme.split(" ")[0])
    return cmu_phonetic_components, ipa_phonetic_components


cmu_phoneme_model = Word2Vec.load("phonetic2vector/phonetic_trained_vector.bin")  # 加载音标向量模型
chinese_model = Word2Vec.load("split_CN_EN/cet4_chinese_vector.model")  # 得到中文向量的模型
for chinese_c, en in chinese_english.items():
    cmu_phonemes, ipa_phonemes = get_phonetic_components(en)
    # Load the pre-trained word2vec model 得到音标的向量
    cmu_phoneme_vector = np.array([cmu_phoneme_model.wv.get_vector(cmu_phoneme) for cmu_phoneme in cmu_phonemes])
    cmu_phoneme_one_D_vector = cmu_phoneme_vector.flatten()  # 得到一维的音标向量
    print(cmu_phoneme_one_D_vector.shape)
    chinese_vector = chinese_model.wv.get_vector(chinese_c)
    print(chinese_vector.shape)
    # 得到神经网络的输入
    NN_input = np.concatenate((cmu_phoneme_one_D_vector, chinese_vector))
    print(NN_input.shape)
