"""
对文件进行预处理 将txt文本分割为纯中文和纯英文，主要使用的是正则表达式
将所有的中英文一一对应，生成所有的中英文对象向量表，到此神经网络的输入得到了
"""
import json
import re
import numpy as np
from word_two_vector.word2vector import get_phonetic_components
# -----------------------------预处理文本，得到纯中英文没有符号---------------
#  使用正则表达式对数据进行预处理,并将数据保存为以空格分割的字符串
# def preprocess_text(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:  # 以UTF-8的方式打开
#         contents_of_txt = ""
#         for line in file:  # 一行一行的读，一行就是文件中的一行
#             line = line.strip()  # 每一行都去头去尾换行符，或者空格
#             line = re.split("[ ,，;；.&]", line)  # 中英文字符都需要加进去分割内容
#             line = " ".join(line) + '\n'  # 将每一行的字符串按照空格链接，且换行
#             if line:
#                 contents_of_txt += line
#         # 返回所有的字符串
#         return contents_of_txt
#
#
# def save_text(file_path, contents_of_txt):
#     with open(file_path, 'w', encoding='utf-8') as file:  # 以UTF-8的方式打开
#         for line in contents_of_txt:  # 读取每一行
#             file.write(line)  # 保存每一行
#
#
# input_file_path = 'cet4.txt'  # 输入文件
# output_file_path = 'cet4_chinese_EN.txt'  # 输出文件
#
#
# chinese_text = preprocess_text(input_file_path)  # 执行预处理操作
# save_text(output_file_path, chinese_text)  # 执行保存文件操作

# # -----------------------------以下代码是为了去除文件中的非中文字，得到中文---------------
# print('主程序执行开始...')
# input_file_name = 'cet4_chinese_EN.txt'
# output_file_name = 'cet4_chinese.txt'
# input_file = open(input_file_name, 'r', encoding='utf-8')
# output_file = open(output_file_name, 'w', encoding='utf-8')
# print('开始读入数据文件...')
# lines = input_file.readlines()
# print('读入数据文件结束！')
# print('分词程序执行开始...')
# count = 1
# cn_reg = '^[\u4e00-\u9fa5]'  # 去除非中文字
# for line in lines:
#     if line != '\n':
#         line = line.strip()
#         line_list = line.split('\n')[0].split(' ')
#         line_list_new = []
#         for word in line_list:
#             if re.search(cn_reg, word):
#                 line_list_new.append(word)
#         output_file.write(' '.join(line_list_new) + '\n')
#         count += 1
# print('分词程序执行结束！')
# print('主程序执行开始...')
#
# # -----------------------------以下代码是为了去除文件中的中文字---------------
# input_file_name = 'cet4_chinese_EN.txt'
# output_file_name = 'cet4_english.txt'
# input_file = open(input_file_name, 'r', encoding='utf-8')
# output_file = open(output_file_name, 'w', encoding='utf-8')
# print('开始读入数据文件...')
# lines = input_file.readlines()
# print('读入数据文件结束！')
# print('分词程序执行开始...')
# count = 1
# cn_reg_en = r'\b[a-zA-Z]+\b|^( | ^|)^⋯'
# cn_reg = r'[\u4e00-\u9fa5]'
# for line in lines:
#     if line != '\n':
#         line = line.strip()
#         line_list = line.split('\n')[0].split(' ')
#         line_list_new = []
#         for word in line_list:
#             if re.search(cn_reg_en, word):
#                 if not re.search(cn_reg, word):
#                     line_list_new.append(word)
#         output_file.write(' '.join(line_list_new) + '\n')
#         count += 1
# print('分词程序执行结束！')
# print('主程序执行开始...')


# -----------------------------以下代码是为了将中文转化为向量,并保存为模型---------------
import multiprocessing
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec

#
# print('主程序开始执行...')
# input_file_name = 'cet4_chinese.txt'
# model_file_name = 'cet4_chinese_vector.model'
# print('转换过程开始...')
# model = Word2Vec(LineSentence(input_file_name),
#                  vector_size=50,  # 词向量长度为400
#                  min_count=1,
#                  workers=multiprocessing.cpu_count())
# print('转换过程结束！')
# print('开始保存模型...')
# model.save(model_file_name)
# print('模型保存结束！')
# print('主程序执行结束！')

# -----------------------------以下代码是为了查看模型的结果，以及数量---------------
# chinese_model = Word2Vec.load("cet4_chinese_vector.model")
# print(chinese_model.wv.index_to_key)
# count = 0
# for word in chinese_model.wv.index_to_key:
#     count += 1
#     print(word, chinese_model.wv[word])
# print(count)

# -----------------------------将中文和英文组合为字典的形式，这样汉语和英文的对应就不用自己找了---------------
# file_path = "cet4_chinese_EN.txt"
# output_cn_en = "cet4_chinese_EN_121.txt"
# output_cn_en_file = open(output_cn_en, 'w', encoding='utf-8')
# word_list = {}
# with open(file_path, 'r', encoding='utf-8') as file:  # 以UTF-8的方式打开
#     for line in file:  # 一行一行的读，一行就是文件中的一行
#         line = line.strip()  # 每一行都去头去尾换行符，或者空格
#         word_split_list = line.split(' ')
#         for chinese in word_split_list[1:]:
#             if '\u4e00' < chinese < '\u9fff':
#                 # cmu_phonemes, ipa_phonemes = get_phonetic_components(line.split(' ')[0])
#                 output_cn_en_file.write(word_split_list[0] + ' ' + chinese + '\n')
#                 break


# --------------------测试所有的单词有没有发音转化为向量，以及汉语可不可以转化为向量,最后生成神经网路的输入---------------
# en_cn_dict = {}
# cmu_phoneme_model = Word2Vec.load("phonetic_trained_vector.bin")  # 加载音标向量模型
# chinese_model = Word2Vec.load("cet4_chinese_vector.model")  # 得到中文向量的模型
# json_dic = {}
# with open("cet4_chinese_EN_121.txt", 'r', encoding='utf-8') as file:  # 以UTF-8的方式打开
#     for line in file:  # 一行一行的读，一行就是文件中的一行
#         line = line.strip()  # 每一行都去头去尾换行符，或者空格
#         word_split_list = line.split(' ')
#         cmu_phonemes, ipa_phonemes = get_phonetic_components(word_split_list[0])
#         # Load the pre-trained word2vec model 得到音标的向量
#         cmu_phoneme_vector = np.array([cmu_phoneme_model.wv.get_vector(cmu_phoneme) for cmu_phoneme in cmu_phonemes])
#         cmu_phoneme_one_D_vector = cmu_phoneme_vector.flatten()  # 得到一维的音标向量
#         # print(cmu_phoneme_one_D_vector)
#         chinese_vector = chinese_model.wv.get_vector(word_split_list[1])
#         # print(chinese_vector.shape)
#         # 得到神经网络的输入
#         NN_input = np.concatenate((cmu_phoneme_one_D_vector, chinese_vector)).tolist()
#         json_dic['chinese'] = word_split_list[1]
#         json_dic['word'] = word_split_list[0]
#         json_dic['phonetic'] = ipa_phonemes
#         json_dic['input_vector'] = NN_input
#         with open('nn_input.json', 'a', encoding='utf-8') as fw:
#             json.dump(json_dic, fw, ensure_ascii=False)
#             fw.write('\n')


# --------------------得到两个文件 file1[汉语，音标]，file2[对应的拼写以字母隔开]---------------

cmu_phoneme_model = Word2Vec.load("phonetic_trained_vector.bin")  # 加载音标向量模型
chinese_model = Word2Vec.load("cet4_chinese_vector.model")  # 得到中文向量的模型，不需要
with open("cet4_chinese_EN_121.txt", 'r', encoding='utf-8') as file:  # 以UTF-8的方式打开
    for line in file:  # 一行一行的读，一行就是文件中的一行
        chinese_phonetic = ''
        letters = ''
        line = line.strip()  # 每一行都去头去尾换行符，或者空格
        word_split_list = line.split(' ')  # 以空格分开是英文，中文
        cmu_phonemes, ipa_phonemes = get_phonetic_components(word_split_list[0])  # 得到英标了
        chinese_phonetic += word_split_list[1]+' '
        # 先处理中文和英标
        for phoneme in ipa_phonemes:
            chinese_phonetic += phoneme+' '
        with open("cet4_chinese_phonetic.txt", 'a', encoding='utf-8') as fw:  # 以UTF-8的方式打开
            fw.write(str(chinese_phonetic))
            fw.write('\n')
        # 处理英文
        for letter in word_split_list[0]:
            letters += letter+' '
        with open("cet4_letter.txt", 'a', encoding='utf-8') as fw:  # 以UTF-8的方式打开
            fw.write(str(letters))
            fw.write('\n')
