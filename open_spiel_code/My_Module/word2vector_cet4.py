"""
对文件进行预处理 将txt文本分割为纯中文和纯英文，主要使用的是正则表达式
"""

import re
import numpy as np


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




