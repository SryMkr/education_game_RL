"""
1: 文件中的cet4.txt中包含中文和英文

"""

import jieba
import re
from gensim.models import Word2Vec

# -----------------------------以下代码是为了将原文本的数据分割变为词组---------------
# 使用jieba对内容进行切割
# def extract_chinese_text(file_path):
#     chinese_text = []
#     with open(file_path, 'r', encoding='utf-8') as file:  # 以UTF-8的方式打开
#         for line in file:  # 一行一行的读，一行就是文件中的一行
#             line = line.strip()  # 每一行都去头去尾换行符，或者空格
#             if line:
#                 segments = jieba.lcut(line)  # 直接切割
#                 # 没啥还是切割
#                 chinese_segments = [seg for seg in segments if seg.isalpha()]
#                 chinese_text.extend(chinese_segments)
#     return chinese_text
#
#
# # 保存切割的内容
# def save_text_to_file(text, output_file):
#     with open(output_file, 'w', encoding='utf-8') as file:
#         for word in text:
#             file.write(word + '\n')  # 保存文件
#
#
# input_file_path = 'cet4.txt'  # 输入文件
# output_file_path = 'cet4_chinese_EN.txt'  # 输出文件
#
# chinese_text = extract_chinese_text(input_file_path)
# save_text_to_file(chinese_text, output_file_path)

# -----------------------------以下代码是为了去除文件中的非中文字，得到中文---------------
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
# cn_reg = '^[\u4e00-\u9fa5]+$'  # 去除英文
# for line in lines:
#     line_list = line.split('\n')[0].split(' ')
#     line_list_new = []
#     for word in line_list:
#         if re.search(cn_reg, word):
#             line_list_new.append(word)
#     output_file.write(' '.join(line_list_new) + '\n')
#     count += 1
#     if count % 10000 == 0:
#         print('目前已分词%d条数据' % count)
# print('分词程序执行结束！')
# print('主程序执行开始...')
#
# -----------------------------以下代码是为了去除文件中的中文字---------------
# input_file_name = 'cet4_chinese_EN.txt'
# output_file_name = 'cet4_english.txt'
# input_file = open(input_file_name, 'r', encoding='utf-8')
# output_file = open(output_file_name, 'w', encoding='utf-8')
# print('开始读入数据文件...')
# lines = input_file.readlines()
# print('读入数据文件结束！')
# print('分词程序执行开始...')
# count = 1
# cn_reg = '[^\u4e00-\u9fa5]+$'  # 去除中文
# for line in lines:
#     line_list = line.split('\n')[0].split(' ')
#     line_list_new = []
#     for word in line_list:
#         if re.search(cn_reg, word):
#             line_list_new.append(word)
#     output_file.write(' '.join(line_list_new) + '\n')
#     count += 1
#     if count % 10000 == 0:
#         print('目前已分词%d条数据' % count)
# print('分词程序执行结束！')

# -----------------------------以下代码是为了去除文件中空行---------------
# def extract_english_text(file_path):
#     english_text = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             line = line.strip()
#             if line:
#                 # Modify the line as needed
#                 line = line.replace('old', 'new')
#                 segments = jieba.lcut(line)
#                 english_text.extend(segments)
#
#     return english_text
#
#
# def save_text_to_file(text, output_file):
#     with open(output_file, 'w', encoding='utf-8') as file:
#         file.write('\n'.join(text))
#
#
# file_path = 'cet4_chinese.txt'
# output_file = 'output_cet4_chinese.txt'
#
# english_text = extract_english_text(file_path)
# save_text_to_file(english_text, output_file)
#
# -----------------------------以下代码是为了去除英文文件中空行---------------
# def extract_english_text(file_path):
#     english_text = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             line = line.strip()
#             if line:
#                 # Modify the line as needed
#                 line = line.replace('old', 'new')
#                 segments = jieba.lcut(line)
#                 english_text.extend(segments)
#
#     return english_text
#
#
# def save_text_to_file(text, output_file):
#     with open(output_file, 'w') as file:
#         file.write('\n'.join(text))


# file_path = 'cet4_english.txt'
# output_file = 'output_cet4_english.txt'
#
# english_text = extract_english_text(file_path)
# save_text_to_file(english_text, output_file)

# -----------------------------以下代码是为了去除英文文件中空行---------------
# import multiprocessing
# from gensim.models.word2vec import LineSentence
#
# print('主程序开始执行...')
# input_file_name = 'output_cet4_chinese.txt'
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



