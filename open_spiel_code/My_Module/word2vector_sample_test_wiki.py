from gensim.models import Word2Vec
# from nltk.corpus import cmudict
import numpy as np
from gensim.corpora import WikiCorpus

"""
# -----------------------------以下代码是为了捕捉维基百科的中文语料库，好像没执行完---------------
# 缺点就是即使我执行结束了了，我的汉语还有没有出现的情况，所以之后直接把训练的中文集变成了单词词库
print('主程序开始...')
input_file_name = 'zhwiki-latest-pages-articles.xml.bz2' # 这个文件需要手动下载
output_file_name = 'wiki.cn.txt'
print('开始读入wiki数据...')
input_file = WikiCorpus(input_file_name, dictionary={})
print('wiki数据读入完成！')
output_file = open(output_file_name, 'w', encoding="utf-8")
print('处理程序开始...')
count = 0
for text in input_file.get_texts():
    output_file.write(' '.join(text) + '\n')
    count = count + 1
    if count % 10000 == 0:
        print('目前已处理%d条数据' % count)
print('处理程序结束！')
output_file.close()
print('主程序结束！')

# -----------------------------以下代码是为了将繁体字改为简体字---------------
    # import zhconv  # 调用这个包将繁体字改为简体字
    # print('主程序执行开始...')
    #
    # input_file_name = 'wiki.cn.txt'
    # output_file_name = 'wiki.cn.simple.txt'
    # input_file = open(input_file_name, 'r', encoding='utf-8')
    # output_file = open(output_file_name, 'w', encoding='utf-8')
    #
    # print('开始读入繁体文件...')
    # lines = input_file.readlines()
    # print('读入繁体文件结束！')
    #
    # print('转换程序执行开始...')
    # count = 1
    # for line in lines:
    #     output_file.write(zhconv.convert(line, 'zh-hans'))
    #     count += 1
    #     if count % 10000 == 0:
    #         print('目前已转换%d条数据' % count)
    # print('转换程序执行结束！')
    #
    # print('主程序执行结束！')
    

    # -----------------------------以下代码是为了去除文件中的非中文字---------------
    # coding:utf-8
    # import re
    #
    # print('主程序执行开始...')
    #
    # input_file_name = 'wiki.cn.simple.txt'
    # output_file_name = 'wiki.txt'
    # input_file = open(input_file_name, 'r', encoding='utf-8')
    # output_file = open(output_file_name, 'w', encoding='utf-8')
    #
    # print('开始读入数据文件...')
    # lines = input_file.readlines()
    # print('读入数据文件结束！')
    #
    # print('分词程序执行开始...')
    # count = 1
    # cn_reg = '^[\u4e00-\u9fa5]+$'
    #
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

    # print('主程序执行结束！')
    

    # # -----------------------------以下代码是为了将中文转为向量---------------
    # import multiprocessing
    # from gensim.models import Word2Vec
    # from gensim.models.word2vec import LineSentence
    #
    # print('主程序开始执行...')
    #
    # input_file_name = 'wiki.txt'
    # model_file_name = 'wiki.model'
    #
    # print('转换过程开始...')
    # model = Word2Vec(LineSentence(input_file_name),
    #                  vector_size=400,  # 词向量长度为400
    #                  window=5,
    #                  min_count=5,
    #                  workers=multiprocessing.cpu_count())
    # print('转换过程结束！')
    #
    # print('开始保存模型...')
    # model.save(model_file_name)
    # print('模型保存结束！')
    # print('主程序执行结束！')
# -----------------------------以下代码是为了将英文翻译成中文---------------
from translate import Translator
def translate_english_to_chinese(word):
    translator = Translator(to_lang='zh')
    translation = translator.translate(word)
    return translation
# Example usage
english_words = ['hello', 'world', 'good', 'morning']
chinese_translations = []

for word in english_words:
    translation = translate_english_to_chinese(word)
    chinese_translations.append(translation)

# Print the English words and their Chinese translations
for i in range(len(english_words)):
    print(f"{english_words[i]}: {chinese_translations[i]}")
"""

