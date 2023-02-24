"""
provide score, the format is .wav
the enduring time should be long enough, or cause error 11010 (short audio)
"""


# import modules
import sys
import uuid
import requests
import wave
import base64
import hashlib
import json
from importlib import reload
import time

reload(sys)  # reload modules

YouDao_URL = "https://openapi.youdao.com/iseapi"  # YouDao_API URL
APP_KEY = '3d96861d86e5d1eb'  # my application ID
APP_SECRET = '0wq0M3dYpvMhlgpzgcTv6MyWLCn7ou0A'  # my application passwords


# q is the Base64 code of .WAV audio
def truncate(q):
    # if q is null
    if q is None:
        return None
    # get the length of q
    size = len(q)
    # if q less than 20, the input is 20, otherwise, q = [the head 10 bytes] +[the length of q]+ [the tail 10 bytes]
    return q if size <= 20 else q[0:10] + str(size) + q[size-10:size]


# hash
def encrypt(signStr):
    hash_algorithm = hashlib.sha256()  # create a has equation
    hash_algorithm.update(signStr.encode('utf-8'))  # encoding
    return hash_algorithm.hexdigest()  # return a string


# the request
def do_request(data):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}  # the header
    # the first argument URL, the second argument data， the third argument header → request link
    return requests.post(YouDao_URL, data=data, headers=headers)


# first argument is the path of player pronunciation, the second argument is 'word'
def connect(audio_file_path, word_text):
    audio_file_path = audio_file_path  # the path of player pronunciation
    lang_type = 'en'  # language type
    # rindex() 返回子字符串 str 在字符串中最后出现的位置 其实就是获得文件的拓展名 可是不就是 .wav么
    extension = audio_file_path[audio_file_path.rindex('.')+1:]
    if extension != 'wav':
        print('不支持的音频类型')
        sys.exit(1)
    wav_info = wave.open(audio_file_path, 'rb')  # open wav audio
    sample_rate = wav_info.getframerate()  # get sample rate
    nchannels = wav_info.getnchannels()  # get the number of channels
    wav_info.close()  # close file
    with open(audio_file_path, 'rb') as file_wav:
        q = base64.b64encode(file_wav.read()).decode('utf-8')  # encoding wav based on Base64
    data = {}  # store necessary data
    data['text'] = word_text  # word spelling
    curtime = str(int(time.time()))  # the start time
    data['curtime'] = curtime  # # the start time
    salt = str(uuid.uuid1())  # 唯一通用识别码 就是全球唯一的一个标志而已
    signStr = APP_KEY + truncate(q) + salt + curtime + APP_SECRET  # 一个签名的原始组成
    sign = encrypt(signStr)   # 将这个签名使用hash算法编码
    data['appKey'] = APP_KEY  # appication ID
    data['q'] = q  # 就是q的值
    data['salt'] = salt  # 唯一通用识别码
    data['sign'] = sign  # hash转换后的值
    data['signType'] = "v2"  # 签名类型
    data['langType'] = lang_type  # 语言类型，只支持英文
    data['rate'] = sample_rate  # 采样率16000
    data['format'] = 'wav'  # 文件类型wav
    data['channel'] = nchannels  # 声道数， 仅支持单声道，请填写固定值1
    data['type'] = 1  # 上传类型， 仅支持base64上传，请填写固定值1

    response = do_request(data)  # 发送打分请求
    j = json.loads(str(response.content, encoding="utf-8"))  # 返回一个json文件，就是返回的结果
    if j['errorCode'] == '11010':  # 如果存在错误编码
        word_information_list = []
    # print(j) # 看得到的数据是什么
    else:
        word = j["words"]  # 获得有关单词的所有信息，返回的是个列表
        word = word[0]  # 0代表去除列表，我就只有一个单词
        word_information_list = []   # 创建一个列表顺序保存所有的数据
        pronunciation_score = int(word['pronunciation'])  # 单词的整体发音分数
        word_information_list.append(pronunciation_score)
        word_phonemes = word['phonemes']  # 获得单词的音素
        for phonemes_index in range(len(word_phonemes)):  # 顺序循环单词的所有音素
            phonemes_list = []
            current_phoneme = word_phonemes[phonemes_index]["phoneme"]  # 当前的音素
            phonemes_list.append(current_phoneme)
            phoneme_pronunciation = int(word_phonemes[phonemes_index]['pronunciation'])  # 获得这个音素的发音分数
            phonemes_list.append(phoneme_pronunciation)
            phoneme_judge = word_phonemes[phonemes_index]["judge"]  # 判断当前的音素发音是否正确
            phonemes_list.append(phoneme_judge)
            word_information_list.append(phonemes_list)
        # 【word,[w,12,True],[o,11,False],.......】
    return word_information_list
