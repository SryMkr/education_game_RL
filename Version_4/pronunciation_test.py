# 需要听到语音提示才开始录音
# 导入所需要的包
import pyaudio
import threading
import wave
from Common_Functions import *  # 导入函数
from YouDao_API import connect  # 发送打分请求
import os


# 语音学习的功能模块
class PronunciationTest(object):
    # 将赋值写初始化前面，子类会直接使用它的值
    words, phonetic, chinese_character = read_xls_shuffle('Word_Pool/Current_Deck.xls')

    # 主要是定义一些global variable 都可以使用或者修改
    def __init__(self, work_maker):
        self.word_maker = work_maker  # 得到主游戏
        self.surface_width, self.surface_height = work_maker.window.get_size()  # 得到主游戏的屏幕的长和宽
        self.pronunciation_surface = pygame.Surface((self.surface_width, self.surface_height))  # 创建一个和主屏幕大小一样的Surface
        self.coordinate_y_incremental = 60  # Y方向上的增量
        self.coordinate_x_incremental = 300  # X方向上的增量
        self.word_rect_dic = {}  # 定义一个单词和rect的字典

    # 写字功能
    def draw_Menu_Text(self, font, text, size, x, y, COLOR=(0, 0, 0)):
        font = pygame.font.Font(font, size)  # 文本的字体及大小
        text_surface = font.render(text, True, COLOR)  # 默认文本为黑色
        text_rect = text_surface.get_rect()  # 相当于给当前的surface 框起来 这样比较容易获得和使用参数
        text_rect.center = (x, y)  # 文本的中心点的坐标
        self.pronunciation_surface.blit(text_surface, text_rect)  # 居中对齐
        return text_rect

    # 发音学习页面
    def display_menu(self):
        self.pronunciation_surface.fill((200, 200, 200))  # 必须每一轮循环都更新画板
        # 画一个单词目录
        self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '单词目录', 50, self.surface_width / 2, 50)
        # 循环所有的单词，包括画，点击事件
        row_y = 0  # 标记第几行
        for word_index in range(len(self.words)):
            if word_index <= 9:
                self.word_rect_dic[word_index] = self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF",
                                                                     self.words[word_index], 50,
                                                                     self.surface_width / 6,
                                                                     self.surface_height / 8 + row_y *
                                                                     self.coordinate_y_incremental)
                # 判断鼠标在不在单词上
                if self.word_rect_dic[word_index].collidepoint(self.word_maker.mouse_current_x,
                                                               self.word_maker.mouse_current_y):
                    self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", self.words[word_index], 50,
                                        self.surface_width / 6, self.surface_height / 8 + row_y *
                                        self.coordinate_y_incremental, (255, 255, 255))

            elif 9 < word_index <= 19:
                self.word_rect_dic[word_index] = self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF",
                                                                     self.words[word_index], 50,
                                                                     self.surface_width / 6 +
                                                                     self.coordinate_x_incremental,
                                                                     self.surface_height / 8 + row_y *
                                                                     self.coordinate_y_incremental)
                # 判断鼠标在不在单词上
                if self.word_rect_dic[word_index].collidepoint(self.word_maker.mouse_current_x,
                                                               self.word_maker.mouse_current_y):
                    self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", self.words[word_index], 50,
                                        self.surface_width / 6 + self.coordinate_x_incremental,
                                        self.surface_height / 8 + row_y *
                                        self.coordinate_y_incremental, (255, 255, 255))

            else:
                self.word_rect_dic[word_index] = self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF",
                                                                     self.words[word_index], 50,
                                                                     self.surface_width / 6 +
                                                                     self.coordinate_x_incremental * 2,
                                                                     self.surface_height / 8 + row_y *
                                                                     self.coordinate_y_incremental)
                # 判断鼠标在不在单词上
                if self.word_rect_dic[word_index].collidepoint(self.word_maker.mouse_current_x,
                                                               self.word_maker.mouse_current_y):
                    self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", self.words[word_index], 50,
                                        self.surface_width / 6 + self.coordinate_x_incremental * 2,
                                        self.surface_height / 8 + row_y *
                                        self.coordinate_y_incremental, (255, 255, 255))

            row_y += 1  # 没写一行则加一
            if row_y == 10 or row_y == 20:
                row_y = 0  # 每十个单词为一组，重新归0

            # 如果鼠标点击了这个单词，则进入独立单词的学习
            if self.word_rect_dic[word_index].collidepoint(self.word_maker.mouse_click_x,
                                                           self.word_maker.mouse_click_y) and \
                    self.word_maker.pronunciation_menu_chance:
                self.word_maker.pronunciation_current_word_index = word_index  # 记住当前是哪个单词
                self.word_maker.current_menu = self.word_maker.pronunciation_individual_menu  # 显示独立单词学习的菜单

        # 返回主菜单的按钮
        return_main_menu = self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '返回菜单', 50,
                                               self.surface_width / 2, self.surface_height - 50)  # 创建一个返回的按钮功能，放到左下角的位置
        if return_main_menu.collidepoint(self.word_maker.mouse_current_x, self.word_maker.mouse_current_y):
            self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '返回菜单', 50,
                                self.surface_width / 2, self.surface_height - 50,
                                (255, 255, 255))  # 创建一个返回的按钮功能，放到左下角的位置
        # 如果选中了返回按钮，返回到主菜单
        if return_main_menu.collidepoint(self.word_maker.mouse_click_x, self.word_maker.mouse_click_y) and \
                self.word_maker.pronunciation_menu_chance:
            self.word_maker.current_menu = self.word_maker.main_menu  # 返回主菜单
        self.word_maker.window.blit(self.pronunciation_surface, (0, 0))  # 将发音学习画到主游戏上


# 独立学习单词的菜单
class IndividualWord(PronunciationTest):
    def __init__(self, work_maker):
        PronunciationTest.__init__(self, work_maker)
        self.player_pronunciation = AudioModel(False)  # 实例化录音功能
        self.show_results = False  # 目前不展示发音结果
        self.word_properties = ['音素', '得分', '对错']
        self.player_audio_path = 'Player_Pronunciation'
        self.rerecord = False  # 重新录音展示

    # 学习单独的一个单词
    def display_menu(self):
        self.pronunciation_surface.fill((200, 200, 200))  # 必须每一轮循环都更新画板
        # 将当前的单词，音标，汉字画到面板上
        self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF",
                            self.words[self.word_maker.pronunciation_current_word_index],
                            80, self.surface_width / 8 + 100, self.surface_height / 8 - 20)
        self.draw_Menu_Text("Game_Fonts/phonetic.ttf", self.phonetic[self.word_maker.pronunciation_current_word_index],
                            60, self.surface_width / 8 + 100, self.surface_height / 8 + 70)
        self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF",
                            self.chinese_character[self.word_maker.pronunciation_current_word_index],
                            80, self.surface_width / 8 + 100, self.surface_height / 8 + 160)

        # 展示图片
        self.words_pictures = pygame.image.load(
            "Word_Pictures/" + self.words[self.word_maker.pronunciation_current_word_index] + '.png')
        self.words_pictures = pygame.transform.scale(self.words_pictures, (300, 300))
        self.pronunciation_surface.blit(self.words_pictures, (80, self.surface_height / 2 - 50))

        # 官方发音
        official_pronunciation_rect = self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '英式发音',
                                                          50, self.surface_width / 2 + 200,
                                                          self.surface_height / 8 - 20)
        if official_pronunciation_rect.collidepoint(self.word_maker.mouse_current_x, self.word_maker.mouse_current_y):
            self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '英式发音', 50, self.surface_width / 2 + 200,
                                self.surface_height / 8 - 20, (255, 255, 255))  # 创建一个返回的按钮功能，放到左下角的位置
        # 如果点击了官方发音按钮，则发音
        if official_pronunciation_rect.collidepoint(self.word_maker.mouse_click_x, self.word_maker.mouse_click_y) and \
                self.word_maker.pronunciation_menu_chance:
            game_Sound('game_sound/mouse_click_1.mp3')
            game_Sound('UK_Pronunciation/' + self.words[self.word_maker.pronunciation_current_word_index] + '.mp3')

        # 开始录音
        begin_pronunciation_rect = self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '开始录音',
                                                       50, self.surface_width / 2 + 100,
                                                       self.surface_height / 8 + 80)
        if begin_pronunciation_rect.collidepoint(self.word_maker.mouse_current_x,
                                                 self.word_maker.mouse_current_y):
            self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '开始录音', 50, self.surface_width / 2 + 100,
                                self.surface_height / 8 + 80, (255, 255, 255))
        #  如果点击了开始录音按钮，则开始录音
        if begin_pronunciation_rect.collidepoint(self.word_maker.mouse_click_x, self.word_maker.mouse_click_y) and \
                self.word_maker.pronunciation_menu_chance:
            game_Sound('game_sound/start.wav')
            self.player_pronunciation.is_recording = True  # 则开始录音
            # 保存当前的发音
            self.player_pronunciation.record_and_save(self.words[self.word_maker.pronunciation_current_word_index])

        # 结束录音
        end_pronunciation_rect = self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '结束录音',
                                                     50, self.surface_width / 2 + 360,
                                                     self.surface_height / 8 + 80)
        if end_pronunciation_rect.collidepoint(self.word_maker.mouse_current_x,
                                               self.word_maker.mouse_current_y):
            self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '结束录音', 50, self.surface_width / 2 + 360,
                                self.surface_height / 8 + 80, (255, 255, 255))
        #  如果点击结束录音按钮，则结束录音
        if end_pronunciation_rect.collidepoint(self.word_maker.mouse_click_x, self.word_maker.mouse_click_y) and \
                self.word_maker.pronunciation_menu_chance:
            game_Sound('game_sound/end.wav')
            self.player_pronunciation.is_recording = False  # 则结束录音

        # 录音结果
        result_pronunciation_rect = self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '录音结果',
                                                        50, self.surface_width / 2 + 200,
                                                        self.surface_height / 8 + 180)
        if result_pronunciation_rect.collidepoint(self.word_maker.mouse_current_x,
                                                  self.word_maker.mouse_current_y):
            self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '录音结果', 50, self.surface_width / 2 + 200,
                                self.surface_height / 8 + 180, (255, 255, 255))
        # 如果选中录音结果
        if result_pronunciation_rect.collidepoint(self.word_maker.mouse_click_x, self.word_maker.mouse_click_y) and \
                self.word_maker.pronunciation_menu_chance:
            game_Sound('game_sound/mouse_click_1.mp3')
            # 在这需要判断音频的长短来决定是否发送请求
            if os.path.exists(os.path.join(self.player_audio_path,
                                           (self.words[self.word_maker.pronunciation_current_word_index] + '.wav'))):
                game_Sound(
                    'Player_Pronunciation/' + self.words[self.word_maker.pronunciation_current_word_index] + '.wav')
                self.player_score = self.player_pronunciation.get_score(
                    self.words[self.word_maker.pronunciation_current_word_index])
                # 切换了单词不再展示结果
                self.show_results = True
                self.rerecord = False
            else:
                self.rerecord = True
        # 展示结果 和 结果不为空
        if self.show_results:
            if self.player_score:
                self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", str(self.player_score[0]), 80,
                                    self.surface_width / 2 + 400, self.surface_height / 8 + 180)  # 单词得分
                for property_index in range(len(self.word_properties)):
                    # 先显示三列的指标命名
                    self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", self.word_properties[property_index], 40,
                                        self.surface_width / 2 + property_index * 200, self.surface_height / 2 - 30)
                    #  顺序展示音素的结果
                    for phoneme_index in range(1, len(self.player_score)):
                        for index in range(len(self.player_score[phoneme_index])):
                            if index == 0:  # 如果是音素用一个字体
                                self.draw_Menu_Text("Game_Fonts/phonetic.ttf", str(self.player_score[phoneme_index][index]),
                                                    30,
                                                    self.surface_width / 2 + index * 200,
                                                    (self.surface_height / 2 - 30) + phoneme_index * 35)
                            else:
                                self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF",
                                                    str(self.player_score[phoneme_index][index]), 30,
                                                    self.surface_width / 2 + index * 200,
                                                    (self.surface_height / 2 - 30) + phoneme_index * 35)
            else:
                self.rerecord = True
        # 让玩家重新录音
        if self.rerecord:
            self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '录音太短,请重新录音', 40,
                                self.surface_width / 2 + 200, self.surface_height / 2 - 30)
        # 返回目录的按钮
        text_rect_return = self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '返回目录', 50, self.surface_width / 2,
                                               self.surface_height - 50)  # 创建一个返回的按钮功能，放到左下角的位置
        if text_rect_return.collidepoint(self.word_maker.mouse_current_x, self.word_maker.mouse_current_y):
            self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '返回目录', 50, self.surface_width / 2,
                                self.surface_height - 50, (255, 255, 255))  # 创建一个返回的按钮功能，放到左下角的位置

        # 如果选中返回到’单词目录‘
        if text_rect_return.collidepoint(self.word_maker.mouse_click_x, self.word_maker.mouse_click_y) and \
                self.word_maker.pronunciation_menu_chance:
            self.word_maker.current_menu = self.word_maker.pronunciation_menu  # 显示单词目录
            self.show_results = False  # 不再展示结果
            self.rerecord = False
        # 创建一个继续的按钮功能，放到右下角的位置
        text_rect_continue = self.draw_Menu_Text("Game_Fonts/symbol_signs.otf", 'R', 80, self.surface_width - 80,
                                                 self.surface_height - 50)
        if text_rect_continue.collidepoint(self.word_maker.mouse_current_x, self.word_maker.mouse_current_y):
            self.draw_Menu_Text("Game_Fonts/symbol_signs.otf", 'R', 80, self.surface_width - 80,
                                self.surface_height - 50, (255, 255, 255))  # 创建一个返回的按钮功能，放到左下角的位置

        # 如果选中了继续按钮，显示下一个单词
        if text_rect_continue.collidepoint(self.word_maker.mouse_click_x, self.word_maker.mouse_click_y) and \
                self.word_maker.pronunciation_menu_chance:
            self.word_maker.pronunciation_current_word_index += 1  # 显示下一个单词
            # 如果下一个单词超过了已有的单词数，则到第一个
            if self.word_maker.pronunciation_current_word_index == len(self.words):
                self.word_maker.pronunciation_current_word_index = 0  # 则回到第一个单词
            self.show_results = False  # 不再展示结果
            self.rerecord = False
        # 创建一个上一个的按钮功能，放到左下角下角的位置
        text_rect_previous = self.draw_Menu_Text("Game_Fonts/symbol_signs.otf", 'L', 80,
                                                 80, self.surface_height - 50)
        if text_rect_previous.collidepoint(self.word_maker.mouse_current_x, self.word_maker.mouse_current_y):
            self.draw_Menu_Text("Game_Fonts/symbol_signs.otf", 'L', 80, 80, self.surface_height - 50,
                                (255, 255, 255))  # 创建一个返回的按钮功能，放到左下角的位置

        # 如果选中了继续按钮，显示上一个单词
        if text_rect_previous.collidepoint(self.word_maker.mouse_click_x, self.word_maker.mouse_click_y) and \
                self.word_maker.pronunciation_menu_chance:
            self.word_maker.pronunciation_current_word_index -= 1  # 显示下一个单词
            # 如果到了第一个单词
            if self.word_maker.pronunciation_current_word_index == -1:
                self.word_maker.pronunciation_current_word_index = (len(self.words) - 1)  # 则到最后一个单词
            self.show_results = False  # 不再展示结果
            self.rerecord = False
        self.word_maker.window.blit(self.pronunciation_surface, (0, 0))  # 将这个屏幕画到主游戏上


# 定义一个声音得model类别
class AudioModel(object):
    def __init__(self, is_recording):
        self.is_recording = is_recording  # 是不是正在录音
        self.audio_chunk_size = 1600  # 缓冲块的大小，多久输出一次
        self.audio_channels = 1  # 单声道
        self.audio_format = pyaudio.paInt16  # 采样点的宽度为16位
        self.audio_rate = 16000  # 采样率

    # 这个是记录玩家的发音，挺有用的，可以让玩家听到自己的发音
    def record(self, audio_name):
        p = pyaudio.PyAudio()  # 实例化一个audio对象
        # 打开一个信号流
        stream = p.open(
            format=self.audio_format,
            channels=self.audio_channels,
            rate=self.audio_rate,
            input=True,
            frames_per_buffer=self.audio_chunk_size
        )
        wf = wave.open(audio_name, 'wb')  # 打开文件
        wf.setnchannels(self.audio_channels)  # 设置声道
        wf.setsampwidth(p.get_sample_size(self.audio_format))  # 设置采样点的位数
        wf.setframerate(self.audio_rate)  # 设置采样率
        # 读取数据写入文件
        while self.is_recording:
            data = stream.read(self.audio_chunk_size)  # 开始读取数据
            wf.writeframes(data)  # 写入数据

        wf.close()  # 关闭文件
        stream.stop_stream()  # 关闭信号
        stream.close()  # 流关闭
        p.terminate()  # 线程结束

    # 这个是记录并且保存结果
    def record_and_save(self, audio_name):
        self.is_recording = True
        self.audio_file_name = 'Player_Pronunciation/' + audio_name + '.wav'  # 保存玩家的发音路径
        # 启动一个线程，target是调用一个函数，args是固定的参数，就是我的文件名字，start是启动线程
        threading.Thread(target=self.record, args=(self.audio_file_name,)).start()

    # 获得玩家的分数,输入单词
    def get_score(self, word):
        audio_path = 'Player_Pronunciation/' + word + '.wav'  # 玩家发音文件要存的地方
        # connect相当于打包发给语音库给打分
        score_result = connect(audio_path, word)  # 第一个参数为玩家的发音路径，第二个参数为文件中的内容
        return score_result
