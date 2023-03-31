"""
这个页面有两个函数功能，第一个逐个展示单词，第二个展示所有单词
两种发音以及检查的模式，一种键盘一种鼠标点击 看看哪个比较多
"""

import pygame.draw
from game_level_function import *
import time


# 首先在游戏之前逐个展示单词，展示结束以后跳到展示所有单词的页面
class PresentWords(object):
    # 主要是定义一些global variable 都可以使用或者修改
    def __init__(self, work_maker):
        # 读取文件中的单词，音标，翻译
        self.words, self.phonetic, self.chinese_character = read_xls_shuffle('Word_Pool/Current_Deck.xls')
        self.word_maker = work_maker  # 得到主游戏
        self.surface_width, self.surface_height = work_maker.window.get_size()  # 得到主游戏的屏幕的长和宽
        self.present_words_surface = pygame.Surface((self.surface_width, self.surface_height))  # 创建一个和主屏幕大小一样的Surface
        self.current_task_index = 0  # 展示单词阶段任务索引
        self.countdown = 40  # 每个单词给40秒的时间，这个是静态的
        self.count_seconds = 40  # 从40秒开始倒数，这个是动态的
        self.base_time = 40  # 记忆的基础时间是40秒
        self.start_time = datetime.now()  # 获取但是展示开始的时间
        self.BACKGROUND_COLOR = (200, 200, 200)  # 游戏界面的背景颜色
        self.word_maker.start_play_time = time.perf_counter()  # 开始展示单词时计时为 0，为了计算玩家玩了多长时间，目前只统计一轮的时间

    # 写字功能
    def draw_Menu_Text(self, font, text, size, x, y, COLOR=(0, 0, 0)):
        font = pygame.font.Font(font, size)  # 文本的字体及大小
        text_surface = font.render(text, True, COLOR)  # 默认文本为黑色
        text_rect = text_surface.get_rect()  # 相当于给当前的surface 框起来 这样比较容易获得和使用参数
        text_rect.center = (x, y)  # 文本的中心点的坐标
        self.present_words_surface.blit(text_surface, text_rect)  # 居中对齐
        return text_rect

    # 展示单词页面
    def display_menu(self):
        self.present_words_surface.fill(self.BACKGROUND_COLOR)  # 必须每一轮循环都更新画板
        # 画一个单词目录
        self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '单词展示' + str(self.countdown)+'秒', 50, self.surface_width / 2, 30,
                            (150, 150, 150))
        # 将当前的单词，音标，汉字画到面板上
        self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", self.words[self.current_task_index],
                            80, self.surface_width / 2, self.surface_height / 8 + 10)
        self.draw_Menu_Text("Game_Fonts/phonetic.ttf", self.phonetic[self.current_task_index],
                            60, self.surface_width / 2, self.surface_height / 8 + 80)
        self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", self.chinese_character[self.current_task_index],
                            80, self.surface_width / 2, self.surface_height / 8 + 170)
        # 展示图片，反正也没图片就不展示了
        # self.words_pictures = pygame.image.load("Word_Pictures/" + self.words[self.current_task_index] + '.png')
        # self.words_pictures = pygame.transform.scale(self.words_pictures, (300, 300))
        # self.present_words_surface.blit(self.words_pictures,
        #                                 (self.surface_width / 2 - 150, self.surface_height / 2 - 65))

        # 按Q可以发音
        if self.word_maker.pronunciation:
            game_Sound('UK_Pronunciation/' + self.words[self.current_task_index] + '.mp3')

        # 画一个progress bar，控制学习时间
        pygame.draw.rect(self.present_words_surface, (0, 0, 0),
                         ((0, self.surface_height - 60), (self.surface_width, 50)),
                         width=4)  # 首先画一个边框
        seconds = self.countdown - self.count_seconds  # 已经过了多少秒
        self.decrease_width = self.surface_width / self.countdown  # 1秒减少多少宽度
        if 0 <= seconds <= 20:
            pygame.draw.rect(self.present_words_surface, (190, 190, 190),
                             ((seconds * self.decrease_width, self.surface_height - 56),
                              (self.surface_width - seconds * self.decrease_width, 42)))  # 画进度条
        elif 20 < seconds <= 30:
            pygame.draw.rect(self.present_words_surface, (180, 180, 180),
                             ((seconds * self.decrease_width, self.surface_height - 56),
                              (self.surface_width - seconds * self.decrease_width, 42)))  # 画进度条
        else:
            pygame.draw.rect(self.present_words_surface, (170, 170, 170),
                             ((seconds * self.decrease_width, self.surface_height - 56),
                              (self.surface_width - seconds * self.decrease_width, 42)))  # 画进度条
        # 提示按钮，按Q可以发音
        self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '按Q发音', 50, self.surface_width - 100,
                            self.surface_height/4 - 20, (150, 150, 150))

        # 点击‘点击发音’也可以发音
        # 创建一个返回的按钮功能，放到左下角的位置
        click_pronunciation_button = self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '点此发音', 50,
                                                         self.surface_width - 100,
                                                         self.surface_height/4 - 80, (0, 150, 0))

        if click_pronunciation_button.collidepoint(self.word_maker.mouse_current_x, self.word_maker.mouse_current_y):
            self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '点此发音', 50, self.surface_width - 100,
                                self.surface_height/4 - 80, (0, 255, 0))  # 创建一个返回的按钮功能，放到左下角的位置
        if click_pronunciation_button.collidepoint(self.word_maker.mouse_click_x, self.word_maker.mouse_click_y) and \
                self.word_maker.present_word_menu_chance:
            game_Sound('UK_Pronunciation/' + self.words[self.current_task_index] + '.mp3')

        # 跳过单词单词按钮，学生觉得自己这个单词已经学习的很好，可以跳过当前单词
        skip_task_button = self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '跳过单词', 50,
                                               self.surface_width - 100,
                                               self.surface_height - 130, (150, 150, 150))  # 创建一个返回的按钮功能，放到左下角的位置
        if skip_task_button.collidepoint(self.word_maker.mouse_current_x, self.word_maker.mouse_current_y):
            self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", '跳过单词', 50, self.surface_width - 100,
                                self.surface_height - 130, (255, 255, 255))  # 创建一个返回的按钮功能，放到左下角的位置
        # 如果选中了跳过单词按钮，则跳过当前单词
        if skip_task_button.collidepoint(self.word_maker.mouse_click_x, self.word_maker.mouse_click_y) and \
                self.word_maker.present_word_menu_chance:
            self.current_task_index += 1  # 展示单词阶段任务索引
            game_Sound('game_sound/mouse_click_2.mp3', 0.3)  # 展示单词阶段，切换单词的声音
            self.count_seconds = self.base_time + self.count_seconds  # 基础时间+节省的时间
            self.countdown = self.count_seconds  # 总时间改变
            if self.current_task_index == len(self.words):  # 如果已经是最后一个单词
                self.word_maker.saved_time = self.count_seconds-self.base_time  # 记录玩家这一轮节省了多少时间
                self.word_maker.present_all_word_menu = PresentAllTasks(self.word_maker)  # 实例化展示单词菜单
                self.word_maker.current_menu = self.word_maker.present_all_word_menu
        self.word_maker.window.blit(self.present_words_surface, (0, 0))  # 将发音学习画到主游戏上

        # 需要用时间控制游戏时间
        if datetime.now() > self.start_time + timedelta(seconds=1):  # 每过一秒
            self.start_time = datetime.now()  # 将现在的时间给过去的时间
            # 控制软件在什么时候发音
            self.count_seconds -= 1  # 倒计时减 1
            if self.count_seconds in np.arange(5, self.countdown, 10).tolist():  # 每隔10秒，强制发音
                game_Sound('UK_Pronunciation/' + self.words[self.current_task_index] + '.mp3')
            if self.count_seconds == -1:  # 如果倒计时结束
                game_Sound('game_sound/mouse_click_1.mp3', 0.3)  # # 展示单词阶段，切换单词的声音
                self.current_task_index += 1  # 换到下一个单词
                self.count_seconds = self.base_time  # 从40秒重新倒数
                self.countdown = self.base_time
                if self.current_task_index > len(self.words) - 1:  # 如果展示结束，展示所有单词
                    self.word_maker.present_all_word_menu = PresentAllTasks(self.word_maker)  # 实例化展示单词菜单
                    self.word_maker.current_menu = self.word_maker.present_all_word_menu  # 跳到展示所有单词的界面


# 在每个单词展示结束以后，再次展示所有的单词120秒，展示结束以后跳到游戏界面
class PresentAllTasks(object):
    def __init__(self, work_maker):  # 参数一定是实例化的游戏
        # 读取文件中的单词，音标，翻译
        self.words, self.phonetic, self.chinese_character = read_xls_shuffle('Word_Pool/Current_Deck.xls')
        self.word_maker = work_maker  # 得到主游戏
        self.surface_width, self.surface_height = self.word_maker.window.get_size()  # 得到主游戏的屏幕的长和宽
        self.present_words_surface = pygame.Surface((self.surface_width, self.surface_height))  # 创建一个和主屏幕大小一样的Surface
        self.countdown_all_tasks = 60 + int(self.word_maker.saved_time/2)  # 展示所有单词30秒，是个变化值
        self.countdown_time = 60 + int(self.word_maker.saved_time/2)  # 展示的总时间，是个固定值
        # 将节省的时间减少一半用于后续的测试，另外一半用于总体记忆
        self.word_maker.saved_time = int(self.word_maker.saved_time/2)
        self.BACKGROUND_COLOR = (200, 200, 200)  # 该界面的背景颜色
        self.start_time = datetime.now()  # 获取但是展示开始的时间
        self.decrease_width = self.surface_width / self.countdown_all_tasks  # 1秒减少多少宽度

    # 写字功能
    def draw_Menu_Text(self, font, text, size, x, y, COLOR=(0, 0, 0)):
        font = pygame.font.Font(font, size)  # 文本的字体及大小
        text_surface = font.render(text, True, COLOR)  # 默认文本为黑色
        text_rect = text_surface.get_rect()  # 相当于给当前的surface 框起来 这样比较容易获得和使用参数
        text_rect.center = (x, y)  # 文本的中心点的坐标
        self.present_words_surface.blit(text_surface, text_rect)  # 居中对齐
        return text_rect

    # 该函数的功能是在每个单词都展示完以后在展示一遍所有的10个任务
    def display_menu(self):
        self.present_words_surface.fill(self.BACKGROUND_COLOR)  # 每次都要刷新屏幕
        # 画一个progress bar，控制学习时间，实现2分钟倒计时按钮
        pygame.draw.rect(self.present_words_surface, (0, 0, 0),
                         ((0, self.surface_height - 60), (self.surface_width, 50)),
                         width=4)  # 首先画一个边框
        seconds = self.countdown_time - self.countdown_all_tasks  # 已经过了多少秒
        if 0 <= seconds <= 20:
            pygame.draw.rect(self.present_words_surface, (190, 190, 190),
                             ((seconds * self.decrease_width, self.surface_height - 56),
                              (self.surface_width - seconds * self.decrease_width, 42)))  # 画进度条
        elif 20 < seconds <= 40:
            pygame.draw.rect(self.present_words_surface, (180, 180, 180),
                             ((seconds * self.decrease_width, self.surface_height - 56),
                              (self.surface_width - seconds * self.decrease_width, 42)))  # 画进度条
        else:
            pygame.draw.rect(self.present_words_surface, (170, 170, 170),
                             ((seconds * self.decrease_width, self.surface_height - 56),
                              (self.surface_width - seconds * self.decrease_width, 42)))  # 画进度条

        for i in range(len(self.words)):
            # 展示英文
            self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", self.words[i], 50, 200, 70 * i + 30)
            # 展示音标
            self.draw_Menu_Text("Game_Fonts/phonetic.ttf", self.phonetic[i], 40, 500, 70 * i + 30)
            # 展示中文翻译
            self.draw_Menu_Text("Game_Fonts/chinese_pixel_font.TTF", self.chinese_character[i], 50, 800, 70 * i + 30)
        self.word_maker.window.blit(self.present_words_surface, (0, 0))  # 将展示所有单词画到主游戏上

        # 控制单词展示的时间
        if datetime.now() > self.start_time + timedelta(seconds=1):  # 如果时间间隔相差一秒
            self.start_time = datetime.now()  # 将现在的时间给过去的时间
            self.countdown_all_tasks -= 1
            if self.countdown_all_tasks == -1:  # 如果时间结束，则进入游戏界面
                self.word_maker.game_level_menu = GameLevel(self.word_maker)  # 重新实例化游戏界面
                self.word_maker.current_menu = self.word_maker.game_level_menu
