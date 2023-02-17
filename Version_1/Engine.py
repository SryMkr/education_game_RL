# import necessary packages
from pronunciation_test import PronunciationTest, IndividualWord  # import speaking recognition
from main_game_function import *
from feedback_training import *
from Common_Functions import read_excel_game_record, select_tasks


class MainGame(object):  # main game menu
    def __init__(self, window_width, window_height):  # input the width and height of display surface
        pygame.init()  # initialize six modules（including music and fonts）
        self.window_width = window_width  # the width of display surface
        self.window_height = window_height  # the height of display surface
        self.window = pygame.display.set_mode((self.window_width, self.window_height))  # initialize game display
        # background picture
        self.GAME_BACKGROUND_PICTURE = pygame.image.load("Game_Pictures/Main_Game_Background_1.png")
        self.gameplay_time = pygame.time.get_ticks()  # record running time (milliseconds)
        self.Main_Game_Running = True  # control the running of game
        self.Menu_Font = "Game_Fonts/chinese_pixel_font.TTF"  # Menu Font
        self.WHITE = (255, 255, 255)  # white
        self.BGC = (200, 200, 200)  # grey
        self.mouse_current_x = 0  # the current x_axis of mouse
        self.mouse_current_y = 0  # the current y_axis of mouse
        self.mouse_click_x = 0  # the click x_axis of mouse
        self.mouse_click_y = 0  # the click y_axis of mouse
        self.mouse_rel_x = 0  # the x coordinate relative distance x_axis of mouse
        self.mouse_rel_y = 0  # the y coordinate relative distance x_axis of mouse
        self.finished_tasks = {}  # record finished tasks in the session (will be moved)
        self.all_tasks = {}  # record all task (feedback)
        self.pronunciation_current_word_index = 0  # 记录发音记忆中，是第几个单词
        # reshape picture
        self.GAME_BACKGROUND_PICTURE = pygame.transform.scale(self.GAME_BACKGROUND_PICTURE,
                                                              (self.window_width, self.window_height))
        self.word_loop = {}  # 记录玩家某一个单词用了几轮
        self.remembered_words_number = 0  # 用来记录玩家已经记住了几个单词
        self.main_menu = MainMenu(self)  # 第一级菜单
        self.game_setting_menu = GameSetting(self)  # 实例化第二季菜单
        self.pronunciation_menu = PronunciationTest(self)  # 实例化发音菜单
        self.pronunciation_individual_menu = IndividualWord(self)  # 实例化独立发音菜单
        self.feedback_train_menu = GameFeedback(self)  # 实例化反馈菜单
        self.pronunciation_menu_chance = False  # 用于转换单词
        self.main_menu_chance = False  # 用来控制主菜单响应
        self.pronunciation_menu_chance = False  # change task in pronunciation test
        self.game_setting_menu_chance = False  # 用来控制主菜单响应
        self.present_word_menu_chance = False  # 控制展示单词菜单
        self.game_level_menu_chance = False  # 控制游戏界面的鼠标响应
        self.current_menu = self.main_menu  # 定义变量指向当前的菜单
        self.click_event = False  # 判断当前鼠标点击没有
        self.check_spelling = False  # 检查拼写
        self.pronunciation = True  # speak authentic pronunciation control by player (press Q )
        self.current_loop = 1  # 用来记录当前是第几轮循环
        self.player_score = 0  # 记录玩家的得分
        self.game_record, self.player_old_score = read_excel_game_record('saved_files/game_record.xls')  # 读取游戏的一些记录,返回的是列表
        self.learning_session_code = self.game_record  # 第几次学习，初始化为-1，每次开始学习都+1
        self.save_game_record = []  # 点击结束游戏以后要保存一些参数
        self.word_red_color_dic = {}  # 存放单词及其对应的错误字母列表

    # 检查事件，鼠标事件
    def check_Events(self):
        for event in pygame.event.get():  # 获得当前所有得事件
            if event.type == pygame.QUIT:  # 如果退出游戏,不允许在这退出游戏
                self.Main_Game_Running = False  # 直接退出游戏
            if event.type == pygame.MOUSEMOTION:  # 鼠标移动
                self.mouse_current_x, self.mouse_current_y = event.pos  # 获得当前鼠标的坐标
                self.mouse_rel_x, self.mouse_rel_y = event.rel  # 获得鼠标移动的相对距离
            if event.type == pygame.MOUSEBUTTONUP:  # 点击了鼠标
                self.click_event = False  # # 释放了鼠标
                self.mouse_click_x, self.mouse_click_y = event.pos
                self.pronunciation_menu_chance = True  # 用来控制语音检测模块的所有参数
                self.pronunciation_menu_chance = True  # 用来控制语音检测模块的所有参数
                self.main_menu_chance = True  # 主菜单的相应
                self.game_setting_menu_chance = True  # 游戏设置的响应
                self.present_word_menu_chance = True  # 控制展示单词菜单
                self.game_level_menu_chance = True  # 控制游戏界面的鼠标响应
            if event.type == pygame.MOUSEBUTTONDOWN:  # 点击了鼠标
                self.click_event = True  # 鼠标点击了左键，代表选中了字母
            if event.type == pygame.KEYDOWN:  # 按键按下事件
                if event.key == pygame.K_RETURN:  # 开始检查单词拼写
                    self.check_spelling = True
                if event.key == pygame.K_q:  # 开始单词发音
                    self.pronunciation = True

    # keys will not reset if not reset the key and use KEYUP cannot timely reset keys
    def reset_Keys(self):
        self.pronunciation_menu_chance = False  # in pronunciation test menu
        self.pronunciation_menu_chance = False  # release the Q keyboard
        self.main_menu_chance = False  # separate main menu from other menus
        self.game_setting_menu_chance = False  # separate game setting menu from other menus
        self.present_word_menu_chance = False  # separate present word menu from other menus
        self.game_level_menu_chance = False  # separate game level menu from other menus
        self.pronunciation = False  # reset pronunciation
        self.check_spelling = False  # reset check spelling

    # constantly running code
    def game_Loop(self):
            self.clock = pygame.time.Clock()  # set a clock
            self.clock.tick(60)  # 60 frame per minutes
            self.window.blit(self.GAME_BACKGROUND_PICTURE, (0, 0))  # draw the background to display surface
            self.check_Events()  # check game events


# 菜单的父类
class CreateMenu(object):
    def __init__(self, word_maker):
        self.word_maker = word_maker  # 主要是为了引用游戏屏幕，及参数
        self.mid_w, self.mid_h = self.word_maker.window_width / 2, self.word_maker.window_height / 2  # 得到屏幕的中心点

    # 加载菜单背景图片
    def load_Image(self, image_path):
        menu_image = pygame.image.load(image_path).convert_alpha()  # 根据文件路径加载菜单背景
        return menu_image

    # 在菜单面板上展示文字
    def draw_Menu_Text(self, surface, text, size, x, y, COLOR=(0, 0, 0)):
        font = pygame.font.Font(self.word_maker.Menu_Font, size)  # 文本的字体及大小
        text_surface = font.render(text, True, COLOR)  # 文本为白色
        text_rect = text_surface.get_rect()  # 相当于给当前的surface 框起来 这样比较容易获得和使用参数
        text_rect.center = (x, y)  # 文本的中心点的坐标
        surface.blit(text_surface, text_rect)  # 将这个文本画到后面的坐标上

    # 在主屏幕上展示菜单
    def display_Menu(self, surface, image_center):
        self.word_maker.window.blit(surface, image_center)


# 继承菜单父类，写第一页的菜单
class MainMenu(CreateMenu):
    def __init__(self, word_maker):  # 参数一定是实例化的游戏
        CreateMenu.__init__(self, word_maker)
        self.word_maker = word_maker  # 得到主游戏的屏幕
        self.image_1 = self.load_Image('Game_Pictures/menu.png')  # 开始游戏
        self.menu_image_1 = pygame.transform.scale(self.image_1, (220, 110))
        self.image_rect_1 = self.menu_image_1.get_rect()  # 获得image的参数
        self.image_2 = self.load_Image('Game_Pictures/menu.png')  # 游戏设置
        self.menu_image_2 = pygame.transform.scale(self.image_2, (220, 110))
        self.image_rect_2 = self.menu_image_2.get_rect()  # 获得image的参数
        self.image_3 = self.load_Image('Game_Pictures/menu.png')  # 结束游戏
        self.menu_image_3 = pygame.transform.scale(self.image_3, (220, 110))
        self.image_rect_3 = self.menu_image_3.get_rect()  # 获得image的参数
        self.image_4 = self.load_Image('Game_Pictures/menu.png')  # 发音检测
        self.menu_image_4 = pygame.transform.scale(self.image_4, (220, 110))
        self.image_rect_4 = self.menu_image_3.get_rect()  # 获得image的参数

    # 展示菜单内容
    def display_menu(self):
        # -------------------------------------------------------------------------
        # ‘开始游戏’模块
        self.image_rect_1.center = (
            self.mid_w - self.image_rect_1.width / 2, self.mid_h - self.image_rect_1.height / 2 - 150)
        # 如果鼠标活动在‘开始游戏’附近
        if self.image_rect_1.collidepoint(self.word_maker.mouse_current_x - self.image_rect_1.width / 2,
                                          self.word_maker.mouse_current_y - self.image_rect_1.height / 2):
            self.draw_Menu_Text(self.menu_image_1, '开始游戏', 39, self.image_rect_1.width / 2,
                                self.image_rect_1.height / 2, COLOR=self.word_maker.WHITE)
        else:
            self.draw_Menu_Text(self.menu_image_1, '开始游戏', 39, self.image_rect_1.width / 2,
                                self.image_rect_1.height / 2)
        self.display_Menu(self.menu_image_1, self.image_rect_1.center)  # 将开始游戏画到主屏幕上
        # 如果鼠标点击了‘开始游戏’选项
        if self.image_rect_1.collidepoint(self.word_maker.mouse_click_x - self.image_rect_2.width / 2,
                                          self.word_maker.mouse_click_y - self.image_rect_2.height / 2) and \
                self.word_maker.main_menu_chance:
            self.word_maker.remembered_words_number = 0  # 每次开始游戏都要将玩家已经记住的单词初始化为0
            self.word_maker.player_score = 0  # 也要将玩家的得分清0
            self.word_maker.word_loop = {}  # 记录玩家某一个单词用了几轮
            print('当前的session是', int(self.word_maker.learning_session_code))
            select_tasks(int(self.word_maker.learning_session_code))  # 挑选要玩的任务,但是有可能单词没写进去
            self.word_maker.present_word_menu = PresentWords(self.word_maker)  # 必须先写晚间再初始化每一次点击开始游戏都要重新初始化展示单词菜单
            self.word_maker.current_loop = 1  # 每次点击开始打游戏都要将轮数初始化
            if str(self.word_maker.learning_session_code) == '10':  # 如果learning session 到了10 要重新从0开始计数
                self.word_maker.learning_session_code = 0
            self.word_maker.current_menu = self.word_maker.present_word_menu  # # 如果鼠标点击的这个位置在方块2中，进入下一级菜单
        self.display_Menu(self.menu_image_1, self.image_rect_1.center)  # 将‘开始游戏画到屏幕上’
        # -------------------------------------------------------------------------
        # ‘游戏设置’模块
        self.image_rect_2.center = (
            self.mid_w - self.image_rect_2.width / 2, self.mid_h - self.image_rect_2.height / 2)
        # 如果鼠标活动在‘游戏设置’附近
        if self.image_rect_2.collidepoint(self.word_maker.mouse_current_x - self.image_rect_2.width / 2,
                                          self.word_maker.mouse_current_y - self.image_rect_2.height / 2):
            self.draw_Menu_Text(self.menu_image_2, '游戏设置', 39, self.image_rect_2.width / 2,
                                self.image_rect_2.height / 2, COLOR=self.word_maker.WHITE)
        else:
            self.draw_Menu_Text(self.menu_image_2, '游戏设置', 39, self.image_rect_2.width / 2,
                                self.image_rect_2.height / 2)
        # 如果鼠标点击了‘游戏设置’选项
        if self.image_rect_2.collidepoint(self.word_maker.mouse_click_x - self.image_rect_2.width / 2,
                                          self.word_maker.mouse_click_y - self.image_rect_2.height / 2) and \
                self.word_maker.main_menu_chance:
            self.word_maker.current_menu = self.word_maker.game_setting_menu  # # 如果鼠标点击的这个位置在方块2中，进入下一级菜单
        self.display_Menu(self.menu_image_2, self.image_rect_2.center)  # 将游戏设置画到主屏幕上
        # -------------------------------------------------------------------------
        # ‘发音学习’模块
        self.image_rect_4.center = (
            self.mid_w - self.image_rect_4.width / 2, self.mid_h - self.image_rect_4.height / 2 + 150)
        # 如果鼠标活动在‘发音学习’附近
        if self.image_rect_4.collidepoint(self.word_maker.mouse_current_x - self.image_rect_2.width / 2,
                                          self.word_maker.mouse_current_y - self.image_rect_2.height / 2):
            self.draw_Menu_Text(self.menu_image_4, '发音学习', 39, self.image_rect_4.width / 2,
                                self.image_rect_4.height / 2, COLOR=self.word_maker.WHITE)
        else:
            self.draw_Menu_Text(self.menu_image_4, '发音学习', 39, self.image_rect_4.width / 2,
                                self.image_rect_4.height / 2)
        # 如果鼠标选中了‘发音学习’菜单
        if self.image_rect_4.collidepoint(self.word_maker.mouse_click_x - self.image_rect_4.width / 2,
                                          self.word_maker.mouse_click_y - self.image_rect_4.height / 2) and \
                self.word_maker.main_menu_chance:
            self.word_maker.current_menu = self.word_maker.pronunciation_menu  # 切换到发音学习菜单
        self.display_Menu(self.menu_image_4, self.image_rect_4.center)  # 将发音学习画到主屏幕上
        # -------------------------------------------------------------------------
        # ‘游戏设置’模块
        self.image_rect_3.center = (
            self.mid_w - self.image_rect_3.width / 2, self.mid_h - self.image_rect_3.height / 2 + 300)
        # 如果鼠标活动在‘结束游戏’附近
        if self.image_rect_3.collidepoint(self.word_maker.mouse_current_x - self.image_rect_3.width / 2,
                                          self.word_maker.mouse_current_y - self.image_rect_3.height / 2):
            self.draw_Menu_Text(self.menu_image_3, '结束游戏', 39, self.image_rect_3.width / 2,
                                self.image_rect_3.height / 2, COLOR=self.word_maker.WHITE)
        else:
            self.draw_Menu_Text(self.menu_image_3, '结束游戏', 39, self.image_rect_3.width / 2,
                                self.image_rect_3.height / 2)
        # 如果鼠标选中了‘结束游戏’菜单
        if self.image_rect_3.collidepoint(self.word_maker.mouse_click_x - self.image_rect_3.width / 2,
                                          self.word_maker.mouse_click_y - self.image_rect_3.height / 2) and \
                self.word_maker.main_menu_chance:

            self.word_maker.Main_Game_Running = False  # 直接退出游戏
        self.display_Menu(self.menu_image_3, self.image_rect_3.center)  # 将结束游戏画到菜单上


# 继承菜单父类，写游戏设置的菜单
class GameSetting(CreateMenu):
    def __init__(self, word_maker):
        CreateMenu.__init__(self, word_maker)
        self.image_1 = self.load_Image('Game_Pictures/menu.png')  # 返回上一级菜单
        self.menu_image_1 = pygame.transform.scale(self.image_1, (220, 110))
        self.image_rect_1 = self.menu_image_1.get_rect()  # 获得image的参数

    # 现实游戏设置菜单
    def display_menu(self):
        # 设置图片的中心坐标
        self.image_rect_1.center = (
            self.mid_w - self.image_rect_1.width / 2, self.mid_h - self.image_rect_1.height / 2 - 150)
        # 如果鼠标在‘返回菜单’附近活动
        if self.image_rect_1.collidepoint(self.word_maker.mouse_current_x - self.image_rect_1.width / 2,
                                          self.word_maker.mouse_current_y - self.image_rect_1.height / 2):
            self.draw_Menu_Text(self.menu_image_1, '返回', 39, self.image_rect_1.width / 2,
                                self.image_rect_1.height / 2, COLOR=self.word_maker.WHITE)
        else:
            self.draw_Menu_Text(self.menu_image_1, '返回', 39, self.image_rect_1.width / 2, self.image_rect_1.height / 2)
        self.display_Menu(self.menu_image_1, self.image_rect_1.center)
        # 如果鼠标点击了‘返回菜单’菜单
        if self.image_rect_1.collidepoint(self.word_maker.mouse_click_x - self.image_rect_1.width / 2,
                                          self.word_maker.mouse_click_y - self.image_rect_1.height / 2):
            self.word_maker.current_menu = self.word_maker.main_menu  # 返回主菜单


# --------------------------------------------------------------------------------------------------------

# 游戏面板上展示文本
def draw_Text(surface, font, text, size, x, y):
    font = pygame.font.Font(font, size)  # 文本的字体及大小
    text_surface = font.render(text, True, (255, 255, 255))  # 文本为白色
    text_rect = text_surface.get_rect()  # 相当于给当前的surface 框起来 这样比较容易获得和使用参数
    text_rect.center = (x, y)  # 文本的中心点的坐标
    surface.blit(text_surface, text_rect)  # 将这个文本画到后面的坐标上
