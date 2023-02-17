"""
就剩强化学习
"""


from Common_Functions import *
from datetime import datetime, timedelta
import copy
from feedback_training import GameFeedback


# 实例化肯定是直接实例化库里的所有单词，因为所有的参数都已经确定了
class GameLevel(object):
    # 游戏关卡里有几个重要的参数，第一个机会次数,第二个参数是单词的长度
    def __init__(self, word_maker):
        self.word_maker = word_maker  # 得到主游戏
        self.surface_width, self.surface_height = self.word_maker.window.get_size()  # 得到主游戏的屏幕的长和宽
        # 创建一个和主屏幕大小一样的Surface
        self.game_level_surface = pygame.Surface((self.surface_width, self.surface_height))
        # 单词，音标，汉语，文件标记，单词长度，难度，机会次数，是否有中文，时间，是否有迷惑字母，是否有音标
        self.tasks_parameters_list = read_tasks_parameters('Word_Pool/game_level_0.xls')
        # 这个字典用来选择迷惑字母
        self.letters_dic = {'a': ['e', 'i', 'o', 'u', 'y'], 'b': ['d', 'p', 'q', 't'], 'c': ['k', 's', 't', 'z'],
                            'd': ['b', 'p', 'q', 't'],
                            'e': ['a', 'o', 'i', 'u', 'y'], 'f': ['v', 'w'], 'g': ['h', 'j'], 'h': ['m', 'n'],
                            'i': ['a', 'e', 'o', 'y'],
                            'j': ['g', 'i'], 'k': ['c', 'g'], 'l': ['i', 'r'], 'm': ['h', 'n'], 'n': ['h', 'm'],
                            'o': ['a', 'e', 'i', 'u', 'y'],
                            'p': ['b', 'd', 'q', 't'], 'q': ['b', 'd', 'p', 't'], 'r': ['l', 'v'], 's': ['c', 'z'],
                            't': ['c', 'd'], 'u': ['v', 'w'],
                            'v': ['f', 'u', 'w'], 'w': ['f', 'v'], 'x': ['s', 'z'], 'y': ['e', 'i'], 'z': ['c', 's']}

        self.BLOCK_SIZE = 90  # 设置框的大小
        self.task_index = 0  # 追踪当前是第几个任务
        self.start_time = datetime.now()  # 玩游戏开始的时间
        self.task_second = 0  # 记录玩家用了多少时间记住这个单词
        self.time_change = True  # 每换一个单词，需要重新记录时间
        self.countdown = 0  # 初始化倒计时
        self.letter_coordinate = {}  # 创建一个字典{字母_顺序：坐标}，为了避免某个单词有重复字母，导致字典重复赋值
        self.letter_Rect = {}  # 创建一个字典{字母_顺序：Rect}
        self.BLACK = (0, 0, 0)  # 字母是黑色的
        self.font = pygame.font.Font("Game_Fonts/chinese_pixel_font.TTF", 60)  # 字母的字体，大小
        self.task_change = True  # 控制切换任务的开关
        self.letter_contact = False  # 判断当前有没有选中字母
        self.current_letter = 0  # 当前选中的字母是是么
        self.letter_original_coordinate = 0  # 保存字母的原始坐标
        self.current_attempt = 0  # 当前的尝试次数
        self.contact_rect_list = []  # 存储已经存在于框中的Rect
        self.occupied_rect = []  # 最终占用的rect
        self.player_spelling_rect = {}  # 用来记录玩家当前的拼写和对应的Rect
        self.player_spelling = ''  # 用来记录玩家的拼写
        self.player_used_spelling = []  # 用来记录玩家在某一关的所有拼写，主要使用给颜色反馈
        self.time_pause = False  # 这个是为了让玩家在某一个任务结束后，时间不在减少
        self.lock_time = 0  # 当耗时结束，代表了没有完成任务，则锁定时间
        self.save_player_feature = []  # 保存玩家每一次表现的记录,然后保存在文件中
        self.save_player_features = []  # 将玩家的数据保存为列表
        self.press_Q = 0  # 玩家主动听发音的次数
        self.press_Enter = 0  # 玩家被动听发音的次数
        self.yellow_color = 0  # 统计总共的黄色的次数
        self.green_color = 0  # 统计总共的绿色的个数
        self.red_color = 0  # 统计总共的红色的个数
        self.red_color_dic = {}  # 统计哪个字母迷惑了玩家
        self.red_letters_list = []  # 将迷惑单词的字典改为列表的形式

    # 该函数的作用是将单词拆分为字母，并固定其初始的位置
    def split_Word(self, word):  # 首先输入一个单词
        letter_list = []  # 创建一个列表准备读取单词的字母
        self.letter_coordinate = {}  # 每次拆分单词都要归0，不然保持以前的记录
        self.letter_Rect = {}  # 将所有需要清零的都清零
        self.current_word_length = len(word)  # 获得当前单词的长度
        self.current_word = word  # 获得当前字母的拼写
        if self.tasks_parameters_list[self.task_index][9]:  # 需要加一个判断，本难度需不需要迷惑字母
            # 要将字典按照对应的单词替换，然后再随机选择字母 # 单词之前出现过错误
            # 如果不空,且当前的单词之前出过错误
            if self.word_maker.word_red_color_dic and (word in list(self.word_maker.word_red_color_dic.keys())):
                letters_dic_new = copy.deepcopy(self.letters_dic)  # 复制一份新坐标
                for key, item in self.word_maker.word_red_color_dic[word].items():  # 循环对应错误的字典
                    letters_dic_new[key] = item  # 将新的迷惑字母的值赋值
                print(letters_dic_new)
                print(self.letters_dic)
                for letter in word:  # 循环读取字母
                        confusing_letter = random.choice(letters_dic_new[letter])  # 随机挑选一个迷惑字母
                        letter_list.append(confusing_letter)  # 将迷惑字谜加到列表中
                        letter_list.append(letter)  # 将字母加入到列表中
                # 如果是空值
            else:  # 如果玩家没有错误字母的记录，还是按照原错误挑选
                for letter in word:  # 循环读取字母
                    confusing_letter = random.choice(self.letters_dic[letter])  # 随机挑选一个迷惑字母
                    letter_list.append(confusing_letter)  # 将迷惑字谜加到列表中
                    letter_list.append(letter)  # 将字母加入到列表中
        else:
            for letter in word:  # 循环读取字母
                letter_list.append(letter)  # 将字母加入到列表中
        print(letter_list)
        random.shuffle(letter_list)  # 将里面加入列表的字母的顺序随机打乱
        letter_x_coordinate = 0  # 横坐标从0开始
        letter_y_coordinate = 50  # 纵坐标从30开始
        increase = 70  # 字母与字母之间的间距为70
        for i in range(len(letter_list)):  # 循环字母的个数
            coordinate = [letter_x_coordinate, letter_y_coordinate]  # 每个字母的坐标都需要重新刷新
            letter_x_coordinate += increase
            self.letter_coordinate[letter_list[i] + '_' + str(i)] = coordinate  # 因为重复字母在字典里不能存在
            if len(self.letter_coordinate) == len(self.current_word):
                letter_x_coordinate = 0  # 横坐标从0开始
                letter_y_coordinate = 120  # 纵坐标从120开始
        self.letter_original_coordinate = copy.deepcopy(self.letter_coordinate)  # 复制一份保留原坐标

    # 该函数的作用是将所有的字母画到主屏幕上
    def draw_Letters(self):
        for key, coordinate in self.letter_coordinate.items():  # 循环读取{字母_顺序：坐标}
            # 直接加载背景图片
            letter_surface = pygame.image.load('Game_Pictures/letter_background_color.png')
            letter_surface.set_alpha(200)  # 设置surface的透明度
            letter = key.split('_')[0]  # 获得字母
            text_surface = self.font.render(letter, True, self.BLACK)  # 要写的文本，以及字体颜色
            text_rect = text_surface.get_rect()  # 相当于给当前的字母的surface 框起来 这样比较容易获得和使用参数
            letter_surface_rect = letter_surface.get_rect()  # 给字母的背景屏幕框起来
            letter_surface_rect.topleft=(coordinate[0], coordinate[1])
            self.letter_Rect[key] = letter_surface_rect  # 字典{字母_位置：Rect}
            letter_surface.blit(text_surface, (30 - text_rect.width / 2, 30 - text_rect.height / 2))  # 将字母放在中间
            self.game_level_surface.blit(letter_surface, letter_surface_rect.topleft)  # 控制画到游戏屏幕的位置

    # 该函数实现字母随着鼠标移动
    def letter_move(self):
        # 第一步 当前没有选中任何字母，所以首先要选中字母
        if not self.letter_contact:  # 如果当前没有选中单词
            # 首先要判定鼠标点击了哪个字母
            for letter, Rect in self.letter_Rect.items():
                # 鼠标点击事件是点了以后一直响应
                if Rect.collidepoint(self.word_maker.mouse_current_x, self.word_maker.mouse_current_y) and self.word_maker.click_event:
                    game_Sound('game_sound/mouse_click_1.mp3', 0.2)  # 游戏鼠标的声音
                    self.letter_contact = True  # 选中了这个字母
                    self.current_letter = letter  # 则得到当前的这个字母
                    break  # 只要选中了就不循环了
        # 第二步 如果选中了字母，让字母随着鼠标移动
        if self.letter_contact:
            # 减30是为了让鼠标时刻在图片的正中心
            self.letter_coordinate[self.current_letter][0] = self.word_maker.mouse_current_x - 30 + self.word_maker.mouse_rel_x
            self.letter_coordinate[self.current_letter][1] = self.word_maker.mouse_current_y - 30 + self.word_maker.mouse_rel_y
        # 第三步，鼠标移动的时候一直检测有没有和框发生碰撞
        if self.word_maker.click_event and self.letter_contact:
            self.Rect_index = self.letter_Rect[self.current_letter].collidelist(self.Blocks_Rect[self.current_attempt])
        # 第四步 如果此时松开了鼠标,但是还在字母上
        if not self.word_maker.click_event and self.letter_contact:
            # 松开了鼠标，字母在框中,而且该框没有被占用不等于-1才在框中
            if self.Rect_index != -1 and self.Rect_index not in self.contact_rect_list:
                game_Sound('game_sound/mouse_click_1.mp3', 0.2)  # 放入框中的声音
                self.letter_coordinate[self.current_letter][0] = self.Blocks_Rect[self.current_attempt][self.Rect_index].x+15
                self.letter_coordinate[self.current_letter][1] = self.Blocks_Rect[self.current_attempt][self.Rect_index].y+15
                self.contact_rect_list.append(self.Rect_index)  # 表示了这个框已经被占用
            else:
                self.letter_coordinate[self.current_letter][0] = \
                    self.letter_original_coordinate[self.current_letter][0]
                self.letter_coordinate[self.current_letter][1] = \
                    self.letter_original_coordinate[self.current_letter][1]
            self.letter_contact = False
        # 第五步：查看框中的坐标，不在框中的索引要从已经占用的删除
        self.occupied_rect = []
        self.player_spelling_rect = {}
        self.player_spelling = ''

        for letter, Rect in self.letter_Rect.items():
            Rect_index = Rect.collidelist(self.Blocks_Rect[self.current_attempt])
            if Rect_index != -1 and Rect_index not in self.occupied_rect:
                self.occupied_rect.append(Rect_index)
                self.player_spelling_rect[Rect.x] = letter
        # 循环框内的坐标排序得到玩家的拼写
        for index in sorted(self.player_spelling_rect):
            self.player_spelling += self.player_spelling_rect[index].split('_')[0]  # 得到玩家的拼写
        # 循环之前占用的框
        for i in self.contact_rect_list:
            # 如果已经不在框中，就将该索引删除
            if i not in self.occupied_rect:
                self.contact_rect_list.remove(i)

    # 颜色和位置，必须锁定当前的拼写，不随字母的移动而改变的方式，然后画图，还得一直显示在屏幕上，所有的玩家拼写都会循环一遍，所以我统计最后一次就行
    # 统计每种颜色的个数，以及红色字母放到一个列表中（最可能迷惑玩家的字母），这个地方可以自适应迷惑玩家字母，有很多想法可以实现
    def indicator_Spelling(self, player_spelling, attempt_number):
        for index in range(len(self.current_word)):  # 一个字母一个字母判断
            if player_spelling[index] not in self.current_word:  # 如果不在单词中
                fill_surface = pygame.Surface((80, 80))
                fill_surface.fill((200, 0, 0))  # 红色
                text_surface = self.font.render(player_spelling[index], True, self.BLACK)  # 要写的文本，以及字体颜色
                text_rect = text_surface.get_rect()  # 相当于给当前的surface 框起来 这样比较容易获得和使用参数
                fill_surface.blit(text_surface, (40 - text_rect.width / 2, 40 - text_rect.height / 2))  # 将字母放在中间
                self.Blocks_Surface.blit(fill_surface, (index * 90+5, attempt_number * 90+5))

            elif player_spelling[index] == self.current_word[index]:  # 如果在正确的位置
                fill_surface = pygame.Surface((80, 80))
                fill_surface.fill((0, 200, 0))  # 绿色
                text_surface = self.font.render(player_spelling[index], True, self.BLACK)  # 要写的文本，以及字体颜色
                text_rect = text_surface.get_rect()  # 相当于给当前的surface 框起来 这样比较容易获得和使用参数
                fill_surface.blit(text_surface, (40 - text_rect.width / 2, 40 - text_rect.height / 2))  # 将字母放在中间
                self.Blocks_Surface.blit(fill_surface, (index * 90+5, attempt_number * 90+5))
            else:
                fill_surface = pygame.Surface((80, 80))
                fill_surface.fill((200, 200, 0))  # 黄色
                text_surface = self.font.render(player_spelling[index], True, self.BLACK)  # 要写的文本，以及字体颜色
                text_rect = text_surface.get_rect()  # 相当于给当前的surface 框起来 这样比较容易获得和使用参数
                fill_surface.blit(text_surface, (40 - text_rect.width / 2, 40 - text_rect.height / 2))  # 将字母放在中间
                self.Blocks_Surface.blit(fill_surface, (index * 90+5, attempt_number * 90+5))

    # 找到正确，错误，黄色的字母，{'b': ['a'], 'e': ['l', 'i'], 'r': ['q']}
    def find_color_numbers(self, player_spelling):
        for index in range(len(self.current_word)):  # 一个字母一个字母判断
            if player_spelling[index] not in self.current_word:  # 如果不在单词中
                # 如果当前的字母还不在红色列表中
                if self.current_word[index] not in self.red_color_dic:
                    self.red_letters_list.append(player_spelling[index])  # 将错误字母添加到列表中
                    self.red_color_dic[self.current_word[index]] = self.red_letters_list  # 将列表与字母对应
                    self.red_letters_list = []  # 将列表元素清空
                else:  # 如果已经有对应的错误,就在对应的列表中再加入这个字母 {'b': ['a'], 'e': ['l', 'i'], 'r': ['q']}
                    self.red_color_dic[self.current_word[index]].append(player_spelling[index])
                self.red_color += 1
            elif player_spelling[index] == self.current_word[index]:  # 如果在正确的位置
                self.green_color += 1
            else:
                self.yellow_color += 1

    # 检查玩家的拼写
    def check_Spelling(self):
        # 如果按了回车键，而且确实已经拼写完毕
        if self.word_maker.check_spelling and len(self.player_spelling) == len(self.current_word):
            # 第一件事要发音
            game_Sound('UK_Pronunciation/' + self.tasks_parameters_list[self.task_index][0] + '.mp3')
            self.press_Enter += 1  # 玩家被动听发音的次数
            # 记录检查开始的时间
            self.start_check_time = self.gameplay_time  # 记录游戏运行的时间，很重要，需要控制游戏进程
            # 将所有的字母送回原坐标
            for key in self.letter_coordinate.keys():
                self.letter_coordinate[key][0] = self.letter_original_coordinate[key][0]
                self.letter_coordinate[key][1] = self.letter_original_coordinate[key][1]
            # 玩家检查说明已经拼写完毕，所以要将玩家的拼写记录下来
            self.player_used_spelling.append(self.player_spelling)
            self.player_spelling = []  # 将这个记录清空
            self.current_attempt += 1

        # 如果玩家的拼写不为空，则将内容一直展示到频幕上
        if self.player_used_spelling:
            for i in range(len(self.player_used_spelling)):
                self.indicator_Spelling(self.player_used_spelling[i], i)
                # 如果当前的单词拼写正确，而且展示了5秒,则跳到下一个单词
                if self.player_used_spelling[i] == self.current_word:
                    self.show_Success()  # 展示成功后得反馈
                    self.time_pause = True  # 让进度条时间暂停
                    if self.gameplay_time > self.start_check_time + 2000:  # 如果回答正确展示2秒
                        # 如果当前难度小于等于3回答正确加1分
                        if self.tasks_parameters_list[self.task_index][5] <= 3:
                            self.word_maker.player_score += 1
                        # 如果当前难度等于4回答正确加2分
                        if self.tasks_parameters_list[self.task_index][5] == 4:
                            self.word_maker.player_score += 2
                        # 如果玩家在第四难度答对了且省了几轮则加轮数乘以2分
                        self.save_player_feature.append(self.task_second)  # 添加玩家用了多少时间
                        self.save_player_feature.append(self.current_attempt)  # 添加玩家用了多少次机会
                        self.save_player_feature.append(1)  # 1代表玩家对了
                        self.save_player_feature.append(self.press_Q)  # 玩家主动听发音的次数
                        self.save_player_feature.append(self.press_Enter)  # 玩家被动听发音的次数
                        for i in range(len(self.player_used_spelling)):
                            self.find_color_numbers(self.player_used_spelling[i])
                        # 将红绿蓝的次数添加进去
                        self.save_player_feature = self.save_player_feature+[self.red_color,self.green_color,self.yellow_color]
                        self.save_player_features.append(self.save_player_feature)
                        game_Sound('game_sound/right.wav')  # 回答正确的反馈
                        self.player_spelling = []  # 将这个记录清空
                        # 如果当前单词存在错误字母，才一一对应
                        if self.red_color_dic:
                            self.word_maker.word_red_color_dic[self.current_word] = self.red_color_dic  # 将单词与错误的字母对应起来
                        # {单词：[音标，翻译，玩家拼写, 当前难度, 文件label]}，这是为了测试玩家不会的单词
                        self.word_maker.finished_tasks[self.current_word] = \
                            [self.tasks_parameters_list[self.task_index][1], self.tasks_parameters_list[self.task_index][2],
                             self.player_used_spelling[-1], self.tasks_parameters_list[self.task_index][5],
                             self.tasks_parameters_list[self.task_index][3]]# 这个是完全拼写正确
                        # {单词：[音标，翻译，玩家拼写]} 这个是为了让玩家学习全部单词
                        self.word_maker.all_tasks[self.current_word] = \
                            [self.tasks_parameters_list[self.task_index][1], self.tasks_parameters_list[self.task_index][2],
                             self.player_used_spelling[-1]]
                        self.player_used_spelling = []  # 将过去的记录也清零，不然和下面得判定重复
                        self.task_change = True  # 修改参数
                        self.task_index += 1  # 并且进行到下一个任务

        # 如果已经到了最后一个单词
        if self.task_index == len(self.tasks_parameters_list):

            write_player_features_xls(self.save_player_features)  # 将玩家的数据写入文件

            self.task_index = 0  # 让任务是第一个
            self.word_maker.feedback_train_menu = GameFeedback(self.word_maker)  # 实例化反馈菜单
            self.word_maker.current_menu = self.word_maker.feedback_train_menu  # 到反馈菜单

        # 如果机会已经用完了，而且最后一次的拼写还错误,而且拼写错误，也要展示5秒并跳到下一个任务
        if self.current_word not in self.player_used_spelling and \
                len(self.player_used_spelling) == self.tasks_parameters_list[self.task_index][6]:
                self.show_Failure()  # 回答错误错误显示的反馈
                self.time_pause = True  # 让进度条时间暂停
                if self.gameplay_time > self.start_check_time + 4000:
                    self.save_player_feature.append(self.task_second)  # 添加玩家用了多少时间
                    self.save_player_feature.append(self.current_attempt)  # 添加玩家用了多少次机会
                    self.save_player_feature.append(0)  # 0代表玩家错了
                    self.save_player_feature.append(self.press_Q)  # 玩家主动听发音的次数
                    self.save_player_feature.append(self.press_Enter)  # 玩家被动听发音的次数
                    for i in range(len(self.player_used_spelling)):
                        self.find_color_numbers(self.player_used_spelling[i])

                    self.save_player_feature = self.save_player_feature + [self.red_color, self.green_color,
                                                                           self.yellow_color]
                    self.save_player_features.append(self.save_player_feature)
                    # 如果当前单词存在错误字母，才一一对应
                    if self.red_color_dic:
                        self.word_maker.word_red_color_dic[self.current_word] = self.red_color_dic  # 将单词与错误的字母对应起来
                    # {单词：[音标，翻译，玩家拼写，任务难度,文件label]}
                    game_Sound('game_sound/wrong.wav')  # 回答错误的反馈
                    self.word_maker.finished_tasks[self.current_word] = \
                        [self.tasks_parameters_list[self.task_index][1], self.tasks_parameters_list[self.task_index][2],
                         self.player_used_spelling[-1],self.tasks_parameters_list[self.task_index][5],
                         self.tasks_parameters_list[self.task_index][3]]
                    # {单词：[音标，翻译，玩家拼写]} 这个是为了让玩家学习全部单词
                    self.word_maker.all_tasks[self.current_word] = \
                        [self.tasks_parameters_list[self.task_index][1], self.tasks_parameters_list[self.task_index][2],
                         self.player_used_spelling[-1]]
                    self.player_spelling = []  # 将这个记录清空
                    self.task_change = True  # 修改参数
                    self.task_index += 1  # 并且进行到下一个任务
        # 如果已经到了最后一个单词
        if self.task_index == len(self.tasks_parameters_list):
            write_player_features_xls(self.save_player_features)  # 将玩家的数据写入文件
            self.task_index = 0  # 让任务是第一个
            self.word_maker.feedback_train_menu = GameFeedback(self.word_maker)  # 实例化反馈菜单
            self.word_maker.current_menu = self.word_maker.feedback_train_menu  # 到反馈菜单

    # 实现画表格的功能, 在画线之前，要判断本次的单词的字母数，以及本关的难度
    def draw_Blocks(self, chance, word_length):
        self.Blocks_Rect = [[] for i in range(chance)]  # 按照行和列把每一个格都框起来
        self.Blocks_Surface = pygame.Surface(
            (word_length * self.BLOCK_SIZE+2, chance * self.BLOCK_SIZE+2))  # 创建一个屏幕
        self.Blocks_Surface.fill((210, 210, 210))  # 填充屏幕的颜色
        # 这个循环是画横线的，代表的是给多少次机会
        for j in range(0, chance+1, 1):
            pygame.draw.line(self.Blocks_Surface, (0, 0, 0), (0, j * self.BLOCK_SIZE),
                             (word_length * self.BLOCK_SIZE, j * self.BLOCK_SIZE), 2)
        # 这个循环是画竖线的，代表的是这个单词有多少个字母
        for i in range(0, word_length+1, 1):
            pygame.draw.line(self.Blocks_Surface, (0, 0, 0), (i * self.BLOCK_SIZE, 0),
                             (i * self.BLOCK_SIZE, chance * self.BLOCK_SIZE), 2)

        # 获得每一个格子的Rect并存入列表中,并且纵坐标要加200匹配到图中的坐标
        for j in range(chance):
            for i in range(word_length):
                self.Blocks_Rect[j].append(
                    pygame.Rect(i * self.BLOCK_SIZE, j * self.BLOCK_SIZE + 200, self.BLOCK_SIZE, self.BLOCK_SIZE))

    # 写字功能
    def draw_Menu_Text(self, font, text, size, x, y, COLOR=(0, 0, 0)):
        font = pygame.font.Font(font, size)  # 文本的字体及大小
        text_surface = font.render(text, True, COLOR)  # 默认文本为黑色
        text_rect = text_surface.get_rect()  # 相当于给当前的surface 框起来 这样比较容易获得和使用参数
        text_rect.center = (x, y)  # 文本的中心点的坐标
        self.game_level_surface.blit(text_surface, text_rect)  # 居中对齐
        return text_rect

    # 写字功能
    def draw_Left_Text(self, font, text, size, x, y, COLOR=(0, 0, 0)):
        font = pygame.font.Font(font, size)  # 文本的字体及大小
        text_surface = font.render(text, True, COLOR)  # 默认文本为黑色
        text_rect = text_surface.get_rect()  # 相当于给当前的surface 框起来 这样比较容易获得和使用参数
        text_rect.topleft = (x, y)  # 文本的中心点的坐标
        self.game_level_surface.blit(text_surface, text_rect)  # 左对齐

    # 是否展示汉语翻译以及音标
    def draw_word_parameters(self):
        #  是否展示任务
        if self.tasks_parameters_list[self.task_index][7]:
            self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '当前任务:' +
                                self.tasks_parameters_list[self.task_index][2], 40, 0, 0)
        else:
            pass
        # 是否展示音标
        if self.tasks_parameters_list[self.task_index][10]:
            self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '音标:', 40, 340, 0)
            self.draw_Left_Text("Game_Fonts/phonetic.ttf", self.tasks_parameters_list[self.task_index][1],
                                30, 440, 0)
        else:
            pass
        # 如果玩家按下了Q键，则单词发音
        if self.word_maker.pronunciation:
            game_Sound('UK_Pronunciation/' + self.tasks_parameters_list[self.task_index][0] + '.mp3')
            self.press_Q += 1  # 玩家主动听发音的次数
            if self.press_Q == 3:  # 如果玩家按3次以上的发音
                self.word_maker.player_score += 1  # 玩家的分数加1
        # 右边的线
        pygame.draw.line(self.game_level_surface, (0, 0, 0), (720, 0), (720, 743), 2)
        # 右边中间的线
        pygame.draw.line(self.game_level_surface, (0, 0, 0), (720, 250), (self.surface_width, 250), 2)

    def draw_Text(self, surface, font_path, size, text, font_color, center_x, center_y):
        font = pygame.font.Font(font_path, size)  # 得到想要用的字体，以及字体的大小
        text_surface = font.render(text, True, font_color)  # 要写的文本，以及字体颜色
        text_rect = text_surface.get_rect()  # 相当于给当前的surface 框起来 这样比较容易获得和使用参数
        text_rect.center = (center_x, center_y)  # 文本要显示的位置
        surface.blit(text_surface, text_rect)

    # 游戏失败要展示的东西 翻译，单词 还要能强制发音
    def show_Success(self):
        failure_surface = pygame.Surface((720, 240))  # 先创建一个720*400大小的屏幕
        failure_surface.fill(self.word_maker.BGC)  # 将屏幕填充为白色
        self.draw_Text(failure_surface, "Game_Fonts/chinese_pixel_font.TTF", 30,
                  '正确,2秒后切到下一个任务!', (255, 0, 0), 360, 50)  # 这个是显示翻译
        self.draw_Text(failure_surface, "Game_Fonts/chinese_pixel_font.TTF", 40,
                  self.current_word, (255, 0, 0), 100, 150)  # 这个是显示英语单词
        self.draw_Text(failure_surface, "Game_Fonts/phonetic.ttf", 40,
                  self.tasks_parameters_list[self.task_index][1], (255, 0, 0), 400, 150)  # 这个是显示音标
        self.draw_Text(failure_surface, "Game_Fonts/chinese_pixel_font.TTF", 40,
                  self.tasks_parameters_list[self.task_index][2], (255, 0, 0), 600, 150)  # 这个是显示翻译
        self.game_level_surface.blit(failure_surface, (0, 500))

    # 游戏失败要展示的东西 翻译，单词 还要能强制发音
    def show_Failure(self):
        failure_surface = pygame.Surface((720, 240))  # 先创建一个720*400大小的屏幕
        failure_surface.fill(self.word_maker.BGC)  # 将屏幕填充为白色
        self.draw_Text(failure_surface, "Game_Fonts/chinese_pixel_font.TTF", 30,
                       '加油,4秒后切到下一个任务!', (255, 0, 0), 360, 50)  # 这个是显示翻译
        self.draw_Text(failure_surface, "Game_Fonts/chinese_pixel_font.TTF", 40,
                       self.current_word, (255, 0, 0), 100, 150)  # 这个是显示英语单词
        self.draw_Text(failure_surface, "Game_Fonts/phonetic.ttf", 40,
                       self.tasks_parameters_list[self.task_index][1], (255, 0, 0), 400, 150)  # 这个是显示音标
        self.draw_Text(failure_surface, "Game_Fonts/chinese_pixel_font.TTF", 40,
                       self.tasks_parameters_list[self.task_index][2], (255, 0, 0), 600, 150)  # 这个是显示翻译
        self.game_level_surface.blit(failure_surface, (0, 500))

    # 画当前的关卡需要的时间
    def draw_progress_bar(self, time):
        # 画一个progress bar，控制学习时间，实现2分钟倒计时按钮
        pygame.draw.rect(self.game_level_surface, (0, 0, 0),
                         ((0, self.surface_height - 60), (self.surface_width, 50)),
                         width=4)  # 首先画一个边框
        # 这个是为了控制每一个单词的时间不一样
        if self.time_change:
            self.countdown = time  # 将这个单词的任务时间给一个常数
            self.decrease_width = self.surface_width / time  # 1秒减少多少宽度
            self.time_change = False  # 将开关关闭
        # 检查完毕任务完成，则时间要暂停
        if not self.time_pause:
            self.task_second = time - self.countdown  # 记录学生玩这一关用了多少秒
        if 0 <= self.task_second <= 20:
            pygame.draw.rect(self.game_level_surface, (190, 190, 190),
                             ((self.task_second * self.decrease_width, self.surface_height - 56),
                              (self.surface_width - self.task_second * self.decrease_width, 42)))  # 画进度条
        elif 20 <= self.task_second <= 40:
            pygame.draw.rect(self.game_level_surface, (180, 180, 180),
                             ((self.task_second * self.decrease_width, self.surface_height - 56),
                              (self.surface_width - self.task_second * self.decrease_width, 42)))  # 画进度条
        else:
            pygame.draw.rect(self.game_level_surface, (170, 170, 170),
                             ((self.task_second * self.decrease_width, self.surface_height - 56),
                              (self.surface_width - self.task_second * self.decrease_width, 42)))  # 画进度条

        # 展示任务时长
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '任务时长:' +
                            str(self.tasks_parameters_list[self.task_index][8])+'秒', 40, 720, 0)
        # 当前任务难度
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '任务难度:'+str(self.tasks_parameters_list[self.task_index][5])+'级'
                            , 40, 720, 50)
        # 展示当前得分
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '当前得分:'+str(self.word_maker.player_score), 40, 720, 100)
        # 首先计算还剩多少个单词
        remaining_tasks = len(self.tasks_parameters_list) - self.task_index
        # 剩余任务数，展示本轮还剩多少单词
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '剩余任务:' + str(remaining_tasks), 40, 720, 150)
        # 统计已经记住的单词
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '记住单词:'+str(self.word_maker.remembered_words_number), 40, 720, 200)
        # 操作指引
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '操作指引', 40, 720, 250)
        # 拖动单词
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '字母拖入框中拼写', 30, 720, 302)
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '字母拖出框取消拼写', 30, 720, 332)
        # 操作指引
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '方框的行数代表机会', 30, 720, 372)
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '次数,请按行拖入字母', 30, 720, 402)
        # 操作指引
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '单词发音:Q键', 30, 720, 444)
        # 操作指引
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '每行结束，按Enter', 30, 720, 486)
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '键检查拼写', 30, 720, 516)
        # 操作指引
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '红色:单词没有该字母', 30, 720, 558)
        # 操作指引
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '黄色:单词中有该字母', 30, 720, 600)
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '但位置不正确', 30, 720, 630)
        # 操作指引
        self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", '绿色:字母完全正确', 30, 720, 672)

        # 控制单词展示的时间
        if datetime.now() > self.start_time + timedelta(seconds=1):  # 如果时间间隔相差一秒
            self.start_time = datetime.now()  # 将现在的时间给过去的时间
            self.countdown -= 1
            if self.countdown == -1:  # 如果时间结束，则进入游戏界面
                self.time_pause = True  # 让进度条时间暂停
                self.lock_time = self.gameplay_time  # 锁定时间耗完的游戏时间
        # 时间已经结束，要展示反馈
        if self.countdown < 0:
            self.show_Failure()
            # 展示5秒以后，进入下一个任务
            if self.gameplay_time > self.lock_time + 4000:
                self.save_player_feature.append(self.task_second)  # 添加玩家用了多少时间
                self.save_player_feature.append(self.current_attempt)  # 添加玩家用了多少次机会
                self.save_player_feature.append(0)  # 0代表玩家错了
                self.save_player_feature.append(self.press_Q)  # 玩家主动听发音的次数
                self.save_player_feature.append(self.press_Enter)  # 玩家被动听发音的次数
                for i in range(len(self.player_used_spelling)):
                    self.find_color_numbers(self.player_used_spelling[i])
                self.save_player_feature = self.save_player_feature + [self.red_color, self.green_color,
                                                                       self.yellow_color]
                self.save_player_features.append(self.save_player_feature)
                # 如果当前单词存在错误字母，才一一对应
                if self.red_color_dic:
                    self.word_maker.word_red_color_dic[self.current_word] = self.red_color_dic  # 将单词与错误的字母对应起来
                game_Sound('game_sound/wrong.wav')  # 回答错误的反馈
                # 如果时间结束了,拼写不为空才记录最后一次的拼写,
                if self.player_used_spelling:
                    # {单词：[音标，翻译，玩家拼写，任务难度,文件label]}
                    self.word_maker.finished_tasks[self.current_word] = \
                        [self.tasks_parameters_list[self.task_index][1], self.tasks_parameters_list[self.task_index][2],
                         self.player_used_spelling[-1], self.tasks_parameters_list[self.task_index][5],
                         self.tasks_parameters_list[self.task_index][3]]
                    # {单词：[音标，翻译，玩家拼写]} 这个是为了让玩家学习全部单词
                    self.word_maker.all_tasks[self.current_word] = \
                        [self.tasks_parameters_list[self.task_index][1], self.tasks_parameters_list[self.task_index][2],
                         self.player_used_spelling[-1]]
                # 如果拼写为空# {单词：[音标，翻译，未完成，任务难度，文件label]}
                else:
                    self.word_maker.finished_tasks[self.current_word] = \
                        [self.tasks_parameters_list[self.task_index][1], self.tasks_parameters_list[self.task_index][2],
                         '未完成', self.tasks_parameters_list[self.task_index][5], self.tasks_parameters_list[self.task_index][3]]
                    # {单词：[音标，翻译，玩家拼写]} 这个是为了让玩家学习全部单词
                    self.word_maker.all_tasks[self.current_word] = \
                        [self.tasks_parameters_list[self.task_index][1], self.tasks_parameters_list[self.task_index][2],
                         '未完成']
                self.time_change = True  # 将开关打开
                self.task_index += 1  # 并且进行到下一个任务
                # 如果已经到了最后一个单词
                if self.task_index == len(self.tasks_parameters_list):
                    write_player_features_xls(self.save_player_features)  # 将玩家的数据写入文件
                    self.task_index = 0  # 让任务是第一个
                    self.word_maker.feedback_train_menu = GameFeedback(self.word_maker)  # 实例化反馈菜单
                    self.word_maker.current_menu = self.word_maker.feedback_train_menu
                self.task_change = True  # 时间结束了需要切换单词

    # 展示玩家已经完成的任务
    def show_Finished_Task(self):
        x_coordinate = 0  # 初始横坐标
        y_coordinate = 500  # 初始纵坐标
        # 获得所有的key
        for key, items in self.word_maker.finished_tasks.items():
            # 先画单词
            self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", key, 40, x_coordinate, y_coordinate)
            x_coordinate += 210
            # 再画后续的音标，翻译
            for index in range(len(items)):
                if index == 1:  # 汉语
                    self.draw_Left_Text("Game_Fonts/chinese_pixel_font.TTF", items[index], 40, x_coordinate, y_coordinate)
                else:
                    pass
            if x_coordinate < 360:
                x_coordinate = 0
            else:
                x_coordinate = 360
            y_coordinate += 50
            if y_coordinate == 750:
                x_coordinate = 360
                y_coordinate = 500

    # 展示所有的元素
    def display_menu(self):
        self.game_level_surface.fill(self.word_maker.BGC)  # 游戏的背景颜色
        # 下面的代码在展示feedback的时候不运行
        self.show_Finished_Task()  # 展示玩家已经完成的任务，放在最底层
        self.gameplay_time = pygame.time.get_ticks()  # 记录游戏运行的时间
        self.draw_Blocks(self.tasks_parameters_list[self.task_index][6],
                         self.tasks_parameters_list[self.task_index][4])  # 将答题框画到游戏界面上
        self.draw_progress_bar(self.tasks_parameters_list[self.task_index][8])
        if self.task_change:  # 如果时间改变，代表单词改变，所以要重新读取单词
            # 切换任务以前记录玩家的单词以及对应的轮数
            self.save_player_feature = []  # 将玩家的特征归0
            self.save_player_feature.append(self.tasks_parameters_list[self.task_index][0])  # 添加单词
            self.save_player_feature.append(self.tasks_parameters_list[self.task_index][4])  # 添加单词长度
            self.save_player_feature.append(self.tasks_parameters_list[self.task_index][5])  # 添加当前难度
            if self.tasks_parameters_list[self.task_index][0] in self.word_maker.word_loop:  # 如果已经有这个单词
                self.word_maker.word_loop[self.tasks_parameters_list[self.task_index][0]] += str(self.tasks_parameters_list[self.task_index][5])
            else:
                self.word_maker.word_loop[self.tasks_parameters_list[self.task_index][0]] = str(self.tasks_parameters_list[self.task_index][5])

            self.split_Word(self.tasks_parameters_list[self.task_index][0])  # 拆分单词
            self.task_change = False  # 关闭切换任务开关
            self.current_attempt = 0  # 将尝试的次数修改为0
            self.press_Q = 0  # 玩家主动听发音的次数归0
            self.press_Enter = 0  # 玩家被动听发音的次数归0
            self.yellow_color = 0  # 统计总共的黄色的次数
            self.green_color = 0  # 统计总共的绿色的个数
            self.red_color = 0  # 统计总共的红色的个数
            self.red_color_dic = {}  # 统计哪个字母迷惑了玩家
            self.player_used_spelling = []  # 将玩家过去的拼写变为空列表
            self.time_pause = False  # 让进度条时间可以继续减少
            self.time_change = True  # 要重新读取游戏时间

        self.check_Spelling()  # 检查玩家的拼写
        self.game_level_surface.blit(self.Blocks_Surface, (0, 200))  # 将表格画到主屏幕上，一定要在画字母之前
        self.draw_Letters()  # 往频幕上画字母
        if self.current_attempt != self.tasks_parameters_list[self.task_index][6]:  # 如果最后一次机会已经用完移动字母就不在起作用
            self.letter_move()  # 移动字母
        # 展示字母及混淆字母
        self.draw_word_parameters()  # 展示音标和任务
        self.word_maker.window.blit(self.game_level_surface, (0, 0))  # 将游戏界面的内容画到游戏主题上
