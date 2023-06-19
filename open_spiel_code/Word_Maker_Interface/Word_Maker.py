from word_maker_environment import WordMakerGame

# define the tasks pool
# _TASKS_POOL = {'人的 h j u m ʌ n': 'h u m a n', '谦逊的 h ʌ m b ʌ l': 'h u m b l e', '湿的 h j u m ʌ d': 'h u m i d',
#                '墨水 ɪ ŋ k': 'i n k', '铁 aɪ ɝ n': 'i r o n', '语言 l æ ŋ ɡ w ʌ dʒ': 'l a n g u a g e',
#                '洗衣房 l ɔ n d r i': 'l a u n d r y', '难题 p ʌ z ʌ l ': 'p u z z l e'}
_TASKS_POOL = {'谦逊的 h ʌ m b ʌ l': 'h u m b l e'}

# define the total game round
_TOTAL_GAME_ROUND = 4

game = WordMakerGame(_TASKS_POOL, _TOTAL_GAME_ROUND)
state = game.new_initial_state()
while not state.is_terminal:
    # print(state.current_player)
    # print(state.legal_action)
    print(f'当前玩家：{state.current_player}, 当前难度设定：{state.current_difficulty_setting}，当前的游戏轮数：{state.current_game_round}, 第几次尝试：{state._current_attempt}, '
          f'学生的拼写：{state.stu_spelling}， 学生反馈：{state.stu_feedback}, 老师反馈：{state.tutor_feedback}')
    # 当前难度设定，迷惑字母设定，学生反馈，老师反馈
    action = state.legal_action
    state.apply_action(action)
    # print(state.current_player)
    # print(state.legal_action)
    # state.apply_action('decide_difficulty_level')
    # print(state.current_difficulty_setting)
    # print(state.legal_action)
    # state.apply_action('student_spelling')

    # print(state.current_player)
    # print(state.legal_action)
    # state.apply_action('give_feedback')
    # print(state.stu_feedback)
    # print(state.current_player)
    # print(state.legal_action)
    # print('当前的轮数',state.current_game_round)
    # print('当前的尝试次数',state._current_attempt)
    # print(state.is_terminal)
