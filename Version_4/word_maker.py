# import all from Engine
from Engine import *


# initialize the width and height of display
Word_Maker = MainGame(1000, 800)


while Word_Maker.Main_Game_Running:   # the switch of Main Menu
    Word_Maker.game_Loop()  # constantly running code
    Word_Maker.current_menu.display_menu()  # display menus
    pygame.display.update()  # refresh surface
    Word_Maker.reset_Keys()  # reset keys
