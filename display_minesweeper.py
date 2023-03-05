import numpy as np
from graphics import (
    GraphWin,
    Point,
    Line,
    Text,
    Rectangle,
    color_rgb,

    GraphicsError
)

background_color = color_rgb(220, 220, 220)
revealed_color = color_rgb(50, 50, 50)
text_color = color_rgb(200, 200, 200)
bomb_color = color_rgb(255, 50, 50)
ga_color = color_rgb(50, 50, 255)
error_color = color_rgb(81, 148, 15)

MINESWEEPER = None

SQUARE_SIZE = 50
FONT_SIZE = 18

PREV_VISIBLE_TILES = None
PREV_GA_SOLUTION = None
PREV_ERROR_TILES = None

def initialize_minesweeper(minesweeper, square_size=SQUARE_SIZE, font_size=FONT_SIZE, win_name='Minesweeper'):
    global MINESWEEPER, SQUARE_SIZE, FONT_SIZE, PREV_VISIBLE_TILES, PREV_GA_SOLUTION, PREV_ERROR_TILES
    MINESWEEPER = minesweeper
    SQUARE_SIZE = square_size
    FONT_SIZE = font_size

    PREV_VISIBLE_TILES = np.full_like(MINESWEEPER, -2)
    PREV_GA_SOLUTION = np.full_like(MINESWEEPER, -2)
    PREV_ERROR_TILES = np.zeros_like(MINESWEEPER)

    board_size = minesweeper.shape

    win = GraphWin(win_name, board_size[0]*square_size, board_size[1]*square_size + 2 * square_size)
    win.setBackground(background_color)

    for i in range(board_size[0]):
        line = Line(Point(i*square_size, 0), Point(i*square_size, board_size[1]*square_size))
        line.draw(win)
    for j in range(board_size[1]):
        line = Line(Point(0, j*square_size), Point(board_size[0]*square_size, j*square_size))
        line.draw(win)
    
    help_me_button = Rectangle(Point(0, board_size[1]*square_size), Point(board_size[0]*square_size, board_size[1]*square_size + 2 * square_size))
    help_me_button.setFill("red")
    help_me_button.draw(win)
    help_me_text = Text(Point(board_size[0]*square_size/2, board_size[1]*square_size + square_size/2), "Help me!")
    help_me_text.setSize(font_size)
    help_me_text.setFill("white")
    help_me_text.draw(win)

    return win


def update_minesweeper(win, visible_tiles, ga_solution=None, error_tiles=None):
    global PREV_VISIBLE_TILES, PREV_GA_SOLUTION, PREV_ERROR_TILES
    if MINESWEEPER.shape != visible_tiles.shape:
        raise ValueError("Minesweeper and visible mask are different sizes")

    if ga_solution is None:
        ga_solution = np.full_like(MINESWEEPER, -2)
    if error_tiles is None:
        error_tiles = np.zeros_like(MINESWEEPER)

    board_difference = (
        (visible_tiles != PREV_VISIBLE_TILES) + \
        (ga_solution != PREV_GA_SOLUTION) + \
        (error_tiles != PREV_ERROR_TILES)
    )

    if not np.any(board_difference):
        return

    for i, j in np.ndindex(MINESWEEPER.shape):
        if not board_difference[i, j]:
            continue

        square = Rectangle(Point(i*SQUARE_SIZE, j*SQUARE_SIZE), Point((i+1)*SQUARE_SIZE, (j+1)*SQUARE_SIZE))
        
        if visible_tiles[i, j] == -1:
            if ga_solution[i, j] == 1:
                square.setFill(background_color)
                square.draw(win)
                text = Text(Point(i*SQUARE_SIZE+SQUARE_SIZE/2, j*SQUARE_SIZE+SQUARE_SIZE/2), 'P')
                text.setFill(ga_color)
                text.setSize(FONT_SIZE)
                text.draw(win)
            else:
                square.setFill(background_color)
                square.draw(win)
        elif visible_tiles[i, j] == 0:
            case = MINESWEEPER[i, j]

            if case == -1:
                square.setFill(bomb_color)
                square.draw(win)
                if ga_solution[i, j] == 1:
                    text = Text(Point(i*SQUARE_SIZE+SQUARE_SIZE/2, j*SQUARE_SIZE+SQUARE_SIZE/2), 'P')
                    text.setFill(ga_color)
                    text.setSize(FONT_SIZE)
                    text.draw(win)
            elif case == 0:
                square.setFill(error_color if error_tiles[i, j] == 1 else revealed_color)
                square.draw(win)
            elif case > 0:
                square.setFill(error_color if error_tiles[i, j] == 1 else revealed_color)
                text = Text(Point(i*SQUARE_SIZE+SQUARE_SIZE/2, j*SQUARE_SIZE+SQUARE_SIZE/2), str(case))
                text.setFill(text_color)
                text.setSize(FONT_SIZE)
                square.draw(win)
                text.draw(win)
        elif visible_tiles[i, j] == 1:
            square.setFill(background_color)
            square.draw(win)
            text = Text(Point(i*SQUARE_SIZE+SQUARE_SIZE/2, j*SQUARE_SIZE+SQUARE_SIZE/2), 'F')
            text.setFill(bomb_color)
            text.setSize(FONT_SIZE)
            text.draw(win)

    PREV_VISIBLE_TILES = visible_tiles.copy()
    PREV_GA_SOLUTION = ga_solution.copy()
    PREV_ERROR_TILES = error_tiles.copy()

