import numpy as np
from SimpleGraphics import (
    setWindowTitle,
    resize,
    background,
    line,
    rect,
    setFill,
    setFont,
    text
)

background_color = 'white smoke'
revealed_color = 'ivory4'
text_color = 'white smoke'
bomb_color = 'brown3'
ga_color = 'blue4'
error_color = 'chartreuse4'

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

    setWindowTitle(win_name)
    resize(board_size[0]*SQUARE_SIZE, board_size[1]*SQUARE_SIZE + 2 * SQUARE_SIZE)
    background(background_color)

    for i in range(board_size[0]):
        line(i*square_size, 0, i*square_size, board_size[1]*square_size)
    for j in range(board_size[1]):
        line(0, j*square_size, board_size[0]*square_size, j*square_size)
    
    setFill("red")
    rect(0, board_size[1]*square_size, board_size[0]*square_size, 2 * square_size)

    setFont("Times", f"{FONT_SIZE}")
    setFill("white")
    text(board_size[0]*square_size/2, board_size[1]*square_size + square_size/2, "Help me!")


def update_minesweeper(visible_tiles, ga_solution=None, error_tiles=None):
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

        rect(i*SQUARE_SIZE, j*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        
        if visible_tiles[i, j] == -1:
            if ga_solution[i, j] == 1:
                setFill(background_color)
                rect(i*SQUARE_SIZE, j*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)

                setFill(ga_color)
                setFont("Times", f"{FONT_SIZE}")
                text(i*SQUARE_SIZE+SQUARE_SIZE/2, j*SQUARE_SIZE+SQUARE_SIZE/2, 'P')
            else:
                setFill(background_color)
                rect(i*SQUARE_SIZE, j*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        elif visible_tiles[i, j] == 0:
            case = MINESWEEPER[i, j]

            if case == -1:
                setFill(bomb_color)
                rect(i*SQUARE_SIZE, j*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            elif case == 0:
                setFill(error_color if error_tiles[i, j] == 1 else revealed_color)
                rect(i*SQUARE_SIZE, j*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            elif case > 0:
                setFill(error_color if error_tiles[i, j] == 1 else revealed_color)
                rect(i*SQUARE_SIZE, j*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                setFill(text_color)
                setFont("Times", f"{FONT_SIZE}")
                text(i*SQUARE_SIZE+SQUARE_SIZE/2, j*SQUARE_SIZE+SQUARE_SIZE/2, str(case))
                
        elif visible_tiles[i, j] == 1:
            setFill(background_color)
            rect(i*SQUARE_SIZE, j*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            setFill(bomb_color)
            setFont("Times", f"{FONT_SIZE}")
            text(i*SQUARE_SIZE+SQUARE_SIZE/2, j*SQUARE_SIZE+SQUARE_SIZE/2, 'F')

    PREV_VISIBLE_TILES = visible_tiles.copy()
    PREV_GA_SOLUTION = ga_solution.copy()
    PREV_ERROR_TILES = error_tiles.copy()

initialize_minesweeper(np.zeros((20, 20)))