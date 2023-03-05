from graphics import (
    GraphWin,
    Point,
    Line,
    Text,
    Rectangle,
    color_rgb,

    GraphicsError
)
import numpy as np
import matplotlib.pyplot as plt

def new_game():
    board_size = 20, 10
    field = (np.random.rand(*board_size) > .8).astype(int)

    big_field = np.zeros((field.shape[0]+2, field.shape[1]+2))
    big_field[1:-1, 1:-1] = field

    neighbors = []
    for i, j in [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]:
        neighbors.append(big_field[i:board_size[0]+i, j:board_size[1]+j])
    neighbors = np.stack(neighbors, axis=-1)

    minesweeper = np.sum(neighbors, axis=-1, dtype=int)
    minesweeper[field == 1] = -1
    plt.matshow(np.transpose(minesweeper))
    plt.colorbar()
    plt.show()

    revealed_color = color_rgb(50, 50, 50)
    text_color = color_rgb(200, 200, 200)
    bomb_color = color_rgb(255, 50, 50)

    square_size = 50
    revealed = np.zeros_like(minesweeper)
    first_zero = np.unravel_index(np.argmax(minesweeper == 0), board_size)
    win = GraphWin(f"Minesweeper, {np.sum(field)} bombs", board_size[0]*square_size, board_size[1]*square_size)

    def handle_click(case, x, y):
        if case == -1:
            square = Rectangle(Point(x*square_size, y*square_size), Point((x+1)*square_size, (y+1)*square_size))
            square.setFill(bomb_color)
            square.draw(win)
        elif case == 0:
            square = Rectangle(Point(x*square_size, y*square_size), Point((x+1)*square_size, (y+1)*square_size))
            square.setFill(revealed_color)
            square.draw(win)
            for i, j in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                if 0 <= x + i < board_size[0] and 0 <= y + j < board_size[1] and revealed[x+i, y+j] == 0:
                    revealed[x+i, y+j] = 1
                    handle_click(minesweeper[x+i, y+j], x+i, y+j)
        elif case > 0:
            square = Rectangle(Point(x*square_size, y*square_size), Point((x+1)*square_size, (y+1)*square_size))
            square.setFill(revealed_color)
            square.draw(win)
            text = Text(Point(x*square_size+square_size/2, y*square_size+square_size/2), str(case))
            text.setFill(text_color)
            text.draw(win)

    try:
        for i in range(board_size[0]):
            line = Line(Point(i*square_size, 0), Point(i*square_size, board_size[1]*square_size))
            line.draw(win)
        for j in range(board_size[1]):
            line = Line(Point(0, j*square_size), Point(board_size[0]*square_size, j*square_size))
            line.draw(win)

        revealed[first_zero] = 1
        handle_click(minesweeper[first_zero], first_zero[0], first_zero[1])

        while(win.isOpen()):
            point, is_right_click = win.getMouse()
            x, y = point.getX(), point.getY()
            x = int(x/square_size)
            y = int(y/square_size)

            if is_right_click:
                if revealed[x, y] == 0:
                    text = Text(Point(x*square_size+square_size/2, y*square_size+square_size/2), 'O')
                    text.setFill(bomb_color)
                    text.draw(win)
                continue
            if revealed[x, y] == 1:
                continue
            revealed[x, y] = 1

            case = minesweeper[x, y]
            handle_click(case, x, y)

    except GraphicsError:
        pass
    finally:
        win.close()