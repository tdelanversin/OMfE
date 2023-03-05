from time import sleep
import numpy as np
from functools import partial
from GeneticAlgorithm import ga_solver
from SimpleGraphics import close, mousePos, rightButtonPressed, leftButtonPressed
from microGeneticAlgorithm import micro_ga_solver
from display_minesweeper_simplegraphics import initialize_minesweeper, update_minesweeper

parameters = {
    "n_generations": 300,
    "n_individuals": 100,
    "n_survivors": 30,
    "n_crossovers": 10,
    "n_mutations": 5,
}

square_size = 50
font_size = 10
board_size = 30, 30
n_mines = 200
neighbors_tuple = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]



# Generate a random field of mines
mines = np.zeros(board_size)
positions = np.random.choice(
    board_size[0] * board_size[1], size=n_mines, replace=False)
positions = np.unravel_index(positions, board_size)
mines[positions] = 1



# Generate mine neighbors
padded_mines = np.zeros((board_size[0]+2, board_size[1]+2))
padded_mines[1:-1, 1:-1] = mines

neighbors = []
for i, j in neighbors_tuple:
    i, j = i + 1, j + 1
    shifted_mines = padded_mines[i:board_size[0]+i, j:board_size[1]+j]
    neighbors.append(shifted_mines)
neighbors = np.stack(neighbors, axis=-1)

minesweeper = np.sum(neighbors, axis=-1, dtype=int)
minesweeper[mines == 1] = -1



# Find initial start squares
if (minesweeper != 0).all():
    # This should never (but can) happen
    raise ValueError("Minesweeper contains no zero square, restart")
potential_starts = np.where(minesweeper == 0)
chosen_option = np.random.choice(potential_starts[0].size, size=1)
start = (potential_starts[0][chosen_option],
         potential_starts[1][chosen_option])
tmp = minesweeper.copy()
tmp[start] = 9



# Generate visible field
# -2 = unexplored, -1 = explored, 0 = empty, 1 = mine
visible_tiles = np.full_like(minesweeper, -2)

def uncover_square(x, y):
    visible_tiles[x, y] = 0
    if minesweeper[x, y] == -1:
        visible_tiles[x, y] = 1
    elif minesweeper[x, y] == 0:
        for i, j in neighbors_tuple:
            if 0 <= x + i < board_size[0] and 0 <= y + j < board_size[1] and visible_tiles[x+i, y+j] < 0:
                uncover_square(x+i, y+j)
    else:
        for i, j in neighbors_tuple:
            if 0 <= x + i < board_size[0] and 0 <= y + j < board_size[1] and visible_tiles[x+i, y+j] == -2:
                visible_tiles[x+i, y+j] = -1

[uncover_square(x, y) for x, y in zip(*start)]



# Solution checker
def satisfies_sweeper(solution, visible_tiles=visible_tiles):
    errors = 0
    for i, j in np.ndindex(visible_tiles.shape):
        if visible_tiles[i, j] == 0:
            neighbors = solution[max(0, i-1):min(board_size[0], i+2), max(0, j-1):min(board_size[1], j+2)]
            neighbors[neighbors == -2] = 0
            number_of_bombs = np.sum(neighbors)
            if number_of_bombs != minesweeper[i, j]:
                errors += np.abs(number_of_bombs - minesweeper[i, j])
    return errors



# Fitness function
def fitness(solution, visible_tiles=visible_tiles):
    solution_full = visible_tiles.copy()
    solution_full[visible_tiles == -1] = solution
    solution = solution_full
    return satisfies_sweeper(solution, visible_tiles)



# Solve obvious squares
def solve_obvious(game_tiles):
    flagged_tiles = game_tiles.copy()
    for x, y in np.ndindex(flagged_tiles.shape):
        if flagged_tiles[x, y] == 0:
            number_of_bombs = minesweeper[x, y]

            neighbors = flagged_tiles[max(0, x-1):min(board_size[0], x+2), max(0, y-1):min(board_size[1], y+2)]
            number_of_covered_tiles = np.sum(neighbors != 0).astype(int)

            if number_of_bombs == number_of_covered_tiles:
                for i, j in neighbors_tuple:
                    if 0 <= x + i < board_size[0] and 0 <= y + j < board_size[1] and flagged_tiles[x+i, y+j] < 0:
                        flagged_tiles[x+i, y+j] = 1
    return flagged_tiles



# Find wrongly predicted tiles
def solve_error_tiles(game_tiles, solution):
    error_tiles = np.zeros_like(solution)
    for i, j in np.ndindex(solution.shape):
        if game_tiles[i, j] == 0:
            neighbors = solution[max(0, i-1):min(board_size[0], i+2), max(0, j-1):min(board_size[1], j+2)]
            number_of_bombs = np.sum((neighbors == 1).astype(int))
            if number_of_bombs != minesweeper[i, j]:
                error_tiles[i, j] = 1
    return error_tiles


game_tiles = visible_tiles.copy()
ga_solution_full = np.zeros_like(game_tiles)
error_tiles = np.zeros_like(game_tiles)

initialize_minesweeper(minesweeper, square_size=square_size, font_size=font_size, win_name=f'Minesweeper\t{n_mines} mines')
try:
    update_minesweeper(game_tiles)

    def handle_click(case, x, y):
        game_tiles[x, y] = 0
        if case == 0:
            for i, j in neighbors_tuple:
                if 0 <= x + i < board_size[0] and 0 <= y + j < board_size[1] and game_tiles[x+i, y+j] < 0:
                    handle_click(minesweeper[x+i, y+j], x+i, y+j)
        elif case > 0:
            for i, j in neighbors_tuple:
                if 0 <= x + i < board_size[0] and 0 <= y + j < board_size[1] and game_tiles[x+i, y+j] == -2:
                    game_tiles[x+i, y+j] = -1

    while(True):
        if rightButtonPressed():
            x, y = mousePos
            x = int(x/square_size)
            y = int(y/square_size)

            if y >= board_size[1]:
                continue

            if game_tiles[x, y] == -1:
                game_tiles[x, y] = 1
            elif game_tiles[x, y] == 1:
                game_tiles[x, y] = -1
            update_minesweeper(game_tiles, ga_solution=ga_solution_full, error_tiles=error_tiles)
        
        elif leftButtonPressed():
            x, y = mousePos
            x = int(x/square_size)
            y = int(y/square_size)

            if y >= board_size[1]:
                game_tiles = solve_obvious(game_tiles)
                update_minesweeper(game_tiles)

                solution_tiles = game_tiles == -1
                solution_space = np.sum(solution_tiles.astype(int))

                ga_solution, error = ga_solver(solution_space, partial(fitness, visible_tiles=game_tiles), **parameters)
                ga_solution_full = game_tiles.copy()
                ga_solution_full[solution_tiles] = ga_solution

                error_tiles = solve_error_tiles(game_tiles, ga_solution_full)

                update_minesweeper(game_tiles, ga_solution=ga_solution_full, error_tiles=error_tiles)
                continue

            if game_tiles[x, y] == 0:
                continue

            case = minesweeper[x, y]
            handle_click(case, x, y)

            update_minesweeper(game_tiles, ga_solution=ga_solution_full, error_tiles=error_tiles)

            if case == -1:
                while not leftButtonPressed:
                    sleep(0.1)
                break

except Exception as e:
    print(e)
finally:
    bombs_found = np.sum(((game_tiles == 1) * (minesweeper == -1)).astype(int))
    bombs_missed = np.sum(((game_tiles < 1) * (minesweeper == -1)).astype(int))
    wrong_flags = np.sum(((game_tiles == 1) * (minesweeper != -1)).astype(int))

    print("Bombs found:", bombs_found)
    print("Bombs missed:", bombs_missed)
    print("Wrong flags:", wrong_flags)

    close()