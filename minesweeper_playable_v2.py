import time
import numpy as np
import pandas as pd
from functools import partial
from GeneticAlgorithm import ga_solver
from microGeneticAlgorithm import micro_ga_solver
from EvolutionStrategy import es_solver
from AntColonyOptimization import aco_solver
from display_minesweeper import initialize_minesweeper, update_minesweeper
from graphics import GraphicsError
from argparse import ArgumentParser
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("-o", "--optimization", dest="optimization_algorithm", default='all', help="choose optimization algorithm")
parser.add_argument("-q", "--quiet", action="store_false", dest="verbose", default=True, help="don't print status messages to stdout")
parser.add_argument("-p", "--plots", action="store_true", dest="show_plots", default=False, help="show plots for error vs. generation")
parser.add_argument("-ss", "--square_size", dest="square_size", default=50, type=int, help="change square size")
parser.add_argument("-fs", "--font_size", dest="font_size", default=15, type=int, help="change font size")
parser.add_argument("-bs", "--board_size", dest="board_size", default=30, type=int, help="change board size")
parser.add_argument("-ms", "--mines", dest="mines", default=200, type=int, help="change number of mines")
parser.add_argument("-tm", "--time_multiplier", dest="time_multiplier", default=1, type=int, help="modify time allowed for optimization by multiplying the base time (~6 secs)")
parser.add_argument("-d", "--demo", action="store_true", dest="demo", default=False, help="run demo")
parser.add_argument("-r", "--repeat", dest="repeat", default=1, type=int, help="repeat optimization algorithm")

args = parser.parse_args()
if args.repeat > 1:
    args.show_plots = False
    args.verbose = False

square_size = args.square_size
font_size = args.font_size
board_size = args.board_size, args.board_size
n_mines = args.mines
neighbors_tuple = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def log(*_args, **kwargs):
    if args.verbose:
        print(*_args, **kwargs)

def _plot_error(error, axes, optimizer, color, min_path = None):
    axes.set_title(f"{optimizer} optimization")
    axes.set_xlabel("Generation")
    axes.set_ylabel("Error")
    axes.axhline(y=0, linestyle='dashed', color='gray', label='zero error')
    axes.plot(error.min(axis=-1), c='black', label="Minimum error")
    for i, row in enumerate(error):
        axes.scatter([i]* len(row), row, c=color)
    if min_path is not None:
        axes.plot(min_path, c='cyan', label="Minimum path error")
    axes.legend()

GAME_OPTIMIZER_SCORES = []

def optimization_algorithm(solution_space, fitness_function):
    log("Solution space:", solution_space)

    ga_params = {
        "n_generations": 500 * args.time_multiplier,
        "n_individuals": 100,
        "n_survivors": 40,
        "n_crossovers": int(solution_space / 10) + 1,
        "n_mutations": int(solution_space / 10) + 1,
    }
    mga_params = {
        "n_generations": 50 * args.time_multiplier,
        "n_sub_generations": 100,
        "n_individuals": 15,
        "n_crossovers": int(solution_space / 3),
    }
    es_params = {
        "n_generations": 700 * args.time_multiplier,
        "n_individuals": 100,
        "n_survivors": 30,
        "n_mutations": solution_space,
    }
    aco_params = {
        "n_generations": 1000 * args.time_multiplier,
        "pheromone_decay": 0.1,
        "pheromone_trail": 1,
    }

    if args.optimization_algorithm == 'all':

        tik = time.time()
        ga_sol, ga_err = ga_solver(solution_space, fitness_function, **ga_params)
        ga_best_err = ga_err[-1].min()
        log("GA time:", round(time.time() - tik, 3), "GA error:", ga_err[-1].min())
        tik = time.time()
        mga_sol, mga_err = micro_ga_solver(solution_space, fitness_function, **mga_params)
        mga_best_err = mga_err[-1].min()
        log("mGA time:", round(time.time() - tik, 3), "mGA error:", mga_err[-1].min())
        tik = time.time()
        es_sol, es_err = es_solver(solution_space, fitness_function, **es_params)
        es_best_err = es_err[-1].min()
        log("ES time:", round(time.time() - tik, 3), "ES error:", es_err[-1].min())
        tik = time.time()
        aco_sol, aco_err, min_path = aco_solver(solution_space, fitness_function, **aco_params)
        aco_best_err = min_path[-1]
        log("ACO time:", round(time.time() - tik, 3), "ACO error:", aco_err[-1].min())

        if args.show_plots:
            _, axs = plt.subplots(4, 1, figsize=(25, 20))
            _plot_error(ga_err, axs[0], "GA", 'red')  # type: ignore
            _plot_error(mga_err, axs[1], "mGA", 'blue')  # type: ignore
            _plot_error(es_err, axs[2], "ES", 'green')  # type: ignore
            _plot_error(aco_err, axs[3], "ACO", 'orange', min_path)  # type: ignore
            plt.show()

        GAME_OPTIMIZER_SCORES.append((ga_best_err, mga_best_err, es_best_err, aco_best_err))

        if ga_best_err <= min(mga_best_err, es_best_err, aco_best_err):
            return ga_sol, ga_best_err
        elif mga_best_err <= min(ga_best_err, es_best_err, aco_best_err):
            return mga_sol, mga_best_err
        elif es_best_err <= min(ga_best_err, mga_best_err, aco_best_err):
            return es_sol, es_best_err
        else:
            return aco_sol, aco_best_err

    elif args.optimization_algorithm == 'ga':
        sol, err = ga_solver(solution_space, fitness_function, **ga_params)
        GAME_OPTIMIZER_SCORES.append(err[-1].min())
        if args.show_plots:
            _, ax = plt.subplots(1, 1, figsize=(25, 5))
            _plot_error(err, ax, "GA", 'red')  # type: ignore
            plt.show()
        return sol, err[-1].min()
    elif args.optimization_algorithm == 'mga':
        sol, err = micro_ga_solver(solution_space, fitness_function, **mga_params)
        GAME_OPTIMIZER_SCORES.append(err[-1].min())
        if args.show_plots:
            _, ax = plt.subplots(1, 1, figsize=(25, 5))
            _plot_error(err, ax, "mGA", 'blue')  # type: ignore
            plt.show()
        return sol, err[-1].min()
    elif args.optimization_algorithm == 'es':
        sol, err = es_solver(solution_space, fitness_function, **es_params)
        GAME_OPTIMIZER_SCORES.append(err[-1].min())
        if args.show_plots:
            _, ax = plt.subplots(1, 1, figsize=(25, 5))
            _plot_error(err, ax, "ES", 'green')  # type: ignore
            plt.show()
        return sol, err[-1].min()
    elif args.optimization_algorithm == 'aco':
        sol, err, min_path = aco_solver(solution_space, fitness_function, **aco_params)
        GAME_OPTIMIZER_SCORES.append(min_path[-1])
        if args.show_plots:
            _, ax = plt.subplots(1, 1, figsize=(25, 5))
            _plot_error(err, ax, "ACO", 'orange', min_path)  # type: ignore
            plt.show()
        return sol, min_path[-1]
    else:
        raise ValueError("Invalid optimization algorithm")


if args.demo:
    minesweeper = pd.read_csv("minesweeper_demo_board.csv", header=None).values
    start = ([14],[18])
    board_size = 30, 30
else:
    # Generate a random field of mines
    mines = np.zeros(board_size)
    positions = np.random.choice(board_size[0] * board_size[1], size=n_mines, replace=False)
    positions = np.unravel_index(positions, board_size)
    mines[positions] = 1


    # Generate mine neighbors
    padded_mines = np.zeros((board_size[0]+2, board_size[1]+2))
    padded_mines[1:-1, 1:-1] = mines

    neighbors = [
        padded_mines[0:board_size[0]+0, 0:board_size[1]+0],
        padded_mines[0:board_size[0]+0, 1:board_size[1]+1],
        padded_mines[0:board_size[0]+0, 2:board_size[1]+2],
        padded_mines[1:board_size[0]+1, 0:board_size[1]+0],
        padded_mines[1:board_size[0]+1, 2:board_size[1]+2],
        padded_mines[2:board_size[0]+2, 0:board_size[1]+0],
        padded_mines[2:board_size[0]+2, 1:board_size[1]+1],
        padded_mines[2:board_size[0]+2, 2:board_size[1]+2]
    ]
    neighbors = np.stack(neighbors, axis=-1)

    minesweeper = np.sum(neighbors, axis=-1, dtype=int)
    minesweeper[mines == 1] = -1


    # Find initial start squares
    if (minesweeper != 0).all():
        # This should never (but can) happen
        raise ValueError(
            "Generated minesweeper contains no zero square, restart. "
            "If this happens again, it is recommended to lower the number of mines.")
    potential_starts = np.where(minesweeper == 0)
    chosen_option = np.random.choice(potential_starts[0].size, size=1)
    start = (potential_starts[0][chosen_option],
            potential_starts[1][chosen_option])



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

[uncover_square(x, y) for x, y in zip(*start)]  # type: ignore



# Solution checker
def satisfies_sweeper(solution, game_tiles):
    solution_mines = (solution == 1).astype(int)

    padded_mines = np.zeros((board_size[0]+2, board_size[1]+2))
    padded_mines[1:-1, 1:-1] = solution_mines

    neighbors = [
        padded_mines[0:board_size[0]+0, 0:board_size[1]+0],
        padded_mines[0:board_size[0]+0, 1:board_size[1]+1],
        padded_mines[0:board_size[0]+0, 2:board_size[1]+2],
        padded_mines[1:board_size[0]+1, 0:board_size[1]+0],
        padded_mines[1:board_size[0]+1, 2:board_size[1]+2],
        padded_mines[2:board_size[0]+2, 0:board_size[1]+0],
        padded_mines[2:board_size[0]+2, 1:board_size[1]+1],
        padded_mines[2:board_size[0]+2, 2:board_size[1]+2]
    ]
    neighbors = np.stack(neighbors, axis=-1)

    solution_minesweeper = np.sum(neighbors, axis=-1, dtype=int)

    difference = np.sum(np.abs(minesweeper[game_tiles == 0] - solution_minesweeper[game_tiles == 0]))
    return difference



# Fitness function
def fitness(solution, game_tiles):
    solution_full = game_tiles.copy()
    solution_full[game_tiles == -1] = solution
    return satisfies_sweeper(solution_full, game_tiles)



# Find wrongly predicted tiles
def find_error_tiles(game_tiles, solution):
    error_tiles = np.zeros_like(solution)
    for i, j in np.ndindex(solution.shape):
        if game_tiles[i, j] == 0:
            neighbors = solution[max(0, i-1):min(board_size[0], i+2), max(0, j-1):min(board_size[1], j+2)]
            number_of_bombs = np.sum((neighbors == 1).astype(int))
            if number_of_bombs != minesweeper[i, j]:
                error_tiles[i, j] = 1
    return error_tiles



# Solve obvious squares
def solve_obvious():
    prev_game_tiles = np.zeros_like(game_tiles)
    while(np.any(prev_game_tiles != game_tiles)):

        prev_game_tiles = game_tiles.copy()
        for x, y in np.ndindex(game_tiles.shape):  # type: ignore

            if game_tiles[x, y] != 0:
                continue

            number_of_bombs = minesweeper[x, y]

            neighbors = game_tiles[max(0, x-1):min(board_size[0], x+2), max(0, y-1):min(board_size[1], y+2)]
            number_of_covered_tiles = np.sum(neighbors != 0).astype(int)
            number_of_flags = np.sum(neighbors == 1).astype(int)

            if number_of_bombs == number_of_covered_tiles:
                for i, j in neighbors_tuple:
                    if 0 <= x + i < board_size[0] and 0 <= y + j < board_size[1] and game_tiles[x+i, y+j] < 0:
                        game_tiles[x+i, y+j] = 1

            if number_of_bombs == number_of_flags:
                for i, j in neighbors_tuple:
                    if 0 <= x + i < board_size[0] and 0 <= y + j < board_size[1] and game_tiles[x+i, y+j] < 0:
                        case = minesweeper[x+i, y+j]
                        handle_click(case, x+i, y+j)
                        if case == -1:
                            update_minesweeper(win, game_tiles)
                            win.getMouse()
                            raise Exception('Game over')


game_tiles = visible_tiles.copy()
ga_solution_full = np.zeros_like(game_tiles)
error_tiles = np.zeros_like(game_tiles)

win = initialize_minesweeper(minesweeper, square_size=square_size, font_size=font_size, win_name=f'Minesweeper\t{n_mines} mines')
try:
    update_minesweeper(win, game_tiles)

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
        elif case == -1:
            game_tiles[minesweeper == -1] = 0



    while(win.isOpen()):
        point, is_right_click = win.getMouse()
        x, y = point.getX(), point.getY()
        x = int(x/square_size)
        y = int(y/square_size)

        if y >= board_size[1]:
            solve_obvious()
            update_minesweeper(win, game_tiles)

            solution_tiles = game_tiles == -1
            solution_space = np.sum(solution_tiles.astype(int))

            ga_solution = None
            err = None
            for _ in range(max(1, args.repeat)):
                new_solution, new_err = optimization_algorithm(solution_space, partial(fitness, game_tiles=game_tiles))
                if ga_solution is None or new_err < err:
                    ga_solution = new_solution
                    err = new_err
            if args.repeat > 1:
                print(f'Repeated {args.repeat} times, error: {err}')
                print(f"Average error for optimizer{'s' if args.optimization_algorithm == 'all' else ''} " +
                    f"'{args.optimization_algorithm}': " +
                    f"{np.array(GAME_OPTIMIZER_SCORES).mean(axis=0)}")

            ga_solution_full = game_tiles.copy()
            ga_solution_full[solution_tiles] = ga_solution # type: ignore

            error_tiles = find_error_tiles(game_tiles, ga_solution_full)

            update_minesweeper(win, game_tiles, ga_solution=ga_solution_full, error_tiles=error_tiles)
            continue

        if game_tiles[x, y] == 0:
            continue

        if is_right_click:
            if game_tiles[x, y] == -1:
                game_tiles[x, y] = 1
            elif game_tiles[x, y] == 1:
                game_tiles[x, y] = -1
            update_minesweeper(win, game_tiles, ga_solution=ga_solution_full, error_tiles=error_tiles)
            continue

        case = minesweeper[x, y]
        handle_click(case, x, y)

        update_minesweeper(win, game_tiles, ga_solution=ga_solution_full, error_tiles=error_tiles)

        if case == -1:
            win.getMouse()
            raise Exception('Game over')

except GraphicsError:
    pass
# except Exception as e:
#     log('\n' + str(e))
finally:
    win.close()