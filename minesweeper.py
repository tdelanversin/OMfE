import numpy as np
import matplotlib.pyplot as plt

board_size = 50, 50
n_mines = 750
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

plt.matshow(minesweeper)
plt.colorbar()
plt.show()



# Find initial start squares
if (minesweeper != 0).all():
    # This should never (but can) happen
    raise ValueError("Minesweeper contains no zero square, restart")
potential_starts = np.where(minesweeper == 0)
chosen_option = np.random.choice(potential_starts[0].size, size=int(potential_starts[0].size / 2))
start = (potential_starts[0][chosen_option],
         potential_starts[1][chosen_option])
tmp = minesweeper.copy()
tmp[start] = 9

# plt.matshow(tmp)
# plt.colorbar()
# plt.show()



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

plt.matshow(visible_tiles)
plt.colorbar()
plt.show()



# Solution checker
def satisfies_sweeper(solution):
    errors = 0
    for i, j in np.ndindex(visible_tiles.shape):
        if visible_tiles[i, j] == 0:
            neighbors = solution[max(0, i-1):min(board_size[0], i+2), max(0, j-1):min(board_size[1], j+2)]
            neighbors[neighbors == -2] = 0
            number_of_bombs = np.sum(neighbors)
            if number_of_bombs != minesweeper[i, j]:
                errors += 1
    
    return errors

