import numpy as np
import sys


def es_solver(solution_space, fitness_function,
              initial_solution=None, n_generations=100,
              n_individuals=100, n_survivors=30,
              n_mutations=300):
    

    def fitnesses(solutions):
        return np.array([fitness_function(solution) for solution in solutions])


    solutions = np.random.randint(0, 2, size=(n_individuals, solution_space))
    if initial_solution is not None:
        solutions[0] = initial_solution
    solution_fitness = fitnesses(solutions)

    iteration_error = []
    generation = 0
    adaptive_mutation = n_mutations
    for generation in range(n_generations):
        solution_fitness = fitnesses(solutions)
        sorted_solutions = solutions[np.argsort(solution_fitness)]
        adaptive_mutation = max(int(n_mutations * (n_generations - generation) / n_generations) * 10, 100)

        iteration_error.append(solution_fitness)
        if (solution_fitness == 0).any():
            break

        progress(generation + 1, n_generations, progress_message(generation, min(solution_fitness), adaptive_mutation))

        parents = parents_selection(solutions, solution_fitness, n_individuals - n_survivors)
        mutations = np.random.choice(parents.size, size=adaptive_mutation, replace=False)
        mutations = np.unravel_index(mutations, parents.shape)
        children = parents.copy()
        children[mutations] += 1
        children %= 2

        solutions[:n_survivors] = sorted_solutions[:n_survivors]
        solutions[n_survivors:] = children

    status = "Finished" if generation + 1 < n_generations else "Solution not found"
    progress(generation + 1, n_generations, progress_message(generation, min(solution_fitness), adaptive_mutation, status))

    return solutions[np.argmin(solution_fitness)], np.array(iteration_error)


def parents_selection(solutions, fitnesses, n_parents):
    reverse_fitness = np.max(fitnesses) - fitnesses
    probability = reverse_fitness / np.sum(reverse_fitness)

    chosen_survivors = np.random.choice(len(solutions), p=probability, size=n_parents, replace=False)
    survivors = solutions[chosen_survivors]

    return survivors

def progress_message(generation, best_fit, mutations, status=None):
    template = "\tES Gen: {}\tError: {}  \tMut: {}  \t{}\n" if status is not None else "\tES Gen: {}\tError: {}  \tMut: {}  "
    msg = template.format(
        generation,
        best_fit,
        mutations,
        status
    )
    return msg

def progress(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '█' * filled_len + ' ' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
