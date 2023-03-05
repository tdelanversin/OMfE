import numpy as np
import sys


def ga_solver(solution_space, fitness_function,
              initial_solution=None, n_generations=100,
              n_individuals=100, n_survivors=30,
              n_mutations=5, n_crossovers=10):
    
    def fitnesses(solutions):
        return np.array([fitness_function(solution) for solution in solutions])


    solutions = np.random.randint(0, 2, size=(n_individuals, solution_space))
    if initial_solution is not None:
        solutions[0] = initial_solution
    solution_fitness = fitnesses(solutions)

    iteration_error = []
    generation = 0
    adaptive_crossovers = n_crossovers
    adaptive_mutations = n_mutations
    for generation in range(n_generations):
        solution_fitness = fitnesses(solutions)
        sorted_solutions = solutions[np.argsort(solution_fitness)]
        adaptive_crossovers = int(n_crossovers * (n_generations - generation) / n_generations) + 1
        adaptive_mutations = int(n_mutations * (n_generations - generation) / n_generations) + 1

        iteration_error.append(solution_fitness)
        if (solution_fitness == 0).any():
            break

        progress(generation + 1, n_generations, progress_message(generation, min(solution_fitness), adaptive_crossovers, adaptive_mutations))

        survivors = survivor_selection(sorted_solutions, solution_space, n_individuals, n_survivors)
        children = children_generation(survivors, solution_space, n_individuals, adaptive_mutations, adaptive_crossovers)

        solutions = children


    status = "Finished" if generation + 1 < n_generations else "Solution not found"
    progress(generation + 1, n_generations, progress_message(generation, min(solution_fitness), adaptive_crossovers, adaptive_mutations, status))

    return solutions[np.argmin(solution_fitness)], np.array(iteration_error)


def survivor_selection(solutions, solution_space, n_individuals, n_survivors):
    survivors = np.zeros((n_survivors, solution_space))

    # Always keep the best individual
    survivors[0] = solutions[0]

    rank_probabilities = np.arange(n_individuals - 1, 0, -1) ** 3
    probability = rank_probabilities / np.sum(rank_probabilities)

    chosen_survivors = np.random.choice(
        n_individuals - 1, p=probability, size=n_survivors-1, replace=False)
    survivors[1:] = solutions[chosen_survivors + 1]

    return survivors


def children_generation(survivors, solution_space, n_individuals, n_mutations, n_crossovers):
    crossovers = np.array([0, solution_space] + [0] * n_crossovers)
    crossovers[2:] = np.random.choice(solution_space, size=n_crossovers, replace=False)
    crossovers = np.sort(crossovers)
    crossover_patches = np.array([[i, j] for i, j in zip(crossovers[:-1], crossovers[1:])])

    n_survivors = len(survivors)

    children = np.zeros((n_individuals, solution_space))
    children[:n_survivors] = survivors

    for i in range(n_individuals - n_survivors):
        parents = children[np.random.choice(n_survivors + i, size=2, replace=False)]
        children[n_survivors + i] = make_child(parents[0], parents[1], solution_space, n_mutations, crossover_patches)

    return children


def make_child(parent1, parent2, solution_space, n_mutations, crossover_patches):
    child = parent1.copy()
    for s, e in crossover_patches[np.random.rand(len(crossover_patches)) < 0.5]:
        child[s:e] = parent2[s:e]

    mutations = np.unique(np.random.choice(solution_space, size=n_mutations))
    child[mutations] = (child[mutations] + 1) % 2

    return child


def progress_message(generation, best_fit, n_crossovers, n_mutations, status=None):
    template = "\tGA Gen: {}\tError: {}  \tCross: {}, Mut: {}  \t{}\n" if status is not None else "\tGA Gen: {}\tError: {}  \tCross: {}, Mut: {}  "
    msg = template.format(
        generation,
        best_fit,
        n_crossovers,
        n_mutations,
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
