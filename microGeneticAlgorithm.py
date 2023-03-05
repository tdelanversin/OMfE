import numpy as np
import sys

def micro_ga_solver(solution_space, fitness_function,
                    initial_solution=None, n_generations=10,
                    n_sub_generations=100, n_individuals=15,
                    n_crossovers=5):

    def fitnesses(solutions):
        return np.array([fitness_function(solution) for solution in solutions])


    solutions = np.random.randint(0, 2, size=(n_individuals, solution_space))
    if initial_solution is not None:
        solutions[0] = initial_solution
    solution_fitness = fitnesses(solutions)

    iteration_error = []
    generation = 0
    sub_generation = 0
    convergence = 0
    adaptive_crossovers = n_crossovers
    for generation in range(n_generations):
        solution_fitness = fitnesses(solutions)
        sorted_solutions = solutions[np.argsort(solution_fitness)]
        # adaptive_crossovers = int(n_crossovers * (n_generations - generation) / n_generations * 3) + 1
        adaptive_crossovers = min(adaptive_crossovers, int(np.min(solution_fitness) * solution_space / 5))

        iteration_error.append(solution_fitness)
        if (solution_fitness == 0).any():
            break

        solutions[0] = sorted_solutions[0]
        solutions[1:] = randomize_population(sorted_solutions[1:])

        for sub_generation in range(n_sub_generations):
            solution_fitness = fitnesses(solutions)
            sorted_solutions = solutions[np.argsort(solution_fitness)]

            iteration_error.append(solution_fitness)
            if (sorted_solutions[0] == sorted_solutions[1:]).all() or (solution_fitness == 0).any():
                convergence += 1
                break

            progress(generation + 1, n_generations,
                     progress_message(generation, sub_generation, adaptive_crossovers, convergence, min(solution_fitness)))

            solutions[0] = sorted_solutions[0]
            solutions[1:] = children_generation(
                sorted_solutions, solution_space, n_individuals, adaptive_crossovers)


    status = "Finished" if generation + 1 < n_generations else "Solution not found"
    progress(generation + 1, n_generations,
             progress_message(generation, sub_generation, adaptive_crossovers, convergence, min(solution_fitness), status))

    return solutions[np.argmin(solution_fitness)], np.array(iteration_error)


def randomize_population(population):
    new_population = np.zeros_like(population)

    new_population = np.random.randint(0, 2, size=population.shape)
    return new_population


def children_generation(population, solution_space, n_individuals, n_crossovers):
    # Pick crossover points
    crossovers = np.array([0, solution_space] + [0] * n_crossovers)
    crossovers[2:] = np.random.choice(solution_space, size=n_crossovers, replace=False)
    crossovers = np.sort(crossovers)
    crossover_patches = np.array([[i, j] for i, j in zip(crossovers[:-1], crossovers[1:])])

    # Crossover
    children = np.zeros_like(population[1:])

    for i in range(n_individuals - 1):
        parents = population[np.random.choice(n_individuals, size=2, replace=False)]

        children[i] = parents[0].copy()
        for s, e in crossover_patches[np.random.rand(len(crossover_patches)) < 0.5]:
            children[i][s:e] = parents[1][s:e]

    return children


def progress_message(generation, sub_generation, n_crossovers, convergence, best_fit, status=None):
    template = "\tmGA Gen: {}-{}\tError: {}  \tCross: {}, Conv: {}  \t{}\n" if status is not None else "\tmGA Gen: {}-{}\tError: {}  \tCross: {}, Conv: {}  "
    msg = template.format(
        generation,
        sub_generation,
        best_fit,
        n_crossovers,
        convergence,
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
