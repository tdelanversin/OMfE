import numpy as np
import sys

PHEROMONE_DISTANCE_RATIO = 0.6
AGENT_COUNT = 50
PHEROMONE_DECAY = 0.1
PHEROMONE_TRAIL = 30


def aco_solver(solution_space, fitness_function,
               initial_solution=None, n_generations=10,
               pheromone_trail=PHEROMONE_TRAIL, pheromone_decay=PHEROMONE_DECAY):
    pheromone_graph = np.ones((solution_space, 2))
    if initial_solution is not None:
        pheromone_graph[initial_solution == 1, 0] = 2
        pheromone_graph[initial_solution == 0, 1] = 2

    iteration_error = []
    generation = 0
    error = -1
    min_paths = []
    for generation in range(n_generations):
        pheromone_graph, err = binlist_iteration(solution_space, pheromone_graph, fitness_function, pheromone_decay, pheromone_trail)

        error = fitness_function(min_path(pheromone_graph))
        if error == 0:
            break
        min_paths.append(error)
        iteration_error.append(err)
        progress(generation + 1, n_generations, progress_message(generation, error))

    status = "Finished" if generation + 1 < n_generations else "Solution not found"
    progress(generation + 1, n_generations, progress_message(generation, error, status))

    return min_path(pheromone_graph), np.array(iteration_error), min_paths

def binlist_iteration(solution_space, pheromone_graph, fitness_function, pheromone_decay, pheromone_trail):
    nodes_taken_list = []
    fitness_values = []
    probabilities = pheromone_to_probability(pheromone_graph)
    for _ in range(AGENT_COUNT):
        nodes_taken = (np.random.rand(solution_space) <= probabilities).astype(int)
        fitness_value = fitness_function(nodes_taken)

        if fitness_value == 0:
            new_pheromones = [[0, 1] if i == 0 else [1, 0] for i in nodes_taken]
            return np.array(new_pheromones, np.float32), fitness_values + [0]

        nodes_taken_list.append(nodes_taken)
        fitness_values.append(fitness_value)

    pheromone_graph *= (1 - pheromone_decay)
    for nodes_taken, fitness_value in zip(nodes_taken_list, fitness_values):
        pheromone_graph[nodes_taken == 1, 0] += pheromone_trail / fitness_value
        pheromone_graph[nodes_taken == 0, 1] += pheromone_trail / fitness_value

    return pheromone_graph, np.array(fitness_values)

def pheromone_to_probability(pheromone_graph):
    return pheromone_graph[:, 0] / pheromone_graph.sum(axis=1)

def min_path(pheromone_graph):
    return (pheromone_to_probability(pheromone_graph) > 0.5).astype(int)

def progress_message(generation, best_fit, status=None):
    template = "\tACO Gen: {}\tError: {}  \t{}\n" if status is not None else "\tACO Gen: {}\tError: {}  "
    msg = template.format(
        generation,
        best_fit,
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















def set_parameters(
        pheromone_distance_ratio=PHEROMONE_DISTANCE_RATIO,
        agent_count=AGENT_COUNT,
        pheromone_decay=PHEROMONE_DECAY,
        pheromone_trail=PHEROMONE_TRAIL):
    global PHEROMONE_DISTANCE_RATIO, AGENT_COUNT, PHEROMONE_DECAY, PHEROMONE_TRAIL

    PHEROMONE_DISTANCE_RATIO = pheromone_distance_ratio
    AGENT_COUNT = agent_count
    PHEROMONE_DECAY = pheromone_decay
    PHEROMONE_TRAIL = pheromone_trail


def iteration(pheromone_graph, distance_graph):
    n_nodes, test_dim_1 = pheromone_graph.shape
    test_dim_2, test_dim_3 = distance_graph.shape
    if n_nodes != test_dim_1 or n_nodes != test_dim_2 or n_nodes != test_dim_3:
        raise ValueError(
            "pheromone_graph must be square, and equal in size to distance_graph")

    agents = np.random.randint(0, n_nodes, size=AGENT_COUNT)
    paths = []

    # agents find paths
    for agent in agents:
        path = _agent_path(pheromone_graph, distance_graph, agent, n_nodes)
        paths.append(path)

    # decay pheromones
    pheromone_graph *= PHEROMONE_DECAY

    # update pheromones
    for path in paths:
        path_length = np.sum(distance_graph[path[:-1], path[1:]])
        pheromone_graph[path[:-1], path[1:]] += 1 / \
            path_length * PHEROMONE_TRAIL


def get_max_path(pheromone_graph, distance_graph):
    n_nodes, test_dim = pheromone_graph.shape
    if n_nodes != test_dim:
        raise ValueError("pheromone_graph must be square")

    start = np.unravel_index(np.argmax(pheromone_graph), pheromone_graph.shape)
    path = [*start]
    while len(path) < n_nodes:
        path.append(
            _next_pheromone_max_node(
                pheromone_graph, distance_graph, path, n_nodes)
        )
    path.append(path[0])
    return np.array(path)


def _agent_path(pheromone_graph, distance_graph, start, n_nodes):
    path = [start]
    while len(path) < n_nodes:
        path.append(_next_node(pheromone_graph, distance_graph, path, n_nodes))
    path.append(start)
    return path


def _next_node(pheromone_graph, distance_graph, visited, n_nodes):
    candidates, probabilities = _path_probabilities(
        pheromone_graph, distance_graph, visited, n_nodes)

    next_node = np.random.choice(candidates, p=probabilities, size=1)[0]
    return next_node


def _next_pheromone_max_node(pheromone_graph, distance_graph, visited, n_nodes):
    candidates, probabilities = _path_probabilities(
        pheromone_graph, distance_graph, visited, n_nodes)

    return candidates[np.argmax(probabilities)]


def _path_probabilities(pheromone_graph, distance_graph, visited, n_nodes):
    node = visited[-1]

    not_visited = np.isin(range(n_nodes), visited, invert=True)
    candidates = np.where(not_visited)[0]

    candidate_distance = distance_graph[node, not_visited]
    candidate_pheromone = pheromone_graph[node, not_visited]

    inverse_distance = 1 / candidate_distance
    distance_probabilities = inverse_distance / np.sum(inverse_distance)

    pheromone_probabilities = candidate_pheromone / np.sum(candidate_pheromone)

    probabilities = PHEROMONE_DISTANCE_RATIO * pheromone_probabilities + \
        (1 - PHEROMONE_DISTANCE_RATIO) * distance_probabilities

    return candidates, probabilities
