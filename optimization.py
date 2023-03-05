from graphics import (
    GraphWin,
    Circle,
    Point,
    Line,
    Text,
    Rectangle,
    color_rgb
)
import numpy as np

window_size = 1500, 1000
number_of_nodes = 30

coordinates = np.array([1,1])
while len(np.unique(coordinates, axis=0)) != len(coordinates):
    x_coordinates = np.random.randint(0, window_size[0], size=30)
    y_coordinates = np.random.randint(0, window_size[1], size=30)
    coordinates = np.stack((x_coordinates, y_coordinates), axis=1)

distance_graph = np.zeros((number_of_nodes, number_of_nodes))
for i in range(number_of_nodes):
    for j in range(number_of_nodes):
        distance_graph[i,j] = np.linalg.norm(coordinates[i] - coordinates[j])
    
n_agents = 50
pheromone_decay = .95
pheromone_trail = 5
p_to_d_ratio = .6
pheromone_graph = np.ones((number_of_nodes, number_of_nodes))
max_path = None

def next_node_max(visited):
    node = visited[-1]

    not_visited = np.isin(range(number_of_nodes), visited, invert=True)
    candidates = np.where(not_visited)[0]

    candidate_distance = distance_graph[node, not_visited]
    candidate_pheromone = pheromone_graph[node, not_visited]

    distance_probabilities = 1 / candidate_distance / np.sum(1 / candidate_distance)
    pheromone_probabilities = candidate_pheromone / np.sum(candidate_pheromone)

    probabilities = p_to_d_ratio * pheromone_probabilities + (1 - p_to_d_ratio) * distance_probabilities

    return candidates[np.argmax(probabilities)]

def get_max_path():
    start = np.unravel_index(np.argmax(pheromone_graph), pheromone_graph.shape)
    path = [*start]
    while len(path) < number_of_nodes:
        path.append(next_node_max(path))
    path.append(path[0])
    return np.array(path)

def next_node(visited):
    node = visited[-1]

    not_visited = np.isin(range(number_of_nodes), visited, invert=True)
    candidates = np.where(not_visited)[0]

    candidate_distance = distance_graph[node, not_visited]
    candidate_pheromone = pheromone_graph[node, not_visited]

    distance_probabilities = 1 / candidate_distance / np.sum(1 / candidate_distance)
    pheromone_probabilities = candidate_pheromone / np.sum(candidate_pheromone)

    probabilities = p_to_d_ratio * pheromone_probabilities + (1 - p_to_d_ratio) * distance_probabilities

    next_node = np.random.choice(candidates, p=probabilities, size=1)[0]
    return next_node

def agent_path(start):
    path = [start]
    while len(path) < number_of_nodes:
        path.append(next_node(path))
    return path

def update_win_with_path(win, path):
    r = Rectangle(Point(0,0), Point(window_size[0], window_size[1]))
    r.setFill('white')
    r.draw(win)
    for i in range(number_of_nodes):
        c = Circle(Point(*coordinates[path[i]]), 6)
        c.setFill('black')
        c.draw(win)
    for i in range(len(path)-1):
        l = Line(Point(*coordinates[path[i]]), Point(*coordinates[path[i+1]]))
        l.setFill('red')
        l.draw(win)
    t = Text(Point(window_size[0]/2, window_size[1]/2),
        str(np.sum(distance_graph[path[:-1], path[1:]]).astype("int")))
    t.setTextColor('blue')
    t.setSize(18)
    t.setStyle("bold")
    t.draw(win)

def interation(win):
    global pheromone_graph, max_path

    agents = np.random.randint(0, number_of_nodes, size=n_agents)
    paths = []

    # agents find paths
    for agent in agents:
        path = agent_path(agent)
        path.append(path[0])
        paths.append(path)

    # decay pheromones
    pheromone_graph *= pheromone_decay

    # update pheromones
    for path in paths:
        path_length = np.sum(distance_graph[path[:-1], path[1:]])

        new_max_path = get_max_path()
        if max_path is None or \
            (max_path != new_max_path).any() and \
            np.sum(distance_graph[new_max_path[:-1], new_max_path[1:]]) \
                < np.sum(distance_graph[max_path[:-1], max_path[1:]]):
            max_path = new_max_path
            update_win_with_path(win, max_path)

        pheromone_graph[path[:-1], path[1:]] += 1 / path_length * pheromone_trail

win = GraphWin("Optimizing", *window_size)
win.setBackground('white')
try:
    if max_path is not None:
        update_win_with_path(win, max_path)
    for i in range(1000):
        
        interation(win)
except:
    pass
finally:
    win.close()
