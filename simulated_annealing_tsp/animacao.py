import pygame
import numpy as np
import numpy.typing as npt
import scipy
import time
pygame.init()

# constants
FPS: int = 60
number_of_cities: np.int_ = 10
temperature: np.float_ = 10.0
delta_t: np.float_ = 0.8
temperature_inferior_limit: np.float_ = 0.0005

# colors
WHITE: tuple = 255, 255, 255
BLACK: tuple = 0, 0, 0

def generate_random_graph(number_of_nodes: np.int_) -> npt.NDArray[np.float_]:
    nodes: npt.NDArray[np.float_] = np.random.rand(number_of_nodes, 2)
    adjacency_matrix: npt.NDArray[np.float_] = scipy.spatial.distance_matrix(nodes, nodes)
    return adjacency_matrix, nodes

SCREEN = pygame.display.set_mode((500, 500))
pygame.display.set_caption("Traveling Salesman Problem - Simulated Annealing")

clock = pygame.time.Clock()
clock.tick(FPS)

number_of_monte_carlo_steps: np.int_ = 1000 # hard coded
initial_temperature: np.float_ = temperature
graph, nodes = generate_random_graph(number_of_nodes=number_of_cities)
current_path: npt.NDArray[np.int_] = np.arange(number_of_cities)
np.random.shuffle(current_path)
edges: npt.NDArray[np.int_] = np.append(np.lib.stride_tricks.sliding_window_view(current_path, 2), [[current_path[-1], current_path[0]]], axis=0)
current_path_cost: np.float_ = np.sum([graph[i, j] for i, j in edges])

nodes = nodes * 500
while temperature > temperature_inferior_limit:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    for _ in range(number_of_monte_carlo_steps):
        edges: npt.NDArray[np.int_] = np.append(np.lib.stride_tricks.sliding_window_view(current_path, 2), [[current_path[-1], current_path[0]]], axis=0)
        SCREEN.fill(BLACK)
        for node in nodes:
            pygame.draw.circle(surface=SCREEN, center=node, color=WHITE, radius=3)
        for edge in edges:
            pygame.draw.line(SCREEN, WHITE, nodes[edge[0]], nodes[edge[1]], 1)
        time.sleep(0.1)
        proposed_x, proposed_y = np.random.choice(np.arange(number_of_cities), size=2, replace=False)
        proposed_path: npt.NDArray[np.int_] = current_path.copy()
        proposed_path[[proposed_x, proposed_y]] = proposed_path[[proposed_y, proposed_x]]
        decrease: np.float_ = graph[current_path[proposed_x - 1], current_path[proposed_x]] + graph[current_path[proposed_x], current_path[(proposed_x + 1) % number_of_cities]] + graph[current_path[proposed_y - 1], current_path[proposed_y]] + graph[current_path[proposed_y], current_path[(proposed_y + 1) % number_of_cities]]
        increase: np.float_ = graph[proposed_path[proposed_x - 1], proposed_path[proposed_x]] + graph[proposed_path[proposed_x], proposed_path[(proposed_x + 1) % number_of_cities]] + graph[proposed_path[proposed_y - 1], proposed_path[proposed_y]] + graph[proposed_path[proposed_y], proposed_path[(proposed_y + 1) % number_of_cities]]
        proposed_path_cost: np.float_ = current_path_cost - decrease + increase
        delta: np.float_ = proposed_path_cost - current_path_cost
        r: np.float_ = np.random.rand()
        P: np.float_ = np.exp(-1 * delta / temperature)
        if delta < 0 or r <= P:
            current_path = proposed_path
            current_path_cost = proposed_path_cost
        
        pygame.display.update()
    temperature *= delta_t
    time.sleep(5)

pygame.quit()