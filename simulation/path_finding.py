"""
Path finding algorithms (Dijkstra) and path validation.
"""

import numpy as np
import heapq
from .utils import euclidean_length


def generate_rectangular_array(vertex_pair):
    '''Generate a rectangular grid of points in a given dimension.'''

    # Ensure p0 and p1 are NumPy arrays
    p0 = np.array(vertex_pair[0], dtype=int)
    p1 = np.array(vertex_pair[1], dtype=int)

    # Determine the number of dimensions
    dim = p0.shape[0]

    # Create the grid points for each dimension
    grid_points = []

    for i in range(dim):
        grid_points.append(np.arange(p0[i], p1[i] + 1))

    # Create the meshgrid for the given dimension
    grid = np.meshgrid(*grid_points, indexing='ij')

    # Stack the grids into a single array (this will be the final grid)
    grid_stacked = np.stack(grid, axis=-1)

    return grid_stacked


def is_path_valid(path, box, len_min):
    """
    Check if a path stays within a bounding box and exceeds a minimum length.
    box = (x0, x1, y0, y1)
    """
    x0, x1, y0, y1 = box
    x, y = path[:, 0], path[:, 1]
    # print(f'minx = {min(x)}, maxx = {max(x)}, miny = {min(y)}, maxy = {max(y)}')
    # print(f'x0 = {x0}, x1 = {x1}, y0 = {y0}, y1 = {y1}')
    in_box = (min(x) >= x0) & (max(x) <= x1) & (min(y) >= y0) & (max(y) <= y1)
    # in_box = (min(y) >= y0) & (max(y) <= y1)
    long_enough = euclidean_length(path) >= len_min
    # print(in_box, long_enough)
    return in_box and long_enough


def dijkstra_2d_least_resistance(K, start_region, end_region, average='harmonic'):
    """
    Finds the least-resistance path on a 2D conductivity lattice using Dijkstra's algorithm,
    considering 8-neighbor connectivity and user-selected averaging (harmonic or arithmetic)
    of bond conductivities.

    Args:
        K (np.ndarray): 2D array of conductivities (higher = easier to move).
        start_region (list[tuple[int,int]]): List of (row, col) coordinates defining the start region.
        end_region (list[tuple[int,int]]): List of (row, col) coordinates defining the end region.
        average (str): Either 'harmonic' or 'arithmetic'. Determines how to average conductivities
                       between neighboring sites.

    Returns:
        tuple:
            - path (np.ndarray): Optimal path as list of (row, col) coordinates.
            - total_resistance (np.longdouble): Total resistance along the optimal path.
            - actual_start_node (tuple[int,int]): The specific node where the path starts.
            - actual_end_node (tuple[int,int]): The specific node where the path ends.
            - resistance_map (np.ndarray): 2D array of minimal cumulative resistance.
    """

    Lx, Ly = K.shape
    pq = []
    resist = {}
    predecessors = {}
    visited = set()

    # --- Initialize all nodes with infinite resistance ---
    for r in range(Lx):
        for c in range(Ly):
            resist[(r, c)] = np.longdouble(np.inf)

    # --- Start region initialization ---
    for r, c in start_region:
        resist[(r, c)] = 0.0
        heapq.heappush(pq, (0.0, r, c))

    # --- 8-neighbor moves ---
    moves = [
        (-1, 0), (1, 0), (0, -1), (0, 1),     # N, S, W, E
        (-1, -1), (-1, 1), (1, -1), (1, 1)    # NW, NE, SW, SE
    ]

    actual_end_node = None

    while pq:
        current_res, r, c = heapq.heappop(pq)
        if (r, c) in visited:
            continue
        visited.add((r, c))

        # --- Check if reached end region ---
        if (r, c) in end_region:
            actual_end_node = (r, c)
            break

        # --- Loop over neighbors ---
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < Lx and 0 <= nc < Ly):
                continue

            k1, k2 = K[r, c], K[nr, nc]
            if k1 <= 0 or k2 <= 0:
                continue  # skip invalid or zero conductivity

            # --- Compute bond conductivity ---
            if average == 'harmonic':
                k_bond = 2.0 * k1 * k2 / (k1 + k2)
            elif average == 'arithmetic':
                k_bond = 0.5 * (k1 + k2)
            else:
                raise ValueError("average must be either 'harmonic' or 'arithmetic'")

            if k_bond <= 0:
                continue

            # --- Distance weighting (diagonals longer) ---
            length_factor = np.sqrt(2.0) if dr != 0 and dc != 0 else 1.0

            # --- Edge resistance = path length / bond conductivity ---
            edge_res = length_factor / k_bond
            new_res = current_res + edge_res

            if new_res < resist[(nr, nc)]:
                resist[(nr, nc)] = new_res
                predecessors[(nr, nc)] = (r, c)
                heapq.heappush(pq, (new_res, nr, nc))

    # --- Backtrack to reconstruct path ---
    path = []
    total_resistance = np.longdouble(np.inf)
    actual_start_node = None

    if actual_end_node is not None:
        total_resistance = resist[actual_end_node]
        node = actual_end_node
        while node is not None:
            path.append(node)
            if node in start_region:
                actual_start_node = node
                break
            node = predecessors.get(node)
        path.reverse()

    # --- Convert resistance dict to 2D array ---
    resistance_map = np.full_like(K, np.inf, dtype=np.longdouble)
    for (r, c), val in resist.items():
        resistance_map[r, c] = val

    # Convert path to numpy array and swap columns (row, col) -> (col, row)
    if path:
        path_array = np.array(path)[:, [1, 0]]
    else:
        path_array = np.empty((0, 2), dtype=int)

    return path_array, total_resistance, actual_start_node, actual_end_node, resistance_map