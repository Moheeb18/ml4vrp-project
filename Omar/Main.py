from load_vrp import load_vrp
from distance_matrix import compute_distance_matrix
from Solver import solve_vrp
from distance import total_distance
from MultiStart import solve_vrp_multistart
from FinalOptimize import optimize_routes
from SavingsAlgo import savings_algorithm
from SavingsRelocate import relocate
from Swap import swap
import os


def run_vrp_pipeline(coords, demands, capacity, distance_matrix):
    """
    Runs the full VRP pipeline:
    Savings → Relocate → Swap → Final Optimization
    Returns optimized routes and total distance.
    """
    routes = savings_algorithm(coords, demands, capacity, distance_matrix)
    routes = relocate(routes, demands, capacity, distance_matrix)
    routes = swap(routes, demands, capacity, distance_matrix)
    routes = optimize_routes(routes, distance_matrix)

    total_dist = total_distance(routes, distance_matrix)
    return routes, total_dist


def compute_total(routes, distance_matrix):
    """
    Computes total distance for given routes.
    """
    total = 0
    for route in routes:
        for i in range(len(route) - 1):
            total += distance_matrix[route[i]][route[i + 1]]
    return total


if __name__ == "__main__":
    data_folder = "data"

    for file in os.listdir(data_folder):
        if file.endswith(".vrp"):
            path = os.path.join(data_folder, file)

            coords, demands, capacity, depot = load_vrp(path)
            distance_matrix = compute_distance_matrix(coords)

            # Nearest Neighbor solution
            nn_routes = solve_vrp(coords, demands, capacity, distance_matrix)
            nn_total = compute_total(nn_routes, distance_matrix)

            # Savings algorithm only
            s_routes = savings_algorithm(coords, demands, capacity, distance_matrix)
            s_total = compute_total(s_routes, distance_matrix)

            # Full optimized pipeline
            f_routes, f_total = run_vrp_pipeline(
                coords, demands, capacity, distance_matrix
            )

            print(f"{file},{nn_total},{s_total},{f_total}")