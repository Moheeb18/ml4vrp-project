from Solver import solve_vrp, improve_routes
from SavingsAlgo import savings_algorithm
import os
import sys

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent)

from src.load_vrp import load_vrp
from src.distance_matrix import compute_distance_matrix


def compute_total(routes, distance_matrix):
    total = 0
    for route in routes:
        for i in range(len(route) - 1):
            total += distance_matrix[route[i]][route[i + 1]]
    return total


def run_vrp_pipeline(coords, demands, capacity, distance_matrix):
    routes = savings_algorithm(coords, demands, capacity, distance_matrix)

    routes = improve_routes(
        routes,
        demands,
        capacity,
        distance_matrix,
        k=25,
        lns_iterations=60,
        remove_ratio=0.20
    )

    total_dist = compute_total(routes, distance_matrix)
    return routes, total_dist


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "..", "data")
    data_folder = os.path.abspath(data_folder)

    print("Resolved data path:", data_folder)
    print("Exists?", os.path.exists(data_folder))
    print(os.listdir(data_folder))

    for file in os.listdir(data_folder):
        if file.endswith(".vrp"):
            path = os.path.join(data_folder, file)

            coords, demands, capacity, depot = load_vrp(path)
            distance_matrix = compute_distance_matrix(coords)

            nn_routes = solve_vrp(coords, demands, capacity, distance_matrix)
            nn_total = compute_total(nn_routes, distance_matrix)

            s_routes = savings_algorithm(coords, demands, capacity, distance_matrix)
            s_total = compute_total(s_routes, distance_matrix)

            f_routes, f_total = run_vrp_pipeline(
                coords, demands, capacity, distance_matrix
            )

            print(f"{file},{nn_total},{s_total},{f_total}")