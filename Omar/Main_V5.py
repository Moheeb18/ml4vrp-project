from Solver_v5 import solve_vrp, improve_routes
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


def compute_score(routes, distance_matrix, penalty_per_vehicle=1000):
    td = compute_total(routes, distance_matrix)
    nv = len(routes)
    score = td + penalty_per_vehicle * nv
    return nv, td, score


def run_vrp_pipeline(coords, demands, capacity, distance_matrix):
    routes = savings_algorithm(coords, demands, capacity, distance_matrix)

    n = len(distance_matrix)

    if n < 80:
        k = 20
        lns_iterations = 40
        remove_ratio = 0.15
    elif n < 200:
        k = 25
        lns_iterations = 60
        remove_ratio = 0.20
    else:
        k = 30
        lns_iterations = 80
        remove_ratio = 0.25

    routes = improve_routes(
        routes,
        demands,
        capacity,
        distance_matrix,
        k=k,
        lns_iterations=lns_iterations,
        remove_ratio=remove_ratio
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
    print("file,nn_total,s_total,f_total,f_nv,f_score")

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

            f_nv, _, f_score = compute_score(f_routes, distance_matrix)

            print(f"{file},{nn_total},{s_total},{f_total},{f_nv},{f_score}")