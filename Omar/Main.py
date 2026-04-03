from Solver import solve_vrp
from SavingsAlgo import savings_algorithm
import os
import sys

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent)

from src.load_vrp import load_vrp
from src.distance_matrix import compute_distance_matrix
from ActualFinalOptimize import actual_final_optimize


def clean_routes(routes):
    cleaned = []
    for route in routes:
        customers = [node for node in route if node != 0]
        if customers:
            cleaned.append([0] + customers + [0])
    return cleaned


def compute_total(routes, distance_matrix):
    total = 0
    for route in routes:
        for i in range(len(route) - 1):
            total += distance_matrix[route[i]][route[i + 1]]
    return total


def number_of_vehicles(routes):
    return len(clean_routes(routes))


def objective_function(routes, distance_matrix, c=1000):
    routes = clean_routes(routes)
    nv = number_of_vehicles(routes)
    td = compute_total(routes, distance_matrix)
    return c * nv + td


def validate_and_evaluate(routes, demands, capacity, distance_matrix, num_customers, c=1000):
    routes = clean_routes(routes)
    visited = []

    for route in routes:
        load = 0
        for node in route:
            if node != 0:
                visited.append(node)
                load += demands[node]
        if load > capacity:
            return False, None, "Capacity exceeded"

    expected = list(range(1, num_customers + 1))
    if sorted(visited) != expected:
        return False, None, "Customers missing or duplicated"

    nv = number_of_vehicles(routes)
    td = compute_total(routes, distance_matrix)
    obj = c * nv + td

    return True, {"NV": nv, "TD": td, "Objective": obj}, "Feasible"


def run_vrp_pipeline(coords, demands, capacity, distance_matrix, c=1000):
    routes = savings_algorithm(coords, demands, capacity, distance_matrix)
    routes, total_dist, total_obj = actual_final_optimize(
        routes, distance_matrix, demands, capacity, c
    )
    return clean_routes(routes), total_dist, total_obj


if __name__ == "__main__":
    data_folder = "data"
    c = 1000

    print("file,nn_nv,nn_td,nn_obj,s_nv,s_td,s_obj,f_nv,f_td,f_obj,status")

    for file in os.listdir(data_folder):
        if file.endswith(".vrp"):
            path = os.path.join(data_folder, file)

            coords, demands, capacity, depot = load_vrp(path)
            distance_matrix = compute_distance_matrix(coords)

            nn_routes = clean_routes(solve_vrp(coords, demands, capacity, distance_matrix))
            nn_nv = number_of_vehicles(nn_routes)
            nn_td = compute_total(nn_routes, distance_matrix)
            nn_obj = objective_function(nn_routes, distance_matrix, c)

            s_routes = clean_routes(savings_algorithm(coords, demands, capacity, distance_matrix))
            s_nv = number_of_vehicles(s_routes)
            s_td = compute_total(s_routes, distance_matrix)
            s_obj = objective_function(s_routes, distance_matrix, c)

            f_routes, f_td, f_obj = run_vrp_pipeline(
                coords, demands, capacity, distance_matrix, c
            )
            f_nv = number_of_vehicles(f_routes)

            feasible, result, message = validate_and_evaluate(
                f_routes, demands, capacity, distance_matrix, len(demands) - 1, c
            )

            print(f"{file},{nn_nv},{nn_td},{nn_obj},{s_nv},{s_td},{s_obj},{f_nv},{f_td},{f_obj},{message}")

            print("========== Final Solution Evaluation ==========")
            if feasible:
                print(f"Feasibility: {message}")
                print(f"Vehicles Used (NV): {result['NV']}")
                print(f"Total Distance (TD): {result['TD']:.2f}")
                print(f"Objective Value: {result['Objective']:.2f}")
            else:
                print(f"Feasibility: {message}")
            print()