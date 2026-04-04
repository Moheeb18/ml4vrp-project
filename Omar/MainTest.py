from MultiStart import solve_vrp_multistart
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


def print_multistart_results(coords, demands, capacity, distance_matrix):
    n = len(coords)
    best_routes = None
    best_distance = float("inf")
    best_start = None

    print(f"{'Start Customer':<18}{'Total Distance':<18}{'Selected as Best'}")
    print("-" * 50)

    for start in range(1, n):
        routes, total = solve_vrp_multistart_from_start(
            coords, demands, capacity, distance_matrix, start
        )

        is_best = "No"
        if total < best_distance:
            best_distance = total
            best_routes = routes
            best_start = start
            is_best = "Yes"

        print(f"{start:<18}{total:<18.2f}{is_best}")

    print("-" * 50)
    print(f"Best starting customer: {best_start}")
    print(f"Best total distance: {best_distance:.2f}")
    print(f"Best routes: {best_routes}")

    return best_routes, best_distance, best_start


def solve_vrp_multistart_from_start(coords, demands, capacity, distance_matrix, start):
    n = len(coords)
    unvisited = set(range(1, n))
    routes = []

    current = 0
    remaining_capacity = capacity
    route = [0]

    if demands[start] <= remaining_capacity:
        route.append(start)
        unvisited.remove(start)
        remaining_capacity -= demands[start]
        current = start

    while unvisited:
        candidates = [i for i in unvisited if demands[i] <= remaining_capacity]

        if not candidates:
            route.append(0)
            routes.append(route)

            current = 0
            remaining_capacity = capacity
            route = [0]
            continue

        next_node = min(candidates, key=lambda i: distance_matrix[current][i])

        route.append(next_node)
        unvisited.remove(next_node)
        remaining_capacity -= demands[next_node]
        current = next_node

    route.append(0)
    routes.append(route)

    total = compute_total(routes, distance_matrix)
    return routes, total


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

            print(f"\n=== {file} ===")
            best_routes, best_distance, best_start = print_multistart_results(
                coords, demands, capacity, distance_matrix
            )

            print(f"\nFinal Best Summary for {file}")
            print(f"Best start: {best_start}")
            print(f"Best distance: {best_distance:.2f}")
            print(f"Best routes: {best_routes}")