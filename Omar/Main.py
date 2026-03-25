from load_vrp import load_vrp
from distance_matrix import compute_distance_matrix
from Solver import solve_vrp
from distance import total_distance
from MultiStart import solve_vrp_multistart
from FinalOptimize import optimize_routes 
from FinalOptimize import two_opt
from SavingsAlgo import savings_algorithm
from SavingsRelocate import relocate 
from Swap import swap
import random
import os


def run_vrp(coords, demands, capacity, distance_matrix):
    routes = savings_algorithm(coords, demands, capacity, distance_matrix)
    routes = relocate(routes, demands, capacity, distance_matrix)
    routes = swap(routes, demands, capacity, distance_matrix)
    routes = optimize_routes(routes, distance_matrix)

    total_dist = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_dist += distance_matrix[route[i]][route[i+1]]

    return total_dist


def run_vrp(coords, demands, capacity, distance_matrix):
    routes = savings_algorithm(coords, demands, capacity, distance_matrix)
    routes = relocate(routes, demands, capacity, distance_matrix)
    routes = swap(routes, demands, capacity, distance_matrix)
    routes = optimize_routes(routes, distance_matrix)

    total_dist = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_dist += distance_matrix[route[i]][route[i+1]]

    return total_dist


def compute_total(routes, distance_matrix):
    total = 0
    for route in routes:
        for i in range(len(route) - 1):
            total += distance_matrix[route[i]][route[i+1]]
    return total


for file in os.listdir("data"):
    if file.endswith(".vrp"):

        path = os.path.join("data", file)

        coords, demands, capacity, depot = load_vrp(path)
        distance_matrix = compute_distance_matrix(coords)

        nn_routes = solve_vrpOk(coords, demands, capacity, distance_matrix)
        nn_total = compute_total(nn_routes, distance_matrix)

        s_routes = savings_algorithm(coords, demands, capacity, distance_matrix)
        s_total = compute_total(s_routes, distance_matrix)

        f_routes = savings_algorithm(coords, demands, capacity, distance_matrix)
        f_routes = relocate(f_routes, demands, capacity, distance_matrix)
        f_routes = swap(f_routes, demands, capacity, distance_matrix)
        f_routes = optimize_routes(f_routes, distance_matrix)
        f_total = compute_total(f_routes, distance_matrix)

        print(f"{file},{nn_total},{s_total},{f_total}")

'''

distance_matrix = compute_distance_matrix(coords)

routes = solve_vrp(coords, demands, capacity, distance_matrix)

print(routes)
print("Total Distance:", total_distance(routes, distance_matrix))

routes, best_distance = solve_vrp_multistart(coords, demands, capacity, distance_matrix)

print("Best Routes:", routes)
print("Best Distance:", best_distance)

###########################################################################################

routes, best_distance = solve_vrp_multistart(coords, demands, capacity, distance_matrix)

optimized_routes = optimize_routes(routes, distance_matrix)

optimized_distance = total_distance(optimized_routes, distance_matrix)

print("Before 2-opt:", best_distance)
print("After 2-opt:", optimized_distance)

# Savings Algo
routes = savings_algorithm(coords, demands, capacity, distance_matrix)

optimized_routes = optimize_routes(routes, distance_matrix)

total = total_distance(optimized_routes, distance_matrix)

if __name__ == "__main__":

    routes = savings_algorithm(coords, demands, capacity, distance_matrix)

    optimized_routes = optimize_routes(routes, distance_matrix)

    total_dist = 0

if __name__ == "__main__":

    routes = savings_algorithm(coords, demands, capacity, distance_matrix)

    optimized_routes = optimize_routes(routes, distance_matrix)

    total_dist = 0

    for route in optimized_routes:
        for i in range(len(route) - 1):
            total_dist += distance_matrix[route[i]][route[i+1]]

    print(f"Total Distance: {total_dist}")
    

    routes = savings_algorithm(coords, demands, capacity, distance_matrix)

routes = relocate(routes, demands, capacity, distance_matrix)

optimized_routes = optimize_routes(routes, distance_matrix)

if __name__ == "__main__":

    routes = savings_algorithm(coords, demands, capacity, distance_matrix)

    routes = relocate(routes, demands, capacity, distance_matrix)

    optimized_routes = optimize_routes(routes, distance_matrix)

    total_dist = 0

    for route in optimized_routes:
        for i in range(len(route) - 1):
            total_dist += distance_matrix[route[i]][route[i+1]]

    print(f"Total Distance: {total_dist}")


    # Swap SavingsAlgo

    routes = savings_algorithm(coords, demands, capacity, distance_matrix)

routes = relocate(routes, demands, capacity, distance_matrix)

routes = swap(routes, demands, capacity, distance_matrix)

optimized_routes = optimize_routes(routes, distance_matrix)

if __name__ == "__main__":

    routes = savings_algorithm(coords, demands, capacity, distance_matrix)

    routes = relocate(routes, demands, capacity, distance_matrix)

    routes = swap(routes, demands, capacity, distance_matrix)

    optimized_routes = optimize_routes(routes, distance_matrix)

    total_dist = 0

    for route in optimized_routes:
        for i in range(len(route) - 1):
            total_dist += distance_matrix[route[i]][route[i+1]]

    print(f"Total Distance: {total_dist}")

    ''' 