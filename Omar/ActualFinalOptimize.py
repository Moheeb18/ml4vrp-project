from FinalOptimize import optimize_routes
from SavingsRelocate import relocate
from Swap import swap


def route_distance(route, distance_matrix):
    total = 0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i]][route[i + 1]]
    return total


def total_distance(routes, distance_matrix):
    total = 0
    for route in routes:
        total += route_distance(route, distance_matrix)
    return total


def actual_final_optimize(routes, distance_matrix, demands, capacity):
    current_routes = [route[:] for route in routes]
    current_distance = total_distance(current_routes, distance_matrix)

    while True:
        improved = False

        new_routes = optimize_routes([route[:] for route in current_routes], distance_matrix)
        new_distance = total_distance(new_routes, distance_matrix)

        if new_distance < current_distance:
            current_routes = [route[:] for route in new_routes]
            current_distance = new_distance
            improved = True
            continue

        new_routes = relocate([route[:] for route in current_routes], demands, capacity, distance_matrix)
        new_distance = total_distance(new_routes, distance_matrix)

        if new_distance < current_distance:
            current_routes = [route[:] for route in new_routes]
            current_distance = new_distance
            improved = True
            continue

        new_routes = swap([route[:] for route in current_routes], demands, capacity, distance_matrix)
        new_distance = total_distance(new_routes, distance_matrix)

        if new_distance < current_distance:
            current_routes = [route[:] for route in new_routes]
            current_distance = new_distance
            improved = True
            continue

        if not improved:
            break

    return current_routes, current_distance