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


def clean_routes(routes):
    cleaned = []
    for route in routes:
        customers = [node for node in route if node != 0]
        if customers:
            cleaned.append([0] + customers + [0])
    return cleaned


def number_of_vehicles(routes):
    return len(clean_routes(routes))


def objective_function(routes, distance_matrix, c=1000):
    cleaned_routes = clean_routes(routes)
    nv = len(cleaned_routes)
    td = total_distance(cleaned_routes, distance_matrix)
    return c * nv + td


def actual_final_optimize(routes, distance_matrix, demands, capacity, c=1000):
    current_routes = clean_routes([route[:] for route in routes])
    current_objective = objective_function(current_routes, distance_matrix, c)

    while True:
        improved = False

        new_routes = optimize_routes([route[:] for route in current_routes], distance_matrix)
        new_routes = clean_routes(new_routes)
        new_objective = objective_function(new_routes, distance_matrix, c)

        if new_objective < current_objective:
            current_routes = [route[:] for route in new_routes]
            current_objective = new_objective
            improved = True
            continue

        new_routes = relocate([route[:] for route in current_routes], demands, capacity, distance_matrix)
        new_routes = clean_routes(new_routes)
        new_objective = objective_function(new_routes, distance_matrix, c)

        if new_objective < current_objective:
            current_routes = [route[:] for route in new_routes]
            current_objective = new_objective
            improved = True
            continue

        new_routes = swap([route[:] for route in current_routes], demands, capacity, distance_matrix)
        new_routes = clean_routes(new_routes)
        new_objective = objective_function(new_routes, distance_matrix, c)

        if new_objective < current_objective:
            current_routes = [route[:] for route in new_routes]
            current_objective = new_objective
            improved = True
            continue

        if not improved:
            break

    final_routes = clean_routes(current_routes)
    final_distance = total_distance(final_routes, distance_matrix)
    final_objective = objective_function(final_routes, distance_matrix, c)

    return final_routes, final_distance, final_objective