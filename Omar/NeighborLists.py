def build_neighbor_lists(distance_matrix, k=20):
    n = len(distance_matrix)
    neighbor_lists = []

    for i in range(n):
        neighbors = [(j, distance_matrix[i][j]) for j in range(n) if j != i]
        neighbors.sort(key=lambda x: x[1])
        neighbor_lists.append([j for j, _ in neighbors[:k]])

    return neighbor_lists


def route_distance(route, distance_matrix):
    total = 0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i]][route[i + 1]]
    return total


def total_distance(routes, distance_matrix):
    return sum(route_distance(route, distance_matrix) for route in routes)


def route_demand(route, demands):
    return sum(demands[node] for node in route if node != 0)