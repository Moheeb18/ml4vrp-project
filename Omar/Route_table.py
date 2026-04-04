import math


def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def build_distance_matrix(coords):
    n = len(coords)
    distance_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = euclidean_distance(coords[i], coords[j])

    return distance_matrix


def route_distance(route, distance_matrix):
    total = 0.0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i]][route[i + 1]]
    return total


def two_opt(route, distance_matrix):
    if len(route) <= 4:
        return route[:]

    best = route[:]
    improved = True

    while improved:
        improved = False
        best_cost = route_distance(best, distance_matrix)

        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i == 1:
                    continue

                candidate = best[:]
                candidate[i:j] = reversed(candidate[i:j])
                candidate_cost = route_distance(candidate, distance_matrix)

                if candidate_cost < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break

            if improved:
                break

    return best


def print_two_opt_table(routes, distance_matrix):
    print(f"{'Route':<8}{'Distance Before 2-opt':>25}{'Distance After 2-opt':>25}{'Improvement':>15}")
    print("-" * 73)

    total_before = 0.0
    total_after = 0.0

    for idx, route in enumerate(routes, start=1):
        before = route_distance(route, distance_matrix)
        optimized_route = two_opt(route, distance_matrix)
        after = route_distance(optimized_route, distance_matrix)
        improvement = before - after

        total_before += before
        total_after += after

        print(f"R{idx:<7}{before:>25.2f}{after:>25.2f}{improvement:>15.2f}")

    print("-" * 73)
    print(f"{'Total':<8}{total_before:>25.2f}{total_after:>25.2f}{(total_before - total_after):>15.2f}")


coords = [
    (0, 0),
    (2, 3),
    (5, 4),
    (1, 7),
    (6, 8),
    (8, 2),
    (7, 6)
]

distance_matrix = build_distance_matrix(coords)

routes = [
    [0, 1, 4, 2, 0],
    [0, 3, 6, 5, 0]
]

print_two_opt_table(routes, distance_matrix)