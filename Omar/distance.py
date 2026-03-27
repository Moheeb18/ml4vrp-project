import math


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def compute_distance_matrix(coords):
    n = len(coords)
    distance_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = euclidean_distance(coords[i], coords[j])

    return distance_matrix


def total_distance(routes, distance_matrix):
    total = 0
    for route in routes:
        for i in range(len(route) - 1):
            total += distance_matrix[route[i]][route[i + 1]]
    return total