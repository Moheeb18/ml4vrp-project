def two_opt(route, distance_matrix):
    best = route
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i == 1:
                    continue

                new_route = best[:]
                new_route[i:j] = best[j-1:i-1:-1]

                old_dist = 0
                new_dist = 0

                for k in range(len(best) - 1):
                    old_dist += distance_matrix[best[k]][best[k+1]]
                    new_dist += distance_matrix[new_route[k]][new_route[k+1]]

                if new_dist < old_dist:
                    best = new_route
                    improved = True

        route = best

    return best

def optimize_routes(routes, distance_matrix):
    optimized = []
    for route in routes:
        optimized.append(two_opt(route, distance_matrix))
    return optimized