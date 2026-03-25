def solve_vrp_multistart(coords, demands, capacity, distance_matrix):
    n = len(coords)
    best_routes = None
    best_distance = float('inf')

    for start in range(1, n):
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
            candidates = [
                i for i in unvisited
                if demands[i] <= remaining_capacity
            ]

            if not candidates:
                route.append(0)
                routes.append(route)

                current = 0
                remaining_capacity = capacity
                route = [0]
                continue

            next_node = min(
                candidates,
                key=lambda i: distance_matrix[current][i]
            )

            route.append(next_node)
            unvisited.remove(next_node)
            remaining_capacity -= demands[next_node]
            current = next_node

        route.append(0)
        routes.append(route)

        total = 0
        for r in routes:
            for i in range(len(r) - 1):
                total += distance_matrix[r[i]][r[i+1]]

        if total < best_distance:
            best_distance = total
            best_routes = routes

    return best_routes, best_distance