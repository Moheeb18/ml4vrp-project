def solve_vrp(coords, demands, capacity, distance_matrix):
    n = len(coords)
    unvisited = set(range(1, n))
    routes = []

    while unvisited:
        current = 0
        remaining_capacity = capacity
        route = [0]

        while True:
            candidates = [
                i for i in unvisited
                if demands[i] <= remaining_capacity
            ]

            if not candidates:
                break

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

    return routes

