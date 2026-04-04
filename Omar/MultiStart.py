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

def print_multistart_results(coords, demands, capacity, distance_matrix):
    n = len(coords)
    best_routes = None
    best_distance = float('inf')
    best_start = None

    print(f"{'Start Customer':<18}{'Total Distance':<18}{'Selected as Best'}")
    print("-" * 50)

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
                total += distance_matrix[r[i]][r[i + 1]]

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