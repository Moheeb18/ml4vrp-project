def validate_and_evaluate(routes, demands, capacity, distance_matrix, num_customers, c=1000):
    visited = []

    for route in routes:
        load = 0
        for node in route:
            if node != 0:
                visited.append(node)
                load += demands[node]
        if load > capacity:
            return False, None, "Capacity exceeded"

    expected = list(range(1, num_customers + 1))
    if sorted(visited) != expected:
        return False, None, "Customers missing or duplicated"

    nv = len([r for r in routes if len([x for x in r if x != 0]) > 0])

    td = 0
    for route in routes:
        for i in range(len(route) - 1):
            td += distance_matrix[route[i]][route[i + 1]]

    objective = c * nv + td
    return True, {"NV": nv, "TD": td, "Objective": objective}, "Feasible"