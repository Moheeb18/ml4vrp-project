def route_distance(route, distance_matrix):

    total = 0

    for i in range(len(route)-1):
        total += distance_matrix[route[i]][route[i+1]]

    return total


def capacity_check(route, demand, capacity):

    load = 0

    for customer in route:

        load += demand[customer]

    return load <= capacity


def split_routes(route, demand, capacity):

    routes = []
    current_route = [0]

    load = 0

    for customer in route:

        if load + demand[customer] <= capacity:

            current_route.append(customer)
            load += demand[customer]

        else:

            current_route.append(0)
            routes.append(current_route)

            current_route = [0, customer]
            load = demand[customer]

    current_route.append(0)
    routes.append(current_route)

    return routes
