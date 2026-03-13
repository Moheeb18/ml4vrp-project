from load_vrp import load_vrp
from distance_matrix import compute_distance_matrix
from routes_utilities import route_distance, capacity_check, split_routes

coords, demand, capacity, depot = load_vrp("C:/Users/asus/ml4vrp-project/data/A-n32-k5")

print("Depot:", depot)
print("Capacity:", capacity)

print("First 5 coordinates:")
print(coords[:5])

print("First 5 demands:")
print(demand[:5])

dist = compute_distance_matrix(coords)

print("Distance matrix shape:", dist.shape)

route = [0,1,2,3,4,0]

print("Route distance:", route_distance(route, dist))

print("Capacity valid:", capacity_check(route, demand, capacity))

print("Split routes example:")
print(split_routes([1,2,3,4,5,6,7], demand, capacity))
