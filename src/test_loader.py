from load_vrp import load_vrp
from distance_matrix import compute_distance_matrix

coords, demand, capacity, depot = load_vrp("ml4vrp-project/data/A-n32-k5.vrp")

print("Depot:", depot)
print("Capacity:", capacity)
print("First 5 coordinates:", coords[:5])
print("First 5 demands:", demand[:5])

dist = compute_distance_matrix(coords)

print("Distance matrix size:", dist.shape)
