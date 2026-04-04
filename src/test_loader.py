from load_vrp import load_vrp
from distance_matrix import compute_distance_matrix
from routes_utilities import route_distance, capacity_check, split_routes
import os

def run_all(data_folder):

    for file in os.listdir(data_folder):
        
       # if file.endswith(".vrp"):

            path = os.path.join(data_folder, file)

            print("\n==========")
            print("Instance:", file)

            coords, demand, capacity, depot = load_vrp(path)

            print("Depot:", depot)
            print("Capacity:", capacity)

            print("First 5 coordinates:")
            print(coords[:5])

            print("First 5 demands:")
            print(demand[:5])

            dist = compute_distance_matrix(coords)
            print("Distance matrix shape:", dist.shape)


if __name__ == "__main__":
    data_folder = "data"
    run_all(data_folder)





# route = [0,1,2,3,4,0]

# print("Route distance:", route_distance(route, dist))

# print("Capacity valid:", capacity_check(route, demand, capacity))

# print("Split routes example:")
# print(split_routes([1,2,3,4,5,6,7], demand, capacity))
