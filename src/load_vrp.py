import numpy as np

def load_vrp(file_path):

    coords = []
    demands = []
    capacity = None

    reading_coords = False
    reading_demands = False

    with open(file_path) as f:
        lines = f.readlines()

    for line in lines:

        if "CAPACITY" in line:
            capacity = int(line.split()[-1])

        if "NODE_COORD_SECTION" in line:
            reading_coords = True
            continue

        if "DEMAND_SECTION" in line:
            reading_coords = False
            reading_demands = True
            continue

        if reading_coords:
            parts = line.split()
            if len(parts) == 3:
                coords.append([float(parts[1]), float(parts[2])])

        if reading_demands:
            parts = line.split()
            if len(parts) == 2:
                demands.append(int(parts[1]))

    coords = np.array(coords)
    demands = np.array(demands)

    depot = coords[0]

    return coords, demands, capacity, depot
