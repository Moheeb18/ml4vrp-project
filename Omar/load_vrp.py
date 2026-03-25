import numpy as np

def load_vrp(file_path):
    coords = []
    demands = []
    capacity = None

    with open(file_path) as f:
        lines = f.readlines()
    
    print("TOTAL LINES:", len(lines))
    print("FIRST 5 LINES:", lines[:5])

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if "CAPACITY" in line:
            capacity = int(line.split()[-1])

        elif "NODE_COORD_SECTION" in line:
            i += 1
            while "DEMAND_SECTION" not in lines[i]:
                parts = lines[i].strip().split()
                if len(parts) == 3:
                    _, x, y = parts
                    coords.append([int(x), int(y)])
                i += 1
            continue

        elif "DEMAND_SECTION" in line:
            i += 1
            while "DEPOT_SECTION" not in lines[i]:
                parts = lines[i].strip().split()
                if len(parts) == 2:
                    _, demand = parts
                    demands.append(int(demand))
                i += 1
            continue

        i += 1

    coords = np.array(coords)
    demands = np.array(demands)

    depot = coords[0]

    return coords, demands, capacity, depot

