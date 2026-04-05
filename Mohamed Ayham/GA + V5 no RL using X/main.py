import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from load_vrp import load_vrp
from distance_matrix import compute_distance_matrix
from ga_core import run_ga
from routes_utilities import route_distance
from ga_fitness import VEHICLE_PENALTY

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GA_CONFIG = {
    "pop_size"       : 200,
    "num_generations": 500,
    "elite_size"     : 10,
    "tournament_size": 4,
    "crossover_rate" : 0.9,
    "mutation_rate"  : 0.03,
    "patience"       : 150,
    "ls_elite_size"  : 5,
    "ls_final"       : True,
}

NUM_RUNS = 3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(BASE_DIR, "solutions")


# ---------------------------------------------------------------------------
# SOLUTION FILE OUTPUT  (CVRPLIB format for ML4VRP submission)
# ---------------------------------------------------------------------------

def write_solution_file(routes, instance_name, output_dir):
    """
    Write solution in CVRPLIB format:
        Route #1: 3 1 2
        Route #2: 6 5 4

    Customer numbering: internal depot=0, customers=1..n are written as-is
    which matches CVRPLIB convention where depot (node 1 in TSPLIB) is
    stripped and customers are numbered 1..n-1.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, instance_name + ".sol")
    with open(out_path, "w") as f:
        for i, route in enumerate(routes, start=1):
            customers = [str(c) for c in route if c != 0]
            f.write(f"Route #{i}: {' '.join(customers)}\n")
    print(f"  Solution written → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def print_routes(routes, demands, distance_matrix):
    total_dist   = sum(route_distance(r, distance_matrix) for r in routes)
    num_vehicles = len(routes)
    obj          = VEHICLE_PENALTY * num_vehicles + total_dist

    print(f"\n{'─'*60}")
    print(f"  Vehicles used : {num_vehicles}")
    print(f"  Total distance: {total_dist:.4f}")
    print(f"  Objective     : {obj:.4f}  "
          f"(= {VEHICLE_PENALTY}×{num_vehicles} + {total_dist:.4f})")
    print(f"{'─'*60}")
    for idx, route in enumerate(routes):
        load  = sum(demands[c] for c in route)
        stops = len(route) - 2
        path  = " → ".join(map(str, route))
        print(f"  Route {idx+1:>2} | {stops} stops | load={load} | {path}")
    print(f"{'─'*60}")


def plot_solution(coords, routes, history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cmap = plt.cm.get_cmap("tab20", max(len(routes), 1))
    for idx, route in enumerate(routes):
        xs = [coords[n][0] for n in route]
        ys = [coords[n][1] for n in route]
        ax1.plot(xs, ys, "-o", color=cmap(idx), linewidth=1.5,
                 markersize=4, label=f"Route {idx+1}")

    depot = coords[0]
    ax1.plot(depot[0], depot[1], "k*", markersize=14, zorder=5, label="Depot")
    ax1.set_title(title)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend(fontsize=7, loc="best", ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.plot(history, color="steelblue", linewidth=1.5)
    ax2.set_title("Convergence — Objective per Generation")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Objective (1000×NV + TD)")
    ax2.set_xlim(0, len(history))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def solve_instance(file_path):
    """Load one .vrp file, run the GA, return best objective and routes."""
    print(f"\n{'='*60}")
    print(f"  Instance: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    coords, demands, capacity, depot = load_vrp(file_path)
    dist          = compute_distance_matrix(coords)
    num_customers = len(coords) - 1

    print(f"  Customers : {num_customers}")
    print(f"  Capacity  : {capacity}")

    # In main.py, update solve_instance:
    config = GA_CONFIG.copy()
    if num_customers > 150:
        config["num_generations"] = 800
        config["patience"]        = 150
        config["pop_size"]        = 150   # smaller pop saves time on large instances
    elif num_customers > 50:
        config["num_generations"] = 600
        config["patience"]        = 150

    best_chrom, best_routes, best_obj, history = run_ga(
        coords, demands, capacity, dist, **config)

    print(f"\n  ✓ Best objective: {best_obj:.4f}")
    print_routes(best_routes, demands, dist)

    instance_name = os.path.splitext(os.path.basename(file_path))[0]
    fig = plot_solution(
        coords, best_routes, history,
        title=f"GA — {instance_name}\nObj: {best_obj:.2f}"
    )
    plot_path = file_path.replace(".vrp", "_solution.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved  → {plot_path}")

    return best_obj, best_routes


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "data")

    if os.path.isfile(target):
        best_obj    = float("inf")
        best_routes = None

        for run in range(NUM_RUNS):
            print(f"\n--- Run {run+1}/{NUM_RUNS} ---")
            obj, routes = solve_instance(target)
            if obj < best_obj:
                best_obj    = obj
                best_routes = routes

        instance_name = os.path.splitext(os.path.basename(target))[0]
        write_solution_file(best_routes, instance_name, SOLUTIONS_DIR)
        print(f"\nBest objective across {NUM_RUNS} runs: {best_obj:.4f}")

    elif os.path.isdir(target):
        results = {}

        for fname in sorted(os.listdir(target)):
            if not fname.endswith(".vrp"):
                continue

            fpath           = os.path.join(target, fname)
            best_obj_run    = float("inf")
            best_routes_run = None

            for run in range(NUM_RUNS):
                print(f"\n--- Run {run+1}/{NUM_RUNS} ---")
                obj, routes = solve_instance(fpath)
                if obj < best_obj_run:
                    best_obj_run    = obj
                    best_routes_run = routes

            results[fname] = best_obj_run
            instance_name  = os.path.splitext(fname)[0]
            write_solution_file(best_routes_run, instance_name, SOLUTIONS_DIR)

        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        for name, obj in results.items():
            print(f"  {name:<30} {obj:.4f}")

    else:
        print(f"Error: '{target}' is not a valid file or directory.")
        sys.exit(1)
