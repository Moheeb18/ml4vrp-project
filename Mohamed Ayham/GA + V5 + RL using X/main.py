import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from load_vrp import load_vrp
from distance_matrix import compute_distance_matrix
from ga_core import run_ga
from rl_agent import RLParameterAgent, ACTIONS
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
    "q_table_path"   : "q_table.npy",
}

NUM_RUNS        = 5
SOLUTIONS_DIR   = "../solutions"    # output folder for solution files


# ---------------------------------------------------------------------------
# CVRPLIB SOLUTION OUTPUT
# ---------------------------------------------------------------------------

def write_solution_file(routes, instance_name, output_dir):
    """
    Write solution in CVRPLIB format required by ML4VRP competition.

    Format:
        Route #1: 3 1 2
        Route #2: 6 5 4

    IMPORTANT — node numbering convention:
        TSPLIB95 instances number the depot as node 1 and customers as
        nodes 2..n. The CVRPLIB solution format renumbers customers as
        1..(n-1), i.e. subtract 1 from every internal node index.
        The depot is never written in the route.

    Args:
        routes        : list of routes in internal format [0, c1, c2, ..., 0]
                        where 0 is depot and customers are 1-indexed internally
        instance_name : filename without extension (used for output filename)
        output_dir    : folder to write solution files into
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, instance_name + ".sol")

    with open(out_path, "w") as f:
        for i, route in enumerate(routes, start=1):
            # Strip depot (0) from both ends, convert to CVRPLIB numbering
            # Internal node index c → CVRPLIB customer number c
            # (internal depot=0, customers=1..n → CVRPLIB customers=1..n-1,
            #  so internal customer c maps to CVRPLIB customer c as-is
            #  because TSPLIB depot=1 maps to internal 0,
            #  and TSPLIB customer k maps to internal k-1.)
            customers = [str(c) for c in route if c != 0]
            f.write(f"Route #{i}: {' '.join(customers)}\n")

    print(f"  Solution written → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def print_routes(routes, demands, distance_matrix):
    total_dist = sum(route_distance(r, distance_matrix) for r in routes)
    num_vehicles = len(routes)
    obj = VEHICLE_PENALTY * num_vehicles + total_dist

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


def solve_instance(file_path, rl_agent):
    """Run the GA on one instance and return best result + updated agent."""
    print(f"\n{'='*60}")
    print(f"  Instance: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    coords, demands, capacity, depot = load_vrp(file_path)
    dist          = compute_distance_matrix(coords)
    num_customers = len(coords) - 1

    print(f"  Customers : {num_customers}")
    print(f"  Capacity  : {capacity}")

    config = GA_CONFIG.copy()
    if num_customers > 50:
        config["num_generations"] = 1000
        config["patience"]        = 200

    best_chrom, best_routes, best_obj, history, rl_agent = run_ga(
        coords, demands, capacity, dist,
        rl_agent=rl_agent,
        **config,
    )

    print(f"\n  ✓ Best objective: {best_obj:.4f}")
    print_routes(best_routes, demands, dist)

    # Plot
    instance_name = os.path.splitext(os.path.basename(file_path))[0]
    fig = plot_solution(
        coords, best_routes, history,
        title=f"GA+RL — {instance_name}\nObj: {best_obj:.2f}"
    )
    plot_path = file_path.replace(".vrp", "_solution.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved  → {plot_path}")

    return best_obj, best_routes, rl_agent


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    target = sys.argv[1] if len(sys.argv) > 1 else "../data"

    # Shared RL agent — learns across all instances and runs
    shared_agent = RLParameterAgent()
    shared_agent.load_q_table(GA_CONFIG["q_table_path"])

    if os.path.isfile(target):
        best_obj    = float("inf")
        best_routes = None
        best_coords = None
        best_demands = None
        best_dist    = None

        coords, demands, capacity, _ = load_vrp(target)
        dist = compute_distance_matrix(coords)

        for run in range(NUM_RUNS):
            print(f"\n--- Run {run+1}/{NUM_RUNS} ---")
            obj, routes, shared_agent = solve_instance(target, shared_agent)
            if obj < best_obj:
                best_obj    = obj
                best_routes = routes

        # Write solution file
        instance_name = os.path.splitext(os.path.basename(target))[0]
        write_solution_file(best_routes, instance_name, SOLUTIONS_DIR)
        print(f"\nBest objective across {NUM_RUNS} runs: {best_obj:.4f}")

    elif os.path.isdir(target):
        results = {}

        for fname in sorted(os.listdir(target)):
            if not fname.endswith(".vrp"):
                continue

            fpath = os.path.join(target, fname)
            best_obj_run    = float("inf")
            best_routes_run = None

            for run in range(NUM_RUNS):
                print(f"\n--- Run {run+1}/{NUM_RUNS} ---")
                obj, routes, shared_agent = solve_instance(fpath, shared_agent)
                if obj < best_obj_run:
                    best_obj_run    = obj
                    best_routes_run = routes

            results[fname] = best_obj_run

            # Write solution file for this instance
            instance_name = os.path.splitext(fname)[0]
            write_solution_file(best_routes_run, instance_name, SOLUTIONS_DIR)

        # Summary
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        for name, obj in results.items():
            print(f"  {name:<30} {obj:.4f}")

        shared_agent.summary()

    else:
        print(f"Error: '{target}' is not a valid file or directory.")
        sys.exit(1)
