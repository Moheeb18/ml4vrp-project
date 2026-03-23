import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from load_vrp import load_vrp
from distance_matrix import compute_distance_matrix
from ga_core import run_ga


# ---------------------------------------------------------------------------
# CONFIG — tweak these to control the GA
# ---------------------------------------------------------------------------
GA_CONFIG = {
    "pop_size"       : 200,
    "num_generations": 500,
    "elite_size"     : 10,
    "tournament_size": 4,
    "crossover_rate" : 0.9,
    "mutation_rate"  : 0.03,
    "patience"       : 80,
    "verbose"        : True,
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def print_routes(routes, demands):
    """Print each route with its load and stop count."""
    total_dist_note = ""
    print(f"\n{'─'*50}")
    print(f"  {'Route':<6} {'Stops':>5}   Path")
    print(f"{'─'*50}")
    for idx, route in enumerate(routes):
        load = sum(demands[c] for c in route)
        stops = len(route) - 2  # exclude depot at each end
        path  = " → ".join(map(str, route))
        print(f"  {idx+1:<6} {stops:>5} stops | load={load} | {path}")
    print(f"{'─'*50}")
    print(f"  Total vehicles used: {len(routes)}")


def plot_solution(coords, routes, title="CVRP – GA Solution"):
    """Visualise all routes on a 2-D scatter plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: solution routes ----
    cmap = plt.cm.get_cmap("tab20", len(routes))
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

    # ---- Right: convergence curve ----
    ax2.set_title("Convergence – Best Distance per Generation")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Total Distance")
    ax2.grid(True, alpha=0.3)
    # history is attached after run; placeholder filled in main()

    plt.tight_layout()
    return fig, ax2  # ax2 is filled after running the GA


def solve_instance(file_path):
    """Load one .vrp file, run the GA, print and plot results."""
    print(f"\n{'='*60}")
    print(f"  Instance: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    # -- Load data -----------------------------------------------------------
    coords, demands, capacity, depot = load_vrp(file_path)
    dist = compute_distance_matrix(coords)
    num_customers = len(coords) - 1

    print(f"  Customers : {num_customers}")
    print(f"  Capacity  : {capacity}")
    print(f"  Depot     : {depot}")

    # -- Run GA --------------------------------------------------------------
    best_chrom, best_routes, best_dist, history = run_ga(
        coords, demands, capacity, dist, **GA_CONFIG
    )

    # -- Results -------------------------------------------------------------
    print(f"\n  ✓ Best total distance : {best_dist:.4f}")
    print_routes(best_routes, demands)

    # -- Plot ----------------------------------------------------------------
    fig, ax_conv = plot_solution(coords, best_routes,
                                 title=f"GA Solution – {os.path.basename(file_path)}"
                                       f"\nTotal distance: {best_dist:.2f}")
    ax_conv.plot(history, color="steelblue", linewidth=1.5)
    ax_conv.set_xlim(0, len(history))

    plot_path = file_path.replace(".vrp", "_solution.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {plot_path}")
    plt.close(fig)

    return best_dist, best_routes


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Accept a specific file or a folder of .vrp instances
    target = sys.argv[1] if len(sys.argv) > 1 else "data"

    if os.path.isfile(target):
        solve_instance(target)

    elif os.path.isdir(target):
        results = {}
        for fname in sorted(os.listdir(target)):
            if fname.endswith(".vrp"):
                fpath = os.path.join(target, fname)
                dist, _ = solve_instance(fpath)
                results[fname] = dist

        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        for name, d in results.items():
            print(f"  {name:<30} {d:.4f}")

    else:
        print(f"Error: '{target}' is not a valid file or directory.")
        sys.exit(1)
