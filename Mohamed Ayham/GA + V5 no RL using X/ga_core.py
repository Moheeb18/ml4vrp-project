import random
import numpy as np

from ga_representation import initialize_population, decode_chromosome
from ga_fitness import population_fitness, fitness_from_routes
from ga_operators import tournament_selection, order_crossover, combined_mutation
from local_search import local_search, routes_to_chromosome


def run_ga(
    coords,
    demands,
    capacity,
    distance_matrix,
    pop_size=200,
    num_generations=500,
    elite_size=10,
    tournament_size=4,
    crossover_rate=0.9,
    mutation_rate=0.03,
    verbose=True,
    patience=150,
    ls_elite_size=5,
    ls_final=True,
):
    """
    Memetic Genetic Algorithm for CVRP.

    Objective: minimise  1000 * num_vehicles + total_distance

    Pipeline:
      - Diverse seeding: 5 Clark-Wright variants (40%) + nearest-neighbour
        (20%) + random (40%) — no two seeds are identical
      - Tournament selection, Order Crossover (OX), combined mutation
      - Local search on top elites every 10 generations:
          2-opt → OR-opt → neighbour-list relocate/swap → 2-opt* →
          try_merge_routes → route elimination → inter-route relocation/swap
      - Elitism: top elite_size chromosomes carried forward each generation
      - Early stopping after patience generations without improvement
      - Final local search pass on best solution

    Args:
        coords          : np.array of node coordinates (row 0 = depot)
        demands         : np.array of demands
        capacity        : int, vehicle capacity
        distance_matrix : 2D np.array of pairwise distances
        pop_size        : population size
        num_generations : max generations
        elite_size      : elites carried forward unchanged each generation
        tournament_size : competitors per selection tournament
        crossover_rate  : probability of OX crossover
        mutation_rate   : probability of each mutation operator firing
        verbose         : print progress every 50 generations
        patience        : early stop after this many gens without improvement
        ls_elite_size   : number of top elites receiving local search per cycle
        ls_final        : run final local search on best solution before return

    Returns:
        best_chromosome (list)
        best_routes     (list)
        best_distance   (float): competition objective (1000*NV + TD)
        history         (list): objective per generation
    """
    num_customers = len(coords) - 1

    # -----------------------------------------------------------------------
    # 1. Initialise population with diverse Clark-Wright + NN seeding
    # -----------------------------------------------------------------------
    population = initialize_population(
        pop_size, num_customers,
        coords=coords, demands=demands,
        capacity=capacity, distance_matrix=distance_matrix,
    )
    scores = population_fitness(population, demands, capacity, distance_matrix)

    best_idx        = int(np.argmin(scores))
    best_chromosome = population[best_idx][:]
    best_distance   = scores[best_idx]
    history         = [best_distance]
    no_improve      = 0

    if verbose:
        print(f"  Gen 0 | Obj: {best_distance:.2f}")

    # -----------------------------------------------------------------------
    # 2. Main GA loop
    # -----------------------------------------------------------------------
    for gen in range(1, num_generations + 1):

        # -- Sort by fitness ------------------------------------------------
        sorted_idx = sorted(range(len(population)), key=lambda i: scores[i])

        # -- Local search on top elites every 10 generations ---------------
        ls_improved = []
        if gen % 10 == 0:
            for rank, idx in enumerate(sorted_idx[:ls_elite_size]):
                chrom      = population[idx][:]
                routes     = decode_chromosome(chrom, demands, capacity)
                opt_routes, opt_obj = local_search(
                    routes, distance_matrix, demands, capacity)
                opt_chrom  = routes_to_chromosome(opt_routes)
                ls_improved.append((opt_chrom, opt_obj))

        # -- Build elite pool -----------------------------------------------
        elites = [chrom for chrom, _ in ls_improved]
        for idx in sorted_idx[len(ls_improved):elite_size]:
            elites.append(population[idx][:])

        # -- Update scores for LS-improved elites ---------------------------
        for rank, (opt_chrom, opt_obj) in enumerate(ls_improved):
            orig_idx             = sorted_idx[rank]
            population[orig_idx] = opt_chrom
            scores[orig_idx]     = opt_obj

        # -- Fill rest with offspring ---------------------------------------
        new_population = elites[:]
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, scores, tournament_size)
            parent2 = tournament_selection(population, scores, tournament_size)

            if random.random() < crossover_rate:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = combined_mutation(child1, mutation_rate)
            child2 = combined_mutation(child2, mutation_rate)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        # -- Re-evaluate ----------------------------------------------------
        population = new_population
        scores     = population_fitness(
            population, demands, capacity, distance_matrix)

        gen_best_idx = int(np.argmin(scores))
        gen_best_obj = scores[gen_best_idx]
        history.append(gen_best_obj)

        if gen_best_obj < best_distance:
            best_distance   = gen_best_obj
            best_chromosome = population[gen_best_idx][:]
            no_improve      = 0
        else:
            no_improve += 1

        if verbose and gen % 50 == 0:
            print(f"  Gen {gen:>4} | Obj: {best_distance:.2f}")

        if no_improve >= patience:
            if verbose:
                print(f"\n  Early stopping at gen {gen}.")
            break

    # -----------------------------------------------------------------------
    # 3. Final local search pass
    # -----------------------------------------------------------------------
    best_routes = decode_chromosome(best_chromosome, demands, capacity)

    if ls_final:
        if verbose:
            print(f"\n  Running final local search...")
        best_routes, final_obj = local_search(
            best_routes, distance_matrix, demands, capacity)
        if final_obj < best_distance:
            if verbose:
                print(f"  Final LS: {best_distance:.2f} → {final_obj:.2f}")
            best_distance   = final_obj
            best_chromosome = routes_to_chromosome(best_routes)

    return best_chromosome, best_routes, best_distance, history
