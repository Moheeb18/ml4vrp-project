import random
import numpy as np

from ga_representation import initialize_population, decode_chromosome
from ga_fitness import population_fitness, fitness_from_routes, fitness
from ga_operators import tournament_selection, order_crossover, MUTATION_FNS
from local_search import (local_search, local_search_light,
                          local_search_medium, routes_to_chromosome)
from operator_bandit import OperatorBandit


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
    Memetic Genetic Algorithm for CVRP with Adaptive Operator Selection.

    Objective: minimise  1000 * num_vehicles + total_distance

    Key design decisions:
      - Patience is enforced to be at least 3x the LS interval, so the GA
        never stops before LS has had a chance to run even once.
      - Diversity restart at patience//2 stagnation: injects 30% fresh CW
        seeds to escape premature convergence on large instances.
      - Three LS tiers chosen automatically by instance size so route
        merging happens during the loop (not just at the final pass).
      - UCB1 bandit learns which mutation operator works best per instance.
    """
    num_customers = len(coords) - 1
    n = len(distance_matrix)

    # -----------------------------------------------------------------------
    # 1. Size-adaptive LS tier + patience enforcement
    #
    #    CRITICAL: patience must be at least 3 x ls_interval.
    #    If patience < ls_interval the GA stops before LS ever runs,
    #    which is exactly what caused early stopping at gen 20.
    # -----------------------------------------------------------------------
    if n < 80:
        ls_fn       = lambda r, dm, d, c: local_search_light(r, dm, d, c)
        ls_interval = 25
        ls_label    = "Light"
    elif n < 200:
        ls_fn       = lambda r, dm, d, c: local_search_medium(r, dm, d, c, lns_iters=5)
        ls_interval = 30
        ls_label    = "Medium"
    else:
        ls_fn       = lambda r, dm, d, c: local_search_medium(r, dm, d, c, lns_iters=3)
        ls_interval = 40
        ls_label    = "Medium(large)"

    # Enforce patience >= 3 x ls_interval regardless of what main.py passes
    min_patience = 3 * ls_interval
    if patience < min_patience:
        if verbose:
            print(f"  [Warning] patience={patience} raised to {min_patience} "
                  f"(must be >= 3 x ls_interval={ls_interval}).", flush=True)
        patience = min_patience

    # Diversity restart threshold: halfway to patience
    restart_threshold = patience // 2

    # -----------------------------------------------------------------------
    # 2. Initialise population
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
    restarts        = 0

    # -----------------------------------------------------------------------
    # 3. UCB1 bandit for adaptive mutation operator selection
    # -----------------------------------------------------------------------
    bandit = OperatorBandit(list(MUTATION_FNS.keys()))

    if verbose:
        print(f"  LS tier   : {ls_label} every {ls_interval} gens | "
              f"patience={patience}", flush=True)
        print(f"  Gen    0  | Obj: {best_distance:.2f}", flush=True)

    # -----------------------------------------------------------------------
    # 4. Main GA loop
    # -----------------------------------------------------------------------
    for gen in range(1, num_generations + 1):

        # -- Sort by fitness ------------------------------------------------
        sorted_idx = sorted(range(len(population)), key=lambda i: scores[i])

        # -- Diversity restart when stuck halfway to patience ---------------
        #    Replace 30% of the population with fresh diverse seeds so the
        #    GA has new genetic material rather than recombining near-identical
        #    chromosomes. Resets no_improve so the clock restarts.
        if no_improve == restart_threshold:
            restarts += 1
            n_inject = max(5, pop_size * 3 // 10)
            fresh = initialize_population(
                n_inject, num_customers,
                coords=coords, demands=demands,
                capacity=capacity, distance_matrix=distance_matrix,
            )
            worst_idx = sorted_idx[-n_inject:]
            for slot, new_chrom in zip(worst_idx, fresh):
                population[slot] = new_chrom
                scores[slot]     = fitness(
                    new_chrom, demands, capacity, distance_matrix)
            no_improve = 0   # restart the stagnation clock
            if verbose:
                print(f"  [Restart #{restarts}] Injected {n_inject} fresh "
                      f"individuals at gen {gen}", flush=True)

        # -- Size-adaptive LS on top elites ---------------------------------
        ls_improved = []
        if gen % ls_interval == 0:
            print(f"  [{ls_label} LS] gen {gen}...", flush=True)
            for rank, idx in enumerate(sorted_idx[:ls_elite_size]):
                chrom      = population[idx][:]
                routes     = decode_chromosome(chrom, demands, capacity)
                opt_routes, opt_obj = ls_fn(
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

        # -- Offspring generation with bandit-selected mutation -------------
        new_population = elites[:]
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, scores, tournament_size)
            parent2 = tournament_selection(population, scores, tournament_size)

            if random.random() < crossover_rate:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            op     = bandit.select()
            mut_fn = MUTATION_FNS[op]

            before_obj = fitness(child1, demands, capacity, distance_matrix)
            child1     = mut_fn(child1, mutation_rate)
            after_obj  = fitness(child1, demands, capacity, distance_matrix)
            bandit.update(op, before_obj, after_obj)

            child2 = mut_fn(child2, mutation_rate)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        # -- Re-evaluate population -----------------------------------------
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

        # Print at gen 1, then every 50 gens
        if verbose and (gen == 1 or gen % 50 == 0):
            print(f"  Gen {gen:>4} | Obj: {best_distance:.2f} | "
                  f"stag={no_improve}/{patience} | {bandit.summary()}",
                  flush=True)

        if no_improve >= patience:
            if verbose:
                print(f"\n  Early stopping at gen {gen} "
                      f"({restarts} restart(s)).", flush=True)
            break

    # -----------------------------------------------------------------------
    # 5. Final FULL local search (LNS + 2-opt* + inter-route moves)
    # -----------------------------------------------------------------------
    best_routes = decode_chromosome(best_chromosome, demands, capacity)

    if ls_final:
        if verbose:
            print(f"\n  Running final local search...", flush=True)
        best_routes, final_obj = local_search(
            best_routes, distance_matrix, demands, capacity, final=True)
        if final_obj < best_distance:
            if verbose:
                print(f"  Final LS improved: {best_distance:.2f} -> {final_obj:.2f}",
                      flush=True)
            best_distance   = final_obj
            best_chromosome = routes_to_chromosome(best_routes)

    if verbose:
        print(f"\n  Bandit final weights : {bandit.summary()}", flush=True)
        print(f"  Restarts             : {restarts}", flush=True)

    return best_chromosome, best_routes, best_distance, history
