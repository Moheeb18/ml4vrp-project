import random
import numpy as np

from ga_representation import initialize_population, decode_chromosome
from ga_fitness import population_fitness, fitness_from_routes
from ga_operators import tournament_selection, order_crossover, combined_mutation
from local_search import local_search, routes_to_chromosome
from rl_agent import RLParameterAgent


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
    rl_agent=None,
    q_table_path="q_table.npy",
):
    """
    Memetic GA with RL-adaptive parameters for CVRP.

    Objective: minimise  1000 * num_vehicles + total_distance

    Pipeline:
      - Nearest neighbour seeding (20% of initial population)
      - RL Q-learning agent adapts mutation/crossover each generation
      - 2-opt + OR-opt local search on top elites every 10 generations
      - Inter-route relocation, swap, and route elimination in local search
      - Final local search pass on best solution before returning

    Args:
        coords          : np.array of node coordinates (row 0 = depot)
        demands         : np.array of demands
        capacity        : int, vehicle capacity
        distance_matrix : 2D np.array of pairwise distances
        pop_size        : population size
        num_generations : max generations
        elite_size      : elites carried forward unchanged per generation
        tournament_size : competitors per selection tournament
        crossover_rate  : initial crossover probability (RL overrides each gen)
        mutation_rate   : initial mutation probability (RL overrides each gen)
        verbose         : print progress
        patience        : early stop if no improvement for this many gens
        ls_elite_size   : elites receiving local search every 10 gens
        ls_final        : run final local search after GA ends
        rl_agent        : RLParameterAgent instance (None = create fresh)
        q_table_path    : path to save/load Q-table

    Returns:
        best_chromosome (list)
        best_routes     (list)
        best_distance   (float)
        history         (list)
        rl_agent        (RLParameterAgent)
    """
    num_customers = len(coords) - 1

    # -----------------------------------------------------------------------
    # 1. RL agent setup
    # -----------------------------------------------------------------------
    if rl_agent is None:
        rl_agent = RLParameterAgent(
            alpha=0.1,
            gamma=0.9,
            epsilon=0.3,
            epsilon_decay=0.995,
            epsilon_min=0.05,
        )
        rl_agent.load_q_table(q_table_path)

    # -----------------------------------------------------------------------
    # 2. Initialise population with Clark-Wright + nearest-neighbour seeding
    # -----------------------------------------------------------------------
    population = initialize_population(
        pop_size, num_customers,
        coords=coords, demands=demands, capacity=capacity,
        distance_matrix=distance_matrix
    )
    scores = population_fitness(population, demands, capacity, distance_matrix)

    best_idx        = int(np.argmin(scores))
    best_chromosome = population[best_idx][:]
    best_distance   = scores[best_idx]
    history         = [best_distance]
    no_improve      = 0
    action_log      = []

    if verbose:
        print(f"  Gen 0 | Obj: {best_distance:.2f} | "
              f"mut={mutation_rate:.3f} cross={crossover_rate:.3f}")

    # -----------------------------------------------------------------------
    # 3. Main GA loop
    # -----------------------------------------------------------------------
    for gen in range(1, num_generations + 1):

        # -- RL agent selects parameters ------------------------------------
        mutation_rate, crossover_rate, action_idx = rl_agent.step(
            no_improve=no_improve,
            history=history,
            scores=scores,
            patience=patience,
            current_best=best_distance,
        )
        action_log.append(action_idx)

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
            print(f"  Gen {gen:>4} | Obj: {best_distance:.2f} | "
                  f"mut={mutation_rate:.3f} cross={crossover_rate:.3f} "
                  f"ε={rl_agent.epsilon:.3f}")

        if no_improve >= patience:
            if verbose:
                print(f"\n  Early stopping at gen {gen}.")
            break

    # -----------------------------------------------------------------------
    # 4. Final local search pass
    # -----------------------------------------------------------------------
    best_routes = decode_chromosome(best_chromosome, demands, capacity)

    if ls_final:
        if verbose:
            print(f"\n  Running final local search...")
        best_routes, final_obj = local_search(
            best_routes, distance_matrix, demands, capacity)
        if final_obj < best_distance:
            if verbose:
                print(f"  Local search: {best_distance:.2f} → {final_obj:.2f}")
            best_distance   = final_obj
            best_chromosome = routes_to_chromosome(best_routes)

    # -----------------------------------------------------------------------
    # 5. Save Q-table
    # -----------------------------------------------------------------------
    rl_agent.save_q_table(q_table_path)

    if verbose:
        from collections import Counter
        from rl_agent import ACTIONS
        counts = Counter(action_log)
        print(f"\n  RL Agent — Action usage:")
        for ai, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            mut, cross = ACTIONS[ai]
            pct = 100 * cnt / len(action_log)
            print(f"    Action {ai} (mut={mut:.2f}, cross={cross:.2f})"
                  f" — {cnt} gens ({pct:.1f}%)")

    return best_chromosome, best_routes, best_distance, history, rl_agent
