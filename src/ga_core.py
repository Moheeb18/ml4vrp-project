import random
import numpy as np

from ga_representation import initialize_population, decode_chromosome
from ga_fitness import population_fitness, fitness
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
    patience=80,
    ls_elite_size=5,
    ls_final=True,
):
    """
    Run the Genetic Algorithm with local search (2-opt + OR-opt) to solve CVRP.

    Local search strategy:
      - Every generation, the top `ls_elite_size` chromosomes are decoded,
        locally optimised, and re-encoded before being passed to the next gen.
        This is the "memetic" approach — it consistently closes 10-15% of the
        gap to optimal vs a plain GA.
      - At the end, a final full local search is run on the best solution found.

    Args:
        coords          : np.array of node coordinates (row 0 = depot)
        demands         : np.array of demands (index 0 = depot, demand = 0)
        capacity        : int, vehicle capacity
        distance_matrix : 2D np.array of pairwise distances
        pop_size        : population size
        num_generations : maximum number of generations
        elite_size      : number of elite individuals carried forward unchanged
        tournament_size : competitors per tournament in selection
        crossover_rate  : probability of OX crossover
        mutation_rate   : probability of each mutation operator firing
        verbose         : print progress every 50 generations
        patience        : early stopping if no improvement for this many gens
        ls_elite_size   : how many of the top elites get 2-opt+OR-opt per gen
        ls_final        : whether to run a final local search pass at the end

    Returns:
        best_chromosome (list)  : the best permutation found
        best_routes     (list)  : decoded + locally optimised routes
        best_distance   (float) : total distance of the best solution
        history         (list)  : best fitness value per generation
    """
    num_customers = len(coords) - 1

    # -----------------------------------------------------------------------
    # 1. Initialise population
    # -----------------------------------------------------------------------
    population = initialize_population(pop_size, num_customers)
    scores     = population_fitness(population, demands, capacity, distance_matrix)

    best_idx        = int(np.argmin(scores))
    best_chromosome = population[best_idx][:]
    best_distance   = scores[best_idx]
    history         = [best_distance]
    no_improve      = 0

    if verbose:
        print(f"Gen 0 | Best distance: {best_distance:.2f}")

    # -----------------------------------------------------------------------
    # 2. Evolve
    # -----------------------------------------------------------------------
    for gen in range(1, num_generations + 1):

        # -- Sort population by fitness --------------------------------------
        sorted_idx = sorted(range(len(population)), key=lambda i: scores[i])

        # -- Apply local search to the top ls_elite_size chromosomes --------
        ls_improved = []
        for rank, idx in enumerate(sorted_idx[:ls_elite_size]):
            chrom      = population[idx][:]
            routes     = decode_chromosome(chrom, demands, capacity)
            opt_routes, opt_dist = local_search(routes, distance_matrix)
            opt_chrom  = routes_to_chromosome(opt_routes)
            ls_improved.append((opt_chrom, opt_dist))

        # -- Build elite pool: LS-improved elites first, then plain elites --
        elites = [chrom for chrom, _ in ls_improved]
        for idx in sorted_idx[ls_elite_size:elite_size]:
            elites.append(population[idx][:])

        # -- Update scores for LS-improved elites in the population ---------
        for rank, (opt_chrom, opt_dist) in enumerate(ls_improved):
            original_idx            = sorted_idx[rank]
            population[original_idx] = opt_chrom
            scores[original_idx]     = opt_dist

        # -- Fill rest of population with offspring --------------------------
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

        # -- Re-evaluate full population ------------------------------------
        population = new_population
        scores     = population_fitness(population, demands, capacity, distance_matrix)

        gen_best_idx  = int(np.argmin(scores))
        gen_best_dist = scores[gen_best_idx]
        history.append(gen_best_dist)

        # -- Track global best ----------------------------------------------
        if gen_best_dist < best_distance:
            best_distance   = gen_best_dist
            best_chromosome = population[gen_best_idx][:]
            no_improve      = 0
        else:
            no_improve += 1

        if verbose and gen % 50 == 0:
            print(f"Gen {gen:>4} | Best distance: {best_distance:.2f}")

        # -- Early stopping -------------------------------------------------
        if no_improve >= patience:
            if verbose:
                print(f"\nEarly stopping at generation {gen} "
                      f"(no improvement for {patience} generations).")
            break

    # -----------------------------------------------------------------------
    # 3. Final local search pass on the overall best solution
    # -----------------------------------------------------------------------
    best_routes = decode_chromosome(best_chromosome, demands, capacity)

    if ls_final:
        if verbose:
            print(f"\nRunning final local search on best solution...")
        best_routes, final_dist = local_search(best_routes, distance_matrix)
        if final_dist < best_distance:
            if verbose:
                print(f"Final local search improved: {best_distance:.2f} -> {final_dist:.2f}")
            best_distance   = final_dist
            best_chromosome = routes_to_chromosome(best_routes)

    return best_chromosome, best_routes, best_distance, history
