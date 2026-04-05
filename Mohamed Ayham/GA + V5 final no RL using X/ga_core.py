import random
import numpy as np

from ga_representation import initialize_population, decode_chromosome
from ga_fitness import population_fitness, fitness_from_routes
from ga_operators import tournament_selection, order_crossover, combined_mutation
from local_search import local_search, routes_to_chromosome


def refine_offspring_chromosome(
    chromosome,
    demands,
    capacity,
    distance_matrix,
    apply_refinement=True,
):
    """
    Decode -> local search -> re-encode a child chromosome before insertion.
    """
    child_routes = decode_chromosome(chromosome[:], demands, capacity)

    if not apply_refinement:
        return chromosome[:], fitness_from_routes(child_routes, distance_matrix)

    improved_routes, improved_obj = local_search(
        [route[:] for route in child_routes],
        distance_matrix,
        demands,
        capacity,
    )
    improved_chromosome = routes_to_chromosome(improved_routes)
    return improved_chromosome[:], improved_obj


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
    offspring_ls_rate=0.10,
):
    """
    Memetic Genetic Algorithm for CVRP.

    Objective: minimise 1000 * num_vehicles + total_distance

    Pipeline:
      - Diverse seeding: 5 Clark-Wright variants (40%) + nearest-neighbour
        (20%) + random (40%)
      - Tournament selection, Order Crossover (OX), combined mutation
      - Immediate offspring refinement after crossover/mutation
      - Local search on top elites every 10 generations
      - Elitism
      - Early stopping
      - Final local search pass

    Returns:
        best_chromosome, best_routes, best_distance, history
    """
    num_customers = len(coords) - 1
    elite_size = min(elite_size, pop_size)
    offspring_ls_rate = max(0.0, min(1.0, offspring_ls_rate))

    population = initialize_population(
        pop_size,
        num_customers,
        coords=coords,
        demands=demands,
        capacity=capacity,
        distance_matrix=distance_matrix,
    )
    scores = population_fitness(population, demands, capacity, distance_matrix)

    best_idx = int(np.argmin(scores))
    best_chromosome = population[best_idx][:]
    best_distance = scores[best_idx]
    history = [best_distance]
    no_improve = 0

    if verbose:
        print(f"  Gen 0 | Obj: {best_distance:.2f}")

    for gen in range(1, num_generations + 1):
        sorted_idx = sorted(range(len(population)), key=lambda i: scores[i])

        if gen % 10 == 0:
            elite_ls_count = min(ls_elite_size, len(sorted_idx))
            for idx in sorted_idx[:elite_ls_count]:
                elite_routes = decode_chromosome(population[idx][:], demands, capacity)
                opt_routes, opt_obj = local_search(
                    [route[:] for route in elite_routes],
                    distance_matrix,
                    demands,
                    capacity,
                )
                population[idx] = routes_to_chromosome(opt_routes)
                scores[idx] = opt_obj

            sorted_idx = sorted(range(len(population)), key=lambda i: scores[i])

        elites = [population[idx][:] for idx in sorted_idx[:elite_size]]
        elite_scores = [scores[idx] for idx in sorted_idx[:elite_size]]

        new_population = [elite[:] for elite in elites]
        new_scores = elite_scores[:]

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, scores, tournament_size)
            parent2 = tournament_selection(population, scores, tournament_size)

            if random.random() < crossover_rate:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = combined_mutation(child1, mutation_rate)
            child2 = combined_mutation(child2, mutation_rate)

            child1, child1_obj = refine_offspring_chromosome(
                child1,
                demands,
                capacity,
                distance_matrix,
                apply_refinement=(random.random() < offspring_ls_rate),
            )
            new_population.append(child1[:])
            new_scores.append(child1_obj)

            if len(new_population) < pop_size:
                child2, child2_obj = refine_offspring_chromosome(
                    child2,
                    demands,
                    capacity,
                    distance_matrix,
                    apply_refinement=(random.random() < offspring_ls_rate),
                )
                new_population.append(child2[:])
                new_scores.append(child2_obj)

        population = new_population
        scores = new_scores

        gen_best_idx = int(np.argmin(scores))
        gen_best_obj = scores[gen_best_idx]
        history.append(gen_best_obj)

        if gen_best_obj < best_distance:
            best_distance = gen_best_obj
            best_chromosome = population[gen_best_idx][:]
            no_improve = 0
        else:
            no_improve += 1

        if verbose and gen % 50 == 0:
            print(f"  Gen {gen:>4} | Obj: {best_distance:.2f}")

        if no_improve >= patience:
            if verbose:
                print(f"\n  Early stopping at gen {gen}.")
            break

    best_routes = decode_chromosome(best_chromosome, demands, capacity)

    if ls_final:
        if verbose:
            print("\n  Running final local search...")
        best_routes, final_obj = local_search(
            [route[:] for route in best_routes],
            distance_matrix,
            demands,
            capacity,
        )
        if final_obj < best_distance:
            if verbose:
                print(f"  Final LS: {best_distance:.2f} -> {final_obj:.2f}")
            best_distance = final_obj
            best_chromosome = routes_to_chromosome(best_routes)

    return best_chromosome, best_routes, best_distance, history
