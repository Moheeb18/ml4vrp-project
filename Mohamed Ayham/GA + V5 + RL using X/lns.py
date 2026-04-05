import random
import numpy as np
from routes_utilities import route_distance
from ga_fitness import fitness_from_routes, VEHICLE_PENALTY
from local_search import local_search, routes_to_chromosome


# ---------------------------------------------------------------------------
# DESTROY OPERATORS
# Each operator removes a subset of customers from the current solution,
# returning (remaining_routes, removed_customers).
# ---------------------------------------------------------------------------

def random_destroy(routes, demands, capacity, num_remove):
    """
    Random destroy: remove `num_remove` customers chosen uniformly at random.
    Simple but effective as a baseline destroy operator.
    """
    all_customers = [c for route in routes for c in route if c != 0]
    if num_remove >= len(all_customers):
        num_remove = max(1, len(all_customers) // 2)

    removed  = random.sample(all_customers, num_remove)
    removed_set = set(removed)

    new_routes = []
    for route in routes:
        new_route = [0] + [c for c in route if c not in removed_set] + [0]
        if len(new_route) > 2:
            new_routes.append(new_route)

    return new_routes, removed


def worst_removal(routes, distance_matrix, num_remove):
    """
    Worst removal: remove the `num_remove` customers that contribute most
    to the total distance (highest removal savings).

    Removing an expensive customer opens up the most room for improvement
    during repair, making this a powerful destroy operator.
    """
    savings = {}
    for route in routes:
        for idx in range(1, len(route) - 1):
            c    = route[idx]
            prev = route[idx - 1]
            next = route[idx + 1]
            # Saving = edges removed - edge added when closing the gap
            saving = (distance_matrix[prev][c]
                      + distance_matrix[c][next]
                      - distance_matrix[prev][next])
            savings[c] = saving

    # Sort by saving descending, take top num_remove
    removed     = sorted(savings, key=lambda c: -savings[c])[:num_remove]
    removed_set = set(removed)

    new_routes = []
    for route in routes:
        new_route = [0] + [c for c in route if c not in removed_set] + [0]
        if len(new_route) > 2:
            new_routes.append(new_route)

    return new_routes, removed


def shaw_removal(routes, distance_matrix, demands, num_remove):
    """
    Shaw removal: remove customers that are similar to each other
    (close in distance and similar in demand).

    Removing similar customers concentrates the disruption in one region
    of the solution, making it easier for the repair operator to find a
    good reinsertion arrangement.
    """
    all_customers = [c for route in routes for c in route if c != 0]
    if not all_customers:
        return routes, []

    # Start with a random customer
    seed     = random.choice(all_customers)
    removed  = [seed]

    while len(removed) < num_remove and len(removed) < len(all_customers):
        # Pick a random already-removed customer as reference
        ref = random.choice(removed)

        # Score remaining customers by similarity to ref
        candidates = [c for c in all_customers if c not in removed]
        if not candidates:
            break

        def similarity(c):
            dist_score   = distance_matrix[ref][c]
            demand_score = abs(demands[ref] - demands[c])
            return dist_score + demand_score * 0.1

        candidates.sort(key=similarity)
        # Pick from the most similar with some randomness
        pick_idx = int(random.random() ** 2 * len(candidates))
        removed.append(candidates[pick_idx])

    removed_set = set(removed)
    new_routes  = []
    for route in routes:
        new_route = [0] + [c for c in route if c not in removed_set] + [0]
        if len(new_route) > 2:
            new_routes.append(new_route)

    return new_routes, removed


# ---------------------------------------------------------------------------
# REPAIR OPERATORS
# Each operator reinserts all removed customers back into the routes.
# ---------------------------------------------------------------------------

def greedy_repair(routes, removed, demands, capacity, distance_matrix):
    """
    Greedy repair: insert each removed customer at its cheapest feasible
    position across all existing routes.

    If no feasible position exists, open a new route for that customer.

    Args:
        routes          : partial routes (after destroy)
        removed         : list of customers to reinsert
        demands         : np.array
        capacity        : int
        distance_matrix : 2D np.array

    Returns:
        routes (list): complete feasible solution
    """
    working = [r[:] for r in routes]

    # Shuffle removed to avoid always inserting in the same order
    to_insert = removed[:]
    random.shuffle(to_insert)

    for customer in to_insert:
        best_cost = float("inf")
        best_r    = -1
        best_pos  = -1

        for ri, route in enumerate(working):
            load = sum(demands[c] for c in route if c != 0)
            if load + demands[customer] > capacity:
                continue

            for pos in range(1, len(route)):
                prev = route[pos - 1]
                next = route[pos]
                cost = (distance_matrix[prev][customer]
                        + distance_matrix[customer][next]
                        - distance_matrix[prev][next])
                if cost < best_cost:
                    best_cost = cost
                    best_r    = ri
                    best_pos  = pos

        if best_r != -1:
            working[best_r] = (working[best_r][:best_pos]
                               + [customer]
                               + working[best_r][best_pos:])
        else:
            # No room in existing routes — open a new one
            working.append([0, customer, 0])

    return working


def regret_repair(routes, removed, demands, capacity, distance_matrix, k=2):
    """
    Regret-k repair: prioritise inserting customers with the highest
    "regret" — the difference between their best and k-th best insertion
    cost.

    A customer with high regret will suffer significantly if it is not
    inserted at its best position now, so it gets priority. This tends to
    produce better quality repairs than greedy insertion.

    Args:
        routes          : partial routes
        removed         : customers to reinsert
        demands         : np.array
        capacity        : int
        distance_matrix : 2D np.array
        k               : regret level (default 2)

    Returns:
        routes (list): complete feasible solution
    """
    working   = [r[:] for r in routes]
    to_insert = set(removed)

    while to_insert:
        regrets = {}

        for customer in to_insert:
            insertion_costs = []

            for ri, route in enumerate(working):
                load = sum(demands[c] for c in route if c != 0)
                if load + demands[customer] > capacity:
                    continue

                for pos in range(1, len(route)):
                    prev = route[pos - 1]
                    nxt  = route[pos]
                    cost = (distance_matrix[prev][customer]
                            + distance_matrix[customer][nxt]
                            - distance_matrix[prev][nxt])
                    insertion_costs.append((cost, ri, pos))

            insertion_costs.sort(key=lambda x: x[0])

            if not insertion_costs:
                # Must open new route
                regrets[customer] = (float("inf"), -1, -1, -1)
            elif len(insertion_costs) < k:
                best_cost, best_r, best_pos = insertion_costs[0]
                regret_val = best_cost  # only one option
                regrets[customer] = (regret_val, best_cost, best_r, best_pos)
            else:
                best_cost, best_r, best_pos = insertion_costs[0]
                kth_cost = insertion_costs[k - 1][0]
                regret_val = kth_cost - best_cost
                regrets[customer] = (regret_val, best_cost, best_r, best_pos)

        # Insert the customer with the highest regret
        best_cust = max(regrets, key=lambda c: regrets[c][0])
        regret_val, best_cost, best_r, best_pos = regrets[best_cust]

        if best_r == -1:
            working.append([0, best_cust, 0])
        else:
            working[best_r] = (working[best_r][:best_pos]
                               + [best_cust]
                               + working[best_r][best_pos:])

        to_insert.remove(best_cust)

    return working


# ---------------------------------------------------------------------------
# ADAPTIVE LARGE NEIGHBOURHOOD SEARCH
# ---------------------------------------------------------------------------

class AdaptiveLNS:
    """
    Adaptive Large Neighbourhood Search (ALNS) for CVRP.

    Maintains three destroy operators and two repair operators.
    A weight vector tracks how well each operator has been performing
    and influences how often it gets selected — better operators are
    chosen more frequently.

    The destroy size (how many customers to remove) starts small and
    grows if the search stagnates, making the perturbation progressively
    more aggressive when stuck.

    Args:
        demands         : np.array of customer demands
        capacity        : int, vehicle capacity
        distance_matrix : 2D np.array of pairwise distances
        max_iter        : maximum LNS iterations per call
        min_remove_pct  : minimum fraction of customers to remove
        max_remove_pct  : maximum fraction of customers to remove
        reaction_factor : how quickly weights adapt (0=no adapt, 1=instant)
        local_search_fn : optional local search function to apply after repair
    """

    # Scoring for weight updates
    SCORE_NEW_BEST    = 10   # found a new global best
    SCORE_IMPROVED    = 5    # improved on current solution
    SCORE_ACCEPTED    = 1    # accepted (e.g. simulated annealing)
    SCORE_REJECTED    = 0    # rejected

    def __init__(
        self,
        demands,
        capacity,
        distance_matrix,
        max_iter=150,
        min_remove_pct=0.10,
        max_remove_pct=0.35,
        reaction_factor=0.15,
    ):
        self.demands         = demands
        self.capacity        = capacity
        self.distance_matrix = distance_matrix
        self.max_iter        = max_iter
        self.min_remove_pct  = min_remove_pct
        self.max_remove_pct  = max_remove_pct
        self.reaction_factor = reaction_factor

        # Destroy operators and their adaptive weights
        self.destroy_ops = [
            ("random",  self._destroy_random),
            ("worst",   self._destroy_worst),
            ("shaw",    self._destroy_shaw),
        ]
        self.destroy_weights = [1.0, 1.0, 1.0]

        # Repair operators and their adaptive weights
        self.repair_ops = [
            ("greedy",  self._repair_greedy),
            ("regret",  self._repair_regret),
        ]
        self.repair_weights = [1.0, 1.0]

        # Usage and score tracking
        self.destroy_scores = [0.0, 0.0, 0.0]
        self.destroy_uses   = [0, 0, 0]
        self.repair_scores  = [0.0, 0.0]
        self.repair_uses    = [0, 0]

    # -- Destroy wrappers ---------------------------------------------------

    def _destroy_random(self, routes, n):
        return random_destroy(routes, self.demands, self.capacity, n)

    def _destroy_worst(self, routes, n):
        return worst_removal(routes, self.distance_matrix, n)

    def _destroy_shaw(self, routes, n):
        return shaw_removal(routes, self.distance_matrix, self.demands, n)

    # -- Repair wrappers ----------------------------------------------------

    def _repair_greedy(self, routes, removed):
        return greedy_repair(
            routes, removed, self.demands, self.capacity, self.distance_matrix)

    def _repair_regret(self, routes, removed):
        return regret_repair(
            routes, removed, self.demands, self.capacity, self.distance_matrix)

    # -- Operator selection -------------------------------------------------

    def _select_operator(self, weights):
        """Roulette-wheel selection based on current weights."""
        total  = sum(weights)
        probs  = [w / total for w in weights]
        cumsum = 0.0
        r      = random.random()
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return i
        return len(weights) - 1

    # -- Weight update ------------------------------------------------------

    def _update_weight(self, weights, scores, uses, idx, score):
        """
        Update the weight of operator `idx` using exponential smoothing.
        operators that score well get higher weights over time.
        """
        uses[idx]   += 1
        scores[idx] += score
        if uses[idx] > 0:
            avg_score   = scores[idx] / uses[idx]
            weights[idx] = (1 - self.reaction_factor) * weights[idx] \
                           + self.reaction_factor * (avg_score + 1)

    # -- Main run -----------------------------------------------------------

    def run(self, initial_routes, apply_local_search=True, verbose=False):
        """
        Run ALNS starting from `initial_routes`.

        Uses simulated annealing acceptance: slightly worse solutions can
        be accepted early on, with acceptance probability decreasing over
        iterations. This helps escape local optima.

        Args:
            initial_routes     : starting solution (list of routes)
            apply_local_search : run local search after each repair
            verbose            : print iteration progress

        Returns:
            best_routes (list) : best solution found
            best_obj    (float): competition objective of best solution
        """
        current_routes = [r[:] for r in initial_routes]
        current_obj    = fitness_from_routes(current_routes, self.distance_matrix)
        best_routes    = [r[:] for r in current_routes]
        best_obj       = current_obj

        num_customers  = sum(len(r) - 2 for r in current_routes)

        # Simulated annealing parameters
        # Start temperature set so ~50% of 1% worse solutions are accepted
        start_temp   = current_obj * 0.01 / (-np.log(0.5) + 1e-10)
        end_temp     = start_temp * 0.001
        cooling      = (end_temp / max(start_temp, 1e-10)) ** (1.0 / max(self.max_iter, 1))
        temperature  = start_temp

        no_improve   = 0

        for it in range(self.max_iter):

            # Adaptive destroy size — grows when stuck
            stag_factor  = min(1.0, no_improve / 30)
            remove_pct   = (self.min_remove_pct
                            + stag_factor * (self.max_remove_pct - self.min_remove_pct))
            num_remove   = max(1, int(num_customers * remove_pct))

            # Select and apply destroy operator
            d_idx        = self._select_operator(self.destroy_weights)
            destroy_fn   = self.destroy_ops[d_idx][1]
            partial, removed = destroy_fn(current_routes, num_remove)

            if not removed:
                continue

            # Select and apply repair operator
            r_idx        = self._select_operator(self.repair_weights)
            repair_fn    = self.repair_ops[r_idx][1]
            new_routes   = repair_fn(partial, removed)

            # Optional local search polish after repair
            if apply_local_search:
                new_routes, new_obj = local_search(
                    new_routes, self.distance_matrix,
                    self.demands, self.capacity)
            else:
                new_obj = fitness_from_routes(new_routes, self.distance_matrix)

            # Score the operators
            score = self.SCORE_REJECTED

            if new_obj < best_obj - 1e-10:
                # New global best
                best_routes = [r[:] for r in new_routes]
                best_obj    = new_obj
                score       = self.SCORE_NEW_BEST
                no_improve  = 0
                if verbose:
                    print(f"    LNS iter {it:>4} | New best: {best_obj:.2f} "
                          f"| destroy={self.destroy_ops[d_idx][0]} "
                          f"repair={self.repair_ops[r_idx][0]}")

            elif new_obj < current_obj - 1e-10:
                # Improved on current (not global best)
                score      = self.SCORE_IMPROVED
                no_improve += 1

            else:
                # Simulated annealing acceptance
                delta = new_obj - current_obj
                if temperature > 1e-10 and random.random() < np.exp(-delta / temperature):
                    score      = self.SCORE_ACCEPTED
                    no_improve += 1
                else:
                    no_improve += 1

            # Accept or reject
            if score > self.SCORE_REJECTED:
                current_routes = new_routes
                current_obj    = new_obj

            # Update operator weights
            self._update_weight(
                self.destroy_weights, self.destroy_scores,
                self.destroy_uses, d_idx, score)
            self._update_weight(
                self.repair_weights, self.repair_scores,
                self.repair_uses, r_idx, score)

            # Cool temperature
            temperature *= cooling

        return best_routes, best_obj

    def operator_stats(self):
        """Print a summary of which operators were most effective."""
        print("\n  ALNS Operator Stats:")
        print(f"  {'Operator':<20} {'Uses':>6} {'Avg Score':>10}")
        for i, (name, _) in enumerate(self.destroy_ops):
            uses = self.destroy_uses[i]
            avg  = self.destroy_scores[i] / uses if uses > 0 else 0
            print(f"  destroy-{name:<12} {uses:>6} {avg:>10.3f}")
        for i, (name, _) in enumerate(self.repair_ops):
            uses = self.repair_uses[i]
            avg  = self.repair_scores[i] / uses if uses > 0 else 0
            print(f"  repair-{name:<13} {uses:>6} {avg:>10.3f}")


def run_lns(routes, demands, capacity, distance_matrix,
            max_iter=150, verbose=False):
    """
    Convenience function: create an AdaptiveLNS and run it.

    Args:
        routes          : starting solution
        demands         : np.array
        capacity        : int
        distance_matrix : 2D np.array
        max_iter        : LNS iterations
        verbose         : print progress

    Returns:
        best_routes (list)
        best_obj    (float)
    """
    alns = AdaptiveLNS(
        demands         = demands,
        capacity        = capacity,
        distance_matrix = distance_matrix,
        max_iter        = max_iter,
    )
    return alns.run(routes, apply_local_search=True, verbose=verbose)
