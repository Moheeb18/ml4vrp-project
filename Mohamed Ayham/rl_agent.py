import numpy as np
import json
import os


# ---------------------------------------------------------------------------
# PARAMETER ACTION SPACE
# Each action is a (mutation_rate, crossover_rate) pair.
# The agent picks one of these each generation based on the GA's state.
# ---------------------------------------------------------------------------

ACTIONS = [
    (0.01, 0.95),   # low mutation,    high crossover  — exploit good solutions
    (0.02, 0.90),   # low mutation,    moderate cross  — balanced exploitation
    (0.03, 0.90),   # moderate mut,    moderate cross  — default balanced
    (0.05, 0.85),   # moderate mut,    moderate cross  — slight exploration
    (0.08, 0.80),   # high mutation,   lower crossover — explore when stuck
    (0.12, 0.75),   # very high mut,   low crossover   — aggressive escape
    (0.03, 0.70),   # moderate mut,    low crossover   — mutation-driven search
    (0.06, 0.95),   # high mutation,   high crossover  — mixed aggressive
]

NUM_ACTIONS = len(ACTIONS)


# ---------------------------------------------------------------------------
# STATE SPACE
# The state is a tuple of three discretised observations:
#   1. stagnation_level : how many generations without improvement (3 bins)
#   2. improvement_rate : recent rate of fitness improvement (3 bins)
#   3. diversity_level  : spread of fitness values in population (3 bins)
# Total: 3 x 3 x 3 = 27 discrete states
# ---------------------------------------------------------------------------

def compute_state(no_improve, history, scores, patience):
    """
    Compute the current RL state from GA observations.

    Args:
        no_improve : int, generations since last improvement
        history    : list of best fitness values per generation
        scores     : list of current population fitness values
        patience   : int, max patience (used to normalise stagnation)

    Returns:
        state (tuple): (stagnation_level, improvement_rate, diversity_level)
                       each in {0, 1, 2}
    """

    # -- 1. Stagnation level -------------------------------------------------
    stag_ratio = no_improve / patience
    if stag_ratio < 0.25:
        stagnation_level = 0    # low stagnation — still improving
    elif stag_ratio < 0.60:
        stagnation_level = 1    # moderate stagnation
    else:
        stagnation_level = 2    # high stagnation — nearly stopping

    # -- 2. Improvement rate -------------------------------------------------
    # Compare average of last 10 gens vs 10 gens before that
    if len(history) >= 20:
        recent   = np.mean(history[-10:])
        previous = np.mean(history[-20:-10])
        if previous > 0:
            improvement = (previous - recent) / previous
        else:
            improvement = 0.0
    elif len(history) >= 2:
        improvement = (history[-2] - history[-1]) / (history[-2] + 1e-10)
    else:
        improvement = 0.0

    if improvement > 0.005:
        improvement_rate = 0    # actively improving
    elif improvement > 0.0001:
        improvement_rate = 1    # slowly improving
    else:
        improvement_rate = 2    # flat / stagnating

    # -- 3. Population diversity ---------------------------------------------
    # Measured as the coefficient of variation of fitness scores
    scores_arr = np.array(scores)
    mean_score = np.mean(scores_arr)
    if mean_score > 0:
        cv = np.std(scores_arr) / mean_score
    else:
        cv = 0.0

    if cv > 0.05:
        diversity_level = 0     # high diversity
    elif cv > 0.01:
        diversity_level = 1     # moderate diversity
    else:
        diversity_level = 2     # low diversity — population converging

    return (stagnation_level, improvement_rate, diversity_level)


# ---------------------------------------------------------------------------
# Q-LEARNING AGENT
# ---------------------------------------------------------------------------

class RLParameterAgent:
    """
    A Q-learning agent that adaptively controls mutation_rate and
    crossover_rate during a GA run.

    The agent maintains a Q-table mapping (state, action) → expected reward.
    Each generation it:
      1. Observes the current GA state
      2. Chooses an action (parameter setting) via epsilon-greedy policy
      3. Applies those parameters to the GA
      4. Receives a reward based on whether the solution improved
      5. Updates the Q-table using the Q-learning update rule

    Q-learning update:
        Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

    Args:
        alpha   : learning rate (how fast Q-values update)
        gamma   : discount factor (how much future rewards matter)
        epsilon : exploration rate (probability of random action)
        epsilon_decay : rate at which epsilon decreases over time
        epsilon_min   : minimum epsilon (always some exploration)
    """

    def __init__(
        self,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.3,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        # Q-table: states are (stag, impr, div) tuples — 27 total
        # Shape: (3, 3, 3, NUM_ACTIONS)
        self.q_table = np.zeros((3, 3, 3, NUM_ACTIONS))

        # Tracking
        self.prev_state  = None
        self.prev_action = None
        self.prev_best   = None

        # Log of (state, action, reward) for analysis
        self.episode_log = []

    # -----------------------------------------------------------------------
    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        With probability epsilon: pick a random action (explore).
        Otherwise: pick the action with the highest Q-value (exploit).
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        else:
            return int(np.argmax(self.q_table[state]))

    # -----------------------------------------------------------------------
    def compute_reward(self, current_best, prev_best):
        """
        Compute the reward signal from the change in best solution quality.

        Positive reward for improvement, negative for stagnation,
        scaled by the magnitude of the improvement.

        Args:
            current_best : best fitness in current generation
            prev_best    : best fitness in previous generation

        Returns:
            reward (float)
        """
        if prev_best is None or prev_best == 0:
            return 0.0

        improvement = (prev_best - current_best) / prev_best

        if improvement > 0.01:
            return 2.0          # big improvement
        elif improvement > 0.001:
            return 1.0          # small improvement
        elif improvement > 0.0:
            return 0.2          # tiny improvement
        else:
            return -0.5         # no improvement

    # -----------------------------------------------------------------------
    def update(self, state, action, reward, next_state):
        """
        Q-learning update rule:
            Q(s,a) ← Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
        """
        current_q  = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q      = current_q + self.alpha * (
                         reward + self.gamma * max_next_q - current_q
                     )
        self.q_table[state][action] = new_q

    # -----------------------------------------------------------------------
    def decay_epsilon(self):
        """Reduce exploration rate after each generation."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # -----------------------------------------------------------------------
    def step(self, no_improve, history, scores, patience, current_best):
        """
        Full agent step called once per GA generation.

        1. Compute current state
        2. If we have a previous state/action, update Q-table with reward
        3. Select new action
        4. Decay epsilon
        5. Return the chosen (mutation_rate, crossover_rate)

        Args:
            no_improve   : generations without improvement
            history      : best fitness history list
            scores       : current population fitness scores
            patience     : GA patience limit
            current_best : best fitness value this generation

        Returns:
            mutation_rate  (float)
            crossover_rate (float)
            action_idx     (int)  — for logging
        """
        state = compute_state(no_improve, history, scores, patience)

        # Update Q-table from previous step
        if self.prev_state is not None:
            reward = self.compute_reward(current_best, self.prev_best)
            self.update(self.prev_state, self.prev_action, reward, state)
            self.episode_log.append((self.prev_state, self.prev_action, reward))

        # Select action for this generation
        action_idx = self.select_action(state)
        mutation_rate, crossover_rate = ACTIONS[action_idx]

        # Decay epsilon
        self.decay_epsilon()

        # Store for next update
        self.prev_state  = state
        self.prev_action = action_idx
        self.prev_best   = current_best

        return mutation_rate, crossover_rate, action_idx

    # -----------------------------------------------------------------------
    def save_q_table(self, path="q_table.npy"):
        """Save the Q-table to disk so learning persists across instances."""
        np.save(path, self.q_table)

    def load_q_table(self, path="q_table.npy"):
        """Load a previously saved Q-table to warm-start the agent."""
        if os.path.exists(path):
            self.q_table = np.load(path)
            print(f"  RL agent: loaded Q-table from {path}")
        else:
            print(f"  RL agent: no saved Q-table found, starting fresh.")

    # -----------------------------------------------------------------------
    def summary(self):
        """Print a summary of what the agent learned."""
        state_labels = {
            0: "low stagnation",
            1: "moderate stagnation",
            2: "high stagnation"
        }
        print("\n  RL Agent — Learned Policy Summary")
        print(f"  {'─'*55}")
        print(f"  {'State (stag/impr/div)':<28} {'Best Action':<20} {'Mut':>6} {'Cross':>6}")
        print(f"  {'─'*55}")

        for s0 in range(3):
            for s1 in range(3):
                for s2 in range(3):
                    state      = (s0, s1, s2)
                    best_action = int(np.argmax(self.q_table[state]))
                    mut, cross  = ACTIONS[best_action]
                    print(f"  ({s0},{s1},{s2})  {'':<23} action {best_action:<13}"
                          f" {mut:>6.2f} {cross:>6.2f}")
        print(f"  {'─'*55}")
        print(f"  Final epsilon: {self.epsilon:.4f}")
        print(f"  Total steps logged: {len(self.episode_log)}")
