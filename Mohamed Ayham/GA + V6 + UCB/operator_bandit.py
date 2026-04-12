import math


class OperatorBandit:
    """
    UCB1 Multi-Armed Bandit for Adaptive Operator Selection.

    Tracks how much fitness improvement each mutation operator has produced
    across all offspring in the current run, and uses the Upper Confidence
    Bound (UCB1) formula to decide which operator to apply next.

    UCB1 balances:
      - Exploitation : prefer operators with the highest average reward so far
      - Exploration  : still try operators that haven't been used much,
                       in case they become useful as the search progresses

    Reward signal:
      reward = max(0, before_obj - after_obj) / before_obj
      (normalised improvement — zero if the operator made things worse or equal)

    This means operators that consistently improve offspring are selected more,
    while operators that stopped helping are gradually deprioritised — but never
    completely abandoned, thanks to the exploration bonus.

    Usage:
        bandit = OperatorBandit(["swap", "inversion", "insertion", "scramble"])
        op = bandit.select()
        child = MUTATION_FNS[op](child, mutation_rate)
        bandit.update(op, before_obj, after_obj)
    """

    def __init__(self, operators):
        """
        Args:
            operators (list[str]): names of the mutation operators to manage.
        """
        self.operators = operators
        # Initialise counts to 1 to avoid division-by-zero on first select
        self.counts    = {op: 1   for op in operators}
        self.rewards   = {op: 0.0 for op in operators}
        self.t         = len(operators)   # total pulls so far

    def select(self):
        """
        UCB1 selection rule.

        score(op) = avg_reward(op) + sqrt(2 * ln(t) / count(op))

        The sqrt term shrinks as an operator is used more, so under-used
        operators get a temporary boost that encourages trying them.

        Returns:
            op (str): name of the selected operator
        """
        ucb_scores = {}
        for op in self.operators:
            avg     = self.rewards[op] / self.counts[op]
            explore = math.sqrt(2.0 * math.log(self.t) / self.counts[op])
            ucb_scores[op] = avg + explore
        return max(ucb_scores, key=ucb_scores.get)

    def update(self, operator, before_obj, after_obj):
        """
        Record the outcome of applying `operator` to one offspring.

        Args:
            operator  (str)  : which operator was applied
            before_obj (float): fitness before mutation
            after_obj  (float): fitness after mutation
        """
        improvement          = max(0.0, before_obj - after_obj)
        reward               = improvement / (before_obj + 1e-10)
        self.counts[operator]  += 1
        self.rewards[operator] += reward
        self.t                 += 1

    def probabilities(self):
        """
        Return normalised selection weights for logging.
        These are proportional to each operator's average reward,
        NOT the UCB scores (so they're easier to interpret as percentages).

        Returns:
            dict[str, float]: operator → weight (sums to 1.0)
        """
        avgs  = {op: self.rewards[op] / self.counts[op] for op in self.operators}
        total = sum(avgs.values())
        if total < 1e-10:
            # No operator has earned reward yet — show uniform weights
            return {op: 1.0 / len(self.operators) for op in self.operators}
        return {op: avgs[op] / total for op in self.operators}

    def summary(self):
        """
        One-line string of operator weights for printing in verbose mode.
        Example: 'swap:0.12 | inversion:0.29 | insertion:0.44 | scramble:0.15'
        """
        probs = self.probabilities()
        return " | ".join(f"{op}:{v:.2f}" for op, v in probs.items())
