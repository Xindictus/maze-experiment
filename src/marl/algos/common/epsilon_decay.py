class EpsilonDecay:
    def __init__(self, eps0: float, eps_min: float, X: int, p: float):
        # TODO: PYDOC
        self.eps0 = eps0
        self.eps_min = eps_min
        self.X = X
        self.p = p
        # current step
        self.t = 0
        self.epsilon = eps0

    def step(self) -> float:
        # Advance one round and return the new epsilon.
        if self.t >= self.X:
            self.epsilon = self.eps_min
            return self.epsilon

        # compute factor
        num = (
            self.eps_min
            + (self.eps0 - self.eps_min)
            * (1 - (self.t + 1) / self.X) ** self.p
        )
        den = (
            self.eps_min
            + (self.eps0 - self.eps_min) * (1 - self.t / self.X) ** self.p
        )
        d_t = num / den

        # update epsilon
        self.epsilon *= d_t
        self.t += 1
        return self.epsilon


###############
# Example use #
###############
# decay = EpsilonDecay(
#     eps0=self.epsilon,
#     eps_min=0.01,
#     X=(self.max_blocks * self.games_per_block),
#     p=0.5,
# )
#
# epsilon = self.decay.step()
