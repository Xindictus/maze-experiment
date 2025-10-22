import math


class EpsilonDecayRate:
    def __init__(self, eps_max=1.0, eps_min=0.01, T=1000):
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.T = T

    def linear(self, t):
        return max(
            self.eps_min,
            self.eps_max - (self.eps_max - self.eps_min) * (t / self.T),
        )

    def exp_half_life(self, t, H=150):
        return self.eps_min + (self.eps_max - self.eps_min) * (0.5 ** (t / H))

    def polynomial(self, t, p=2.0):
        x = max(0.0, 1.0 - t / self.T)
        return self.eps_min + (self.eps_max - self.eps_min) * (x**p)

    def inverse_time(self, t, k=200.0, alpha=1.0):
        return self.eps_min + (self.eps_max - self.eps_min) / (
            (1.0 + t / k) ** alpha
        )

    def logistic(self, t, t_mid=None, s=100.0):
        if t_mid is None:
            t_mid = 0.5 * self.T
        return self.eps_min + (self.eps_max - self.eps_min) / (
            1.0 + math.exp((t - t_mid) / s)
        )

    def cosine(self, t):
        x = min(1.0, t / self.T)
        return self.eps_min + 0.5 * (self.eps_max - self.eps_min) * (
            1.0 + math.cos(math.pi * x)
        )
