import math
from typing import Literal


class EpsilonDecayRate:
    def __init__(
        self,
        eps_max: float = 1.0,
        eps_min: float = 0.01,
        T: int = 1000,
        method: Literal[
            "linear",
            "exp_half_life",
            "polynomial",
            "inverse_time",
            "logistic",
            "cosine",
        ] = "linear",
    ):
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.T = T
        self.method = method

    # Linear
    def __linear(self, t) -> float:
        return max(
            self.eps_min,
            self.eps_max - (self.eps_max - self.eps_min) * (t / self.T),
        )

    # Exponential with half-life
    def __exp_half_life(self, t, H=150) -> float:
        return self.eps_min + (self.eps_max - self.eps_min) * (0.5 ** (t / H))

    # Polynomial
    def __polynomial(self, t, p=2.0) -> float:
        x = max(0.0, 1.0 - t / self.T)
        return self.eps_min + (self.eps_max - self.eps_min) * (x**p)

    # Inverse-time
    def __inverse_time(self, t, k=200.0, alpha=1.0) -> float:
        return self.eps_min + (self.eps_max - self.eps_min) / (
            (1.0 + t / k) ** alpha
        )

    # Logistic
    def __logistic(self, t, t_mid=None, s=100.0) -> float:
        if t_mid is None:
            t_mid = 0.5 * self.T
        return self.eps_min + (self.eps_max - self.eps_min) / (
            1.0 + math.exp((t - t_mid) / s)
        )

    # Cosine annealing
    def __cosine(self, t) -> float:
        x = min(1.0, t / self.T)
        return self.eps_min + 0.5 * (self.eps_max - self.eps_min) * (
            1.0 + math.cos(math.pi * x)
        )

    def decay(self, t) -> float:
        match self.method:
            case "linear":
                return self.__linear(t)
            case "exp_half_life":
                return self.__exp_half_life(t)
            case "polynomial":
                return self.__polynomial(t)
            case "inverse_time":
                return self.__inverse_time(t)
            case "logistic":
                return self.__logistic(t)
            case "cosine":
                return self.__cosine(t)
            case _:
                raise ValueError(f"Unknown method: {self.method}")
