from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

import numpy as np

NumT = Union[np.ndarray, float]

@dataclass
class MixtureDist(ABC):
    pi: NumT
    K: int = field(init=False)

    def __post_init__(self):
        self.K = len(self.pi)

    @abstractmethod
    def sample(self, L: int) -> NumT:
        raise NotImplementedError

    @abstractmethod
    def p_xz(self, x: NumT, k: int) -> NumT:
        raise NotImplementedError

    def p_x(self, x: NumT) -> NumT:
        return np.sum([self.pi[k] * self.p_xz(x, k) for k in range(self.K)], axis=0)

@dataclass
class GMDist(MixtureDist):
    mu: NumT
    sigma2: NumT

    @classmethod
    def random(cls, K: int) -> GMDist:
        mu = np.random.normal(size=K).astype(np.float32)
        sigma2 = np.exp(np.random.normal(size=K) - 1).astype(np.float32)
        pi = np.random.dirichlet(5 * np.ones(K)).astype(np.float32)

        return cls(pi, mu, sigma2)

    def sample(self, L: int) -> NumT:
        choices = np.random.choice(np.arange(self.K), size=L, p=self.pi)
        sample = np.random.normal(self.mu[choices], np.sqrt(self.sigma2[choices])).astype(np.float32)

        return sample

    @staticmethod
    def normal_pdf(x: NumT, mu: NumT, sigma2: NumT) -> NumT:
        return np.exp(-0.5 * (x - mu) ** 2 / sigma2) / np.sqrt(2 * np.pi * sigma2)

    def p_xz(self, x: NumT, k: int) -> NumT:
        return self.normal_pdf(x, self.mu[k], self.sigma2[k])
    
    # 𝑝𝜽(𝑧𝑘|𝒙)=(𝜋𝑘 ⋅ 𝑝𝜽(𝒙|𝑧𝑘)) / (∑(𝑝𝜽(𝒙|𝑧𝑖) ⋅ 𝜋𝑖)).
    def p_zx(self, x: NumT, k: int) -> NumT:
        return self.pi[k] * self.p_xz(x, k) / self.p_x(x)

@dataclass
class UMDist(MixtureDist):
    a: NumT
    b: NumT

    @classmethod
    def random(cls, K: int) -> UMDist:
        a = np.random.uniform(low=-10, high=10, size=K).astype(np.float32)
        b = np.random.uniform(low=-10, high=10, size=K).astype(np.float32)
        pi = np.random.uniform(size=K)
        pi = pi / np.sum(pi)
        
        for i in range(K):
            if a[i] >= b[i]:
                temp = a[i]
                a[i] = b[i]
                b[i] = temp
        
        return cls(pi, a, b)

    def sample(self, L: int) -> NumT:
        choices = np.random.choice(np.arange(self.K), size=L, p=self.pi) # Odaberi nasumicno klase
        sample = np.random.uniform(low=self.a[choices], high=self.b[choices], size=L).astype(np.float32) # odaberi nasumicno vrijednosti iz odabrane klase

        return sample

    @staticmethod
    def uniform_pdf(x: NumT, a: NumT, b: NumT) -> NumT:
        return [1 / (b-a) if a <= _x <= b else 0 for _x in x]

    def p_xz(self, x: NumT, k: int) -> NumT:
        return np.array(self.uniform_pdf(x, self.a[k], self.b[k]))
