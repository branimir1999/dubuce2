from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Collection

import tensorflow as tf

from tf_utils import TFData

@dataclass
class Dist(ABC):
    dim: int

    def __default_pdf(self, sample: TFData = None) -> TFData:
        return tf.exp(-self.score(sample))

    def __default_score(self, sample: TFData = None) -> TFData:
        epsilon = 1e-37
        return -tf.math.log(tf.maximum(self.pdf(sample), epsilon))

    def __post_init__(self):
        is_overriden = lambda attr: getattr(Dist, attr) != getattr(type(self), attr)

        if not (is_overriden("pdf") or is_overriden("score")):
            raise TypeError("At least one of `pdf` or `score` must be implemented.")

        if not is_overriden("pdf"):
            setattr(self, "pdf", self.__default_pdf)

        if not is_overriden("score"):
            setattr(self, "score", self.__default_score)

    @property
    @abstractmethod
    def params(self) -> Collection[str]:
        pass

    @property
    @abstractmethod
    def param_shapes(self) -> Collection[int]:
        pass

    def update_params(self, *args) -> None:
        for param, value in zip(self.params, args):
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Parameter `{param}` of `{type(self)}` not found.")

    @abstractmethod
    def sample(self, eps: TFData = None) -> TFData:
        pass

    def score(self, sample: TFData) -> TFData:
        return NotImplemented

    def pdf(self, sample: TFData) -> TFData:
        pass

@dataclass
class Normal(Dist):
    mean: TFData = field(default=None, init=False)
    logvar: TFData = field(default=None, init=False)

    @property
    def params(self) -> Collection[str]:
        return "mean", "logvar"

    @property
    def param_shapes(self) -> Collection[int]:
        return self.dim, self.dim

    def sample(self, eps: TFData = None) -> TFData:
        if eps is None:
            eps = tf.random.normal(shape=tf.shape(self.mean))

        stddev = tf.exp(self.logvar / 2)

        return eps * stddev + self.mean

    def score(self, sample: TFData) -> TFData:
        var = tf.exp(self.logvar)
        score = 0.5 * ((sample - self.mean) ** 2 / var + self.logvar + tf.math.log(2 * math.pi))
        score = tf.reduce_sum(score, axis=-1)

        return score

@dataclass
class Binary(Dist):
    logpi: TFData = field(default=None, init=False)

    @property
    def params(self) -> Collection[str]:
        return "logpi"

    @property
    def param_shapes(self) -> Collection[int]:
        return 1,

    def sample(self, eps: TFData = None) -> TFData:
        if eps is None:
            eps = tf.random.uniform(shape=tf.shape(self.logpi))

        pi = tf.nn.sigmoid(self.logpi)

        return tf.where(eps < pi, 1, 0)

    def score(self, sample: TFData) -> TFData:
        return tf.math.log1p(tf.exp(tf.where(sample == 1, -self.logpi, self.logpi)))
