import math
from typing import Union, Collection, TypeVar

import tensorflow as tf

TFData = TypeVar('TFData')
def f(x: TFData) -> Union[tf.Tensor, tf.Variable, float]:
    if x:
        return "x"


class GMModel:
    def __init__(self, K):
        self.K = K
        self.mean = tf.Variable(tf.random.normal(shape=[K]))
        self.logvar = tf.Variable(tf.random.normal(shape=[K]))
        self.logpi = tf.Variable(tf.zeros(shape=[K]))
        self.var = self.inverse_log_var()
        self.pi = self.inverse_log_pi()
    
    # 𝝈^2=𝐞𝐱𝐩(𝐥𝐨𝐠(𝝈^2))
    def inverse_log_var(self) -> TFData:
        return tf.math.exp(self.logvar)
    
    # 𝑝𝜽(𝑧𝑘) = 𝜋𝑘 = 𝐬𝐨𝐟𝐭𝐦𝐚𝐱(𝐥𝐨𝐠𝜋𝑘)
    def inverse_log_pi(self) -> TFData:
        return tf.nn.softmax(self.logpi)
    
    def update_inverses(self):
        self.var = self.inverse_log_var()
        self.pi = self.inverse_log_pi()
        
    @property
    def variables(self) -> Collection[TFData]:
        return self.mean, self.logvar, self.logpi

    @staticmethod
    def neglog_normal_pdf(x: TFData, mean: TFData, logvar: TFData):
        var = tf.exp(logvar)
        return 0.5 * (tf.math.log(2 * math.pi) + logvar + (x - mean) ** 2 / var)
    
    def normal_pdf(self, x: TFData, mean: TFData, var: TFData) -> TFData:
        return tf.math.exp(-0.5 * (x - mean) ** 2 / var) / tf.math.sqrt(2 * math.pi * var)

    @tf.function
    def loss(self, data: TFData):
        return self.loss_x(data)
    
    # 𝐿𝜽(𝑥^(𝑖)|𝑧𝑘) = 0.5 ⋅ (log(2𝜋) + log(𝜎^2) + (𝑥−𝜇)^2/𝜎^2) 
    def loss_xz(self, x: TFData, k: int):
        return self.neglog_normal_pdf(x, self.mean[k], self.logvar[k])
    
    # 𝐿𝜽(𝑧𝑘) = log∑exp(𝐥𝐨𝐠𝜋𝑗)−(log𝜋𝑘)
    def loss_z(self, k: int):
        return tf.reduce_logsumexp(self.logpi) - self.logpi[k]
    
    # 𝐿𝜽(𝑥(𝑖)) = −log∑exp(−(𝐿𝜽(𝑥^(𝑖)|𝑧𝑘) + 𝐿𝜽(𝑧𝑘)))
    def loss_x(self, x: TFData):
        exp_arg = [(-1) * (self.loss_xz(x, k) + self.loss_z(k)) for k in range(self.K)]
        return (-1) * tf.reduce_logsumexp(exp_arg, axis=0) # Axis == 0 da ocuvamo (1000,1) jer smo vec u exp_arg sumirali po k
    
    def p_z(self, k: int) -> TFData:
        return self.pi[k]
    
    def p_xz(self, x: TFData, k: int) -> TFData:
        return self.normal_pdf(x, self.mean[k], self.var[k])

    def p_x(self, x: TFData) -> TFData:
        return tf.math.reduce_sum([self.pi[k] * self.p_xz(x, k) for k in range(self.K)], axis=0)
        
    
