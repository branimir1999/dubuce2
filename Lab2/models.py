from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Union

import tensorflow as tf
# noinspection PyUnresolvedReferences
import tensorflow.keras as K

from dists import Dist
from tf_utils import TFData, TFBuiltin

@dataclass
class NetworkConfiguration:
    input_dim: Union[tuple[int, ...], int]
    hidden_dims: Sequence[int]
    output_dim: Union[tuple[int, ...], int]
    activation_fn: TFBuiltin = None
    output_fn: TFBuiltin = None
    l1_decay: float = 0.0
    l2_decay: float = 0.0
    dropout_rate: float = None
    kernel_initializer: Optional[TFBuiltin] = None
    output_bias_initializer: Optional[TFBuiltin] = None

    def build(self, name: Optional[str] = None) -> K.Model:
        kernel_regularizer = K.regularizers.l1_l2(self.l1_decay, self.l2_decay)

        def layers_gen():
            yield K.layers.InputLayer(input_shape=self.input_dim, dtype=tf.float32)

            for units in self.hidden_dims:
                yield K.layers.Dense(units,
                                     activation=self.activation_fn,
                                     kernel_regularizer=kernel_regularizer,
                                     kernel_initializer=self.kernel_initializer)

                if self.dropout_rate != 0.0:
                    yield K.layers.Dropout(self.dropout_rate)

            yield K.layers.Dense(self.output_dim,
                                 activation=self.output_fn,
                                 kernel_regularizer=kernel_regularizer,
                                 kernel_initializer=self.kernel_initializer,
                                 bias_initializer=self.output_bias_initializer)

        return K.Sequential([layer for layer in layers_gen()], name=name)

@dataclass
class Coder(Callable):
    network: K.Model
    distribution: Dist

    def __call__(self, sample: TFData, **kwargs) -> Dist:
        outputs = self.network.call(sample, **kwargs)
        output_shapes = self.distribution.param_shapes
        params = tf.split(outputs, output_shapes, axis=-1)

        self.distribution.update_params(*params)

        return self.distribution
